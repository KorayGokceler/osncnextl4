#!/usr/bin/env python
'''
Booked L4 HDF5 -> the .npz training set of ONE classifier, without the notebook.

    process_L4.py --output-hdf5 ...  -->  make_dataset.py  -->  train_L4_classifier.py

It does what notebook sections 1, 4, 5 and 6 do, in that order, for one stage:

  1  pick the production in config/productions.json, keep the samples whose
     ROLE the stage needs, and set the noise weight unit -- the one fact that
     fails SILENTLY (a wrong unit scales every noise rate by 1e9)
  4  load every sample's HDF5 parts
  5  weight them (w_phys); detector data by 1/livetime, the livetime from the
     Good Run List when it covers every run and from the event times otherwise
  6  stack the signal, draw the split, and write L4_<stage>_dataset.npz

The functions are the notebook's own (oscnext_l4/dataset.py), and so is the
order of the random draws: the signal split is the FIRST draw of
default_rng(12345), the noise background the second, the muon background its
own default_rng(12346).  That is what makes the noise and the muon set carry
the SAME signal split when they are made by two separate runs -- open risk 5c:
the L4 cut is the conjunction of the two classifiers, so they need a common
held-out set.

THE ORDER OF THE STAGES IS THE PRODUCTION'S: noise first, train it, then the
muon set is built only from events with P_noise >= 0.70 (note sec. 3.6.3,
open risk 5i).  --stage muon therefore needs --noise-model, and refuses to run
without it unless --no-noise-cut says the deviation out loud.

Usage:

    python scripts/make_dataset.py --stage noise --production pass2 \\
        --hdf-base /data/user/$USER/L4_output/hdf5_pass2 \\
        --outdir   /data/user/$USER/L4_output/ds_pass2
    python scripts/train_L4_classifier.py --tag noise \\
        --dataset /data/user/$USER/L4_output/ds_pass2/L4_noise_dataset.npz \\
        --outdir  models
    python scripts/make_dataset.py --stage muon --production pass2 \\
        --hdf-base ... --outdir ... --noise-model models/L4_noise_model.txt

The HDF5 layout is the notebook's: <hdf-base>/<sample>/L4_<sample>*.hdf5, each
with its .meta.json (process_L4.py writes both; the meta file carries the L3
file count the weights are divided by).

The .npz also carries a "provenance" record -- production, samples, weighting,
noise cut -- that train_L4_classifier.py copies into the model's .json.

DEPENDENCIES: numpy, pytables; lightgbm for --stage muon.  No icetray.
'''

import os
import sys
import glob
import json
import argparse
import datetime
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                 # repo root, for `oscnext_l4`

from oscnext_l4 import data as l4data                          # noqa: E402
from oscnext_l4.data import (WANTED, NOISE_FEATURES, MUON_FEATURES,  # noqa: E402
                             load_sample, add_weights, scheme_of,
                             set_noise_weight_unit, set_data_livetime,
                             livetime_from_headers, livetime_from_grl)
from oscnext_l4.dataset import (RNG_SEED, TRAIN_FRAC, NOISE_CUT,  # noqa: E402
                                stack, report_missing_inputs, split_by_shower,
                                build_dataset, load_noise_prob, sha256_of)


def _code_version():
    """`git describe` of this checkout, or None outside one."""
    try:
        out = subprocess.run(["git", "-C", ROOT, "describe", "--always",
                              "--dirty"], capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Section 1: the production, the samples, the noise weight unit
# ---------------------------------------------------------------------------

def select_samples(config, production, stage, hdf_base):
    """
    SAMPLES for this stage, built exactly as notebook section 1 builds them:
    the production's samples whose ROLE the stage needs, each pointed at
    <hdf_base>/<name>/L4_<name>.hdf5.  Returns (prod, SAMPLES).
    """
    if production not in config["productions"]:
        sys.exit("production %r is not in the config; it has: %s"
                 % (production, ", ".join(sorted(config["productions"]))))
    prod = config["productions"][production]
    roles = tuple(config["bdt_roles"][stage])
    cleaned = prod.get("cleaned_pulses")

    samples = {}
    for name, spec in prod["samples"].items():
        if spec["kind"] not in roles:
            continue
        cfg = {k: v for k, v in spec.items() if not k.startswith("__")}
        cfg["flags"] = list(cfg["flags"])
        if cleaned:
            cfg["flags"] += ["--cleaned-pulses", cleaned]
        cfg["hdf5"] = os.path.join(hdf_base, name, "L4_%s.hdf5" % name)
        samples[name] = cfg

    print("production     : %s" % production)
    print("  stage        : %s  (roles: %s)" % (stage, ", ".join(roles)))
    print("  HDF5 base    : %s\n" % hdf_base)
    # The L3 file count is the fallback weight divisor when a part has no
    # meta.json; the notebook takes it from the same glob.
    for name, cfg in samples.items():
        pats = cfg["l3"] if isinstance(cfg["l3"], list) else [cfg["l3"]]
        cfg["n_l3_files"] = sum(len(glob.glob(p)) for p in pats)
        print("  %-8s %-10s %6d L3 files" % (name, cfg["kind"], cfg["n_l3_files"]))
    return prod, samples


# ---------------------------------------------------------------------------
# Section 5: detector-data livetime
# ---------------------------------------------------------------------------

def set_livetime(data, samples, grl_patterns):
    """Notebook section 5's livetime block -> {"source", "seconds"} or None."""
    record = None
    for name in [n for n in data if scheme_of(samples[n]) == "data"]:
        print("=== livetime: from the event times (%s) ===" % name)
        t_hdr, _ = livetime_from_headers(data[name]["_files"])
        print("\n=== livetime: from the Good Run List ===")
        runs = np.unique(data[name]["Run"])
        t_grl, found = livetime_from_grl(runs, patterns=grl_patterns)
        if len(found) == len(runs):
            print("\n  GRL / event-time span = %.4f"
                  % (t_grl / t_hdr if t_hdr else float("nan")))
            set_data_livetime(t_grl)
            record = {"source": "good_run_list", "seconds": t_grl,
                      "patterns": list(grl_patterns)}
        else:
            print("\n  [!] the GRL does not cover every run -- falling back to "
                  "the event-time span.")
            set_data_livetime(t_hdr)
            record = {"source": "event_time_span", "seconds": t_hdr,
                      "patterns": list(grl_patterns)}
    return record


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _sample_record(name, cfg, d):
    pats = cfg["l3"] if isinstance(cfg["l3"], list) else [cfg["l3"]]
    return {"name": name, "kind": cfg["kind"], "weight": scheme_of(cfg),
            "l3": pats, "n_events": int(len(d["Run"])),
            "n_hdf5": int(d["_n_hdf5"]), "n_l3_files": float(d["_n_files"])}


def provenance(args, config_path, prod, samples, data, signal, background,
               livetime, noise_cut):
    """What made this training set, for the model's sidecar.  Facts only."""
    schemes = sorted({scheme_of(samples[n]) for n in signal})
    weighting = {"schemes": schemes}
    if "genie" in schemes:
        # The single power law genie_weight applies -- recorded as the
        # numbers it used, so a change to data.py shows up here.
        weighting.update(norm=l4data.NORM, gamma=l4data.GAMMA,
                         nu_frac=l4data.NU_FRAC, nubar_frac=l4data.NUBAR_FRAC)
    return {
        "schema": 1,
        "made_by": "scripts/make_dataset.py",
        "made_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "code_version": _code_version(),
        "config": os.path.abspath(config_path),
        "config_sha256": sha256_of(config_path),
        "production": args.production,
        "stage": args.stage,
        "features": list(NOISE_FEATURES if args.stage == "noise"
                         else MUON_FEATURES),
        "cleaned_pulses": prod.get("cleaned_pulses"),
        "rng_seed": RNG_SEED,
        "train_frac": TRAIN_FRAC,
        "signal": [_sample_record(n, samples[n], data[n]) for n in signal],
        "background": [_sample_record(n, samples[n], data[n])
                       for n in background],
        "signal_weighting": weighting,
        "noise_weight_unit": l4data.NOISE_WEIGHT_UNIT,
        "data_livetime": livetime,
        "noise_cut": noise_cut,
    }


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="booked L4 HDF5 -> L4_<stage>_dataset.npz (notebook "
                    "sections 1, 4, 5, 6)")
    ap.add_argument("--stage", required=True, choices=("noise", "muon"),
                    help="which classifier's set: noise first, muon after the "
                         "noise model is trained")
    ap.add_argument("--production", required=True,
                    help="a production in the config: pass2 or pass3")
    ap.add_argument("--hdf-base", required=True,
                    help="the HDF5 tree: <hdf-base>/<sample>/L4_<sample>*.hdf5")
    ap.add_argument("--outdir", required=True,
                    help="where L4_<stage>_dataset.npz is written")
    ap.add_argument("--config", default=None,
                    help="productions.json (default: $OSCNEXT_L4_CONFIG, else "
                         "config/productions.json beside this package)")
    ap.add_argument("--noise-model", default=None,
                    help="--stage muon: the trained L4_noise_model.txt; only "
                         "events with P_noise >= %.2f enter the muon set"
                         % NOISE_CUT)
    ap.add_argument("--no-noise-cut", action="store_true",
                    help="--stage muon WITHOUT the noise cut.  Not what the "
                         "production trained on; recorded in the provenance.")
    ap.add_argument("--grl-pattern", action="append", default=None,
                    # argparse %-formats help text, so every % in it -- the
                    # patterns' own %(y)d included -- must arrive as %%, or
                    # --help itself dies with KeyError.  Concatenated, not
                    # %-formatted here, so nothing unescapes them first.
                    help="Good Run List path pattern with %%(y)d for the year, "
                         "tried in order; repeatable.  Default: "
                         + " then ".join(l4data.GRL_PATTERNS).replace("%", "%%"))
    args = ap.parse_args()

    if args.stage == "muon":
        if args.noise_model and args.no_noise_cut:
            ap.error("--noise-model and --no-noise-cut contradict each other.")
        if not args.noise_model and not args.no_noise_cut:
            ap.error("--stage muon needs --noise-model: the production trains "
                     "the muon BDT only on events that pass the noise cut.  "
                     "Pass --no-noise-cut to build it on the uncut population "
                     "anyway (a real deviation, recorded as such).")
        if args.noise_model and not os.path.exists(args.noise_model):
            ap.error("no noise model at %s" % args.noise_model)
    elif args.noise_model or args.no_noise_cut:
        ap.error("--noise-model / --no-noise-cut belong to --stage muon.")

    config_path = (args.config or os.environ.get("OSCNEXT_L4_CONFIG")
                   or os.path.join(ROOT, "config", "productions.json"))
    with open(config_path) as fh:
        config = json.load(fh)
    print("production config: %s" % config_path)
    grl = tuple(args.grl_pattern) if args.grl_pattern else l4data.GRL_PATTERNS

    # --- section 1 -----------------------------------------------------------
    prod, samples = select_samples(config, args.production, args.stage,
                                   args.hdf_base)
    if "noise_weight_unit" not in prod:
        sys.exit("production %r declares no noise_weight_unit -- refusing to "
                 "guess: a wrong one scales every noise rate by 1e9."
                 % args.production)
    # BEFORE any weight is computed, as in the notebook's section 1 cell.
    set_noise_weight_unit(prod["noise_weight_unit"])

    # --- section 4 -----------------------------------------------------------
    data = {}
    for name in samples:
        d = load_sample(name, samples, WANTED)
        if d is not None:
            data[name] = d
    print("\nLoaded:", {k: len(v["Run"]) for k, v in data.items()})
    missing = [n for n in samples if n not in data]
    if missing:
        print("[!] not loaded:", ", ".join(missing))

    # --- section 5 -----------------------------------------------------------
    livetime = set_livetime(data, samples, grl)
    add_weights(data, samples)

    # --- section 6 -----------------------------------------------------------
    all_features = sorted(set(NOISE_FEATURES) | set(MUON_FEATURES))
    signal = [n for n, c in samples.items()
              if c.get("kind") == "signal" and n in data]
    if not signal:
        sys.exit("no signal sample loaded -- nothing to train on.")
    print("signal sets: %s" % ", ".join(signal))

    rng = np.random.default_rng(RNG_SEED)
    sig = stack(data, signal, all_features)
    sig_istrain = rng.random(len(sig["w_phys"])) < TRAIN_FRAC       # draw 1
    print("signal: %d events, train %d / test %d  (BOTH classifiers use this "
          "split)" % (len(sig_istrain), sig_istrain.sum(), (~sig_istrain).sum()))
    os.makedirs(args.outdir, exist_ok=True)

    if args.stage == "noise":
        noise_bg = [n for n, c in samples.items()
                    if c.get("kind") == "noise_bg" and n in data]
        if not noise_bg:
            sys.exit("no noise background loaded -- nothing to build.")
        print("=== noise BDT  (signal = %s, background = %s) ==="
              % ("+".join(signal), "+".join(noise_bg)))
        bg = stack(data, noise_bg, all_features)
        report_missing_inputs(sig, NOISE_FEATURES, "sig")
        report_missing_inputs(bg, NOISE_FEATURES, "bg")
        # vuvuzela has no oversampling -> an event-level split is fine
        bg_istrain = rng.random(len(bg["w_phys"])) < TRAIN_FRAC     # draw 2
        prov = provenance(args, config_path, prod, samples, data, signal,
                          noise_bg, livetime, None)
        build_dataset("noise", sig, bg, NOISE_FEATURES, sig_istrain,
                      bg_istrain, args.outdir, provenance=prov)
        return

    # --- muon ------------------------------------------------------------------
    muon_bg = [n for n, c in samples.items()
               if c.get("kind") == "muon_bg" and n in data]
    if not muon_bg:
        sys.exit("no muon background loaded -- nothing to build.  pass3 has "
                 "'corsika', pass2 'muongun' and 'data'.")
    # NEVER STACK SEVERAL MUON BACKGROUNDS: a model fitted on MuonGun + data
    # learns the difference between simulation and measurement.  Detector data
    # is the production's background, so it wins (notebook section 6).
    if len(muon_bg) > 1:
        pref = [n for n in muon_bg if scheme_of(samples[n]) == "data"]
        print("  [!] %d muon backgrounds loaded (%s) -- they are NOT stacked."
              % (len(muon_bg), ", ".join(muon_bg)))
        muon_bg = pref[:1] or muon_bg[:1]
        print("      using %r (the production's background is detector data)."
              % muon_bg[0])
    print("\n=== muon BDT  (signal = %s, background = %s) ==="
          % ("+".join(signal), muon_bg[0]))
    bg = stack(data, muon_bg, all_features)

    # The noise cut, on BOTH classes.  The signal is masked together with its
    # train/test flag, so a surviving event keeps the side it was already on
    # -- which is what keeps the split shared with the noise set.
    sig_m, sig_m_istrain = sig, sig_istrain
    if args.noise_model:
        noise_prob, nfeat = load_noise_prob(args.noise_model)
        print("noise model : %s" % args.noise_model)
        print("  features  : %s" % ", ".join(nfeat))
        ks = noise_prob(sig) >= NOISE_CUT
        kb = noise_prob(bg) >= NOISE_CUT
        print("  noise cut P(nu) >= %.2f" % NOISE_CUT)
        print("    signal     %8d -> %8d  (%.1f%% kept)"
              % (len(ks), int(ks.sum()), 100.0 * ks.mean()))
        print("    background %8d -> %8d  (%.1f%% kept)"
              % (len(kb), int(kb.sum()), 100.0 * kb.mean()))
        sig_m = {k: v[ks] for k, v in sig.items()}
        sig_m_istrain = sig_istrain[ks]
        bg = {k: v[kb] for k, v in bg.items()}
        if not kb.sum():
            sys.exit("the noise cut removed EVERY background event.")
        noise_cut = {"applied": True, "threshold": NOISE_CUT,
                     "model": os.path.abspath(args.noise_model),
                     "model_sha256": sha256_of(args.noise_model),
                     "signal_kept": int(ks.sum()), "signal_total": len(ks),
                     "background_kept": int(kb.sum()),
                     "background_total": len(kb)}
    else:
        print("  [!] NO NOISE CUT APPLIED (--no-noise-cut) -- the muon BDT is "
              "trained on a population the production's never saw.")
        noise_cut = {"applied": False, "threshold": None, "model": None,
                     "model_sha256": None}

    report_missing_inputs(sig_m, MUON_FEATURES, "sig")
    report_missing_inputs(bg, MUON_FEATURES, "bg")
    # Split by Run -- for CORSIKA that is the shower -- with its OWN generator,
    # so this draw does not depend on whether the noise half ever ran.
    bg_istrain = split_by_shower(bg["Run"], TRAIN_FRAC,
                                 np.random.default_rng(RNG_SEED + 1))
    print("  %s: %d distinct Run values, %d events"
          % (muon_bg[0], len(np.unique(bg["Run"])), len(bg["Run"])))
    prov = provenance(args, config_path, prod, samples, data, signal,
                      muon_bg, livetime, noise_cut)
    build_dataset("muon", sig_m, bg, MUON_FEATURES, sig_m_istrain, bg_istrain,
                  args.outdir, provenance=prov)


if __name__ == "__main__":
    main()
