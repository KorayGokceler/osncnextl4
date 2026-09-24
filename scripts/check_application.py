#!/usr/bin/env python
'''
Does the model score the same event the same way in the tray as in training?

Training reads a model's inputs out of HDF5 COLUMNS; the tray reads them out of
I3 FRAME objects (oscnext_l4/classifier.py) and writes the score back into the
frame.  The two paths name the same quantities through config/variables.json,
and varmap.check() confirms the NAMES agree -- but not that the frame read
finds anything.  It did not, for cog_z: the frame side named the tableio
column `cog_z` where the object's binding has `cog.z`, every event was scored
with that input missing, and nothing raised.

This is the test at the binding level.  Run process_L4.py with --apply-cut AND
--output-hdf5: the HDF5 then holds every model input as training reads it and
the score the tray computed from the frame (L4_HDF5_KEYS books both).  For each
model this recomputes the score from the COLUMNS -- resolved exactly as
training resolves them (data.py's REGISTRY/ALTS) and matched to their events
through the same __I3Index__ machinery -- and compares it, event by event,
with the booked one.

Identical inputs give bitwise identical scores (same booster, same float64
row), so the default tolerance is tight.  A frame read that finds nothing
shows up as a difference wherever the missing value takes another branch.

    python scripts/process_L4.py --gcd GCD --input L3.i3.zst \\
        --output-hdf5 check.hdf5 --apply-cut --model-dir models [flags]
    python scripts/check_application.py check.hdf5 --model-dir models

Exit status 0 only when every model was compared on at least one event and
none differs beyond --atol.  Needs numpy, pytables, lightgbm; no icetray.
'''

import os
import sys
import glob
import json
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                 # repo root, for `oscnext_l4`

from oscnext_l4.data import load_one_file          # noqa: E402

# The frame keys classifier.py writes the scores to.  Spelled out here because
# classifier.py imports icetray and this script must not need it; checked
# against its source below, so a rename there fails here instead of comparing
# nothing.
SCORE_KEYS = {"noise": "L4_NoiseClassifier_ProbNu",
              "muon": "L4_MuonClassifier_Data_ProbNu"}


def _check_score_keys():
    src = open(os.path.join(ROOT, "oscnext_l4", "classifier.py")).read()
    for tag, key in SCORE_KEYS.items():
        if '"%s"' % key not in src:
            sys.exit("classifier.py no longer writes %r (%s) -- update "
                     "SCORE_KEYS here." % (key, tag))


def load_model(model_dir, tag):
    """(booster, features) -- the .json's order, checked against the booster."""
    import lightgbm as lgb
    txt = os.path.join(model_dir, "L4_%s_model.txt" % tag)
    js = os.path.join(model_dir, "L4_%s_model.json" % tag)
    for f in (txt, js):
        if not os.path.exists(f):
            sys.exit("missing %s" % f)
    booster = lgb.Booster(model_file=txt)
    with open(js) as fh:
        features = list(json.load(fh)["features"])
    if list(booster.feature_name()) != features:
        sys.exit("%s: the .json's feature order %s is not the booster's %s"
                 % (tag, features, booster.feature_name()))
    return booster, features


def main():
    ap = argparse.ArgumentParser(
        description="compare the tray's booked classifier scores with the "
                    "scores recomputed from the booked HDF5 columns")
    ap.add_argument("hdf5", nargs="+",
                    help="HDF5 file(s) from process_L4.py --apply-cut "
                         "--output-hdf5 (glob patterns accepted)")
    ap.add_argument("--model-dir", required=True,
                    help="the models the tray ran with")
    ap.add_argument("--atol", type=float, default=1e-9,
                    help="largest acceptable |recomputed - booked| (default "
                         "1e-9: identical inputs give identical scores)")
    ap.add_argument("--show", type=int, default=5,
                    help="print this many of the worst events per model")
    args = ap.parse_args()

    _check_score_keys()
    files = []
    for pat in args.hdf5:
        files.extend(sorted(glob.glob(pat)) or [pat])
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        sys.exit("not found: %s" % ", ".join(missing))

    models = {tag: load_model(args.model_dir, tag) for tag in SCORE_KEYS}
    wanted = sorted({f for _, feats in models.values() for f in feats})
    extra = {"__score_%s" % tag: (key, "value") for tag, key in SCORE_KEYS.items()}

    totals = {tag: dict(n=0, compared=0, no_score=0, nan_input=0, bad=0,
                        maxdiff=0.0, worst=[]) for tag in SCORE_KEYS}
    for path in files:
        d = load_one_file(path, wanted, extra=extra)
        for tag, (booster, feats) in models.items():
            t = totals[tag]
            X = np.column_stack([np.asarray(d[f], dtype=np.float64) for f in feats])
            booked = np.asarray(d["__score_%s" % tag], dtype=np.float64)
            have = np.isfinite(booked)
            t["n"] += len(booked)
            t["no_score"] += int((~have).sum())
            t["nan_input"] += int((~np.isfinite(X)).any(axis=1).sum())
            if not have.any():
                continue
            diff = np.abs(booster.predict(X[have]) - booked[have])
            t["compared"] += int(have.sum())
            t["bad"] += int((diff > args.atol).sum())
            t["maxdiff"] = max(t["maxdiff"], float(diff.max()))
            idx = np.flatnonzero(have)
            for j in np.argsort(-diff)[:args.show]:
                if diff[j] > args.atol:
                    t["worst"].append((float(diff[j]), os.path.basename(path),
                                       int(d["Run"][idx[j]]), int(d["Event"][idx[j]]),
                                       int(d["SubEvent"][idx[j]])))

    ok = True
    print("\n%-6s %9s %9s %9s %9s %12s %9s"
          % ("model", "events", "compared", "no score", "NaN in", "max |diff|",
             "> atol"))
    for tag, t in totals.items():
        print("%-6s %9d %9d %9d %9d %12.3g %9d"
              % (tag, t["n"], t["compared"], t["no_score"], t["nan_input"],
                 t["maxdiff"], t["bad"]))
        if not t["compared"]:
            print("  [!] %s: no event carries a booked %s -- was the run made "
                  "with --apply-cut and --output-hdf5?" % (tag, SCORE_KEYS[tag]))
            ok = False
        if t["bad"]:
            ok = False
            print("  [!] %s: %d event(s) differ beyond %g.  The frame read and "
                  "the column read of some input disagree -- check the frame "
                  "side of each input in config/variables.json." % (tag, t["bad"], args.atol))
            for diff, f, r, e, s in sorted(t["worst"], reverse=True)[:args.show]:
                print("      |diff| %.3g  %s  Run %d Event %d SubEvent %d" % (diff, f, r, e, s))
    print("\n%s" % ("PASS: the tray scores every compared event exactly as "
                    "training would." if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
