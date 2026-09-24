"""
L4 training sets: loaded samples -> one .npz per classifier.

WHY A SEPARATE FILE: these functions were defined in notebook section 6, so a
training set could only be made by running the notebook.  They live here now,
and the notebook and scripts/make_dataset.py call the same code.  The logic
is the notebook's, line for line; what changed is that what the cells read as
globals -- `data`, `DS_BASE`, the loaded booster -- is passed in.

numpy only at import time; lightgbm is imported when a noise model is loaded,
so this module imports anywhere the HDF5 side does.
"""

import os
import json
import hashlib

import numpy as np


# The three numbers a training set is drawn with.  One definition, read by
# the notebook and by make_dataset.py: a signal split that is to be SHARED by
# the two classifiers (CLAUDE.md open risk 5c) is only shared if both are drawn
# from the same seed, the same fraction and the same order of draws.
#
#   RNG_SEED    the signal split is the FIRST draw of default_rng(RNG_SEED),
#               the noise background the second; the muon background has its
#               own default_rng(RNG_SEED + 1), so whether the noise half ran
#               cannot change it.
#   TRAIN_FRAC  0.5 on ~330k signal gives the production's regime of ~2e5
#               training events -- do NOT copy their 0.02, which targets an
#               absolute size on a far larger sample (CLAUDE.md, "Five real
#               differences", item 4).
#   NOISE_CUT   the production's threshold on P_noise before the muon BDT is
#               trained (note sec. 3.6.3; open risk 5i).
RNG_SEED = 12345
TRAIN_FRAC = 0.5
NOISE_CUT = 0.70


def stack(data, samples, features):
    """Concatenate several samples into one {name: array} dict."""
    out = {f: np.concatenate([data[s][f] for s in samples]) for f in features}
    for extra in ("w_phys", "Run"):
        out[extra] = np.concatenate([data[s][extra] for s in samples])
    return out


def report_missing_inputs(d, features, label):
    """
    Report events where at least one input is not finite.

    LightGBM routes missing values itself, so this is NOT a filter -- but we
    want to see how many events are affected and through which variable.  A
    variable that is 100% NaN stands out immediately here.
    """
    n = len(d["w_phys"])
    ok = np.ones(n, dtype=bool)
    for f in features:
        ok &= np.isfinite(np.asarray(d[f], dtype=np.float64))
    n_bad = int((~ok).sum())
    if n_bad:
        print("  [i] %-3s %d / %d events have a NaN input (%.2f%%)"
              % (label, n_bad, n, 100.0 * n_bad / n))
        for f in features:
            nf = int((~np.isfinite(np.asarray(d[f], dtype=np.float64))).sum())
            if nf:
                print("      %-22s %d" % (f, nf))
    return ok


def split_by_shower(runs, frac, rng_):
    """
    Train/test split at SHOWER level.

    In CORSIKA the same air shower is repeated OverSampling times; an
    event-level split puts copies of one shower on both sides and inflates the
    test efficiency.  Events sharing a `Run` always move together.
    """
    uniq = np.unique(runs)
    # A degenerate split is the danger here, and it is SILENT.  oscNext's
    # FixSimEventHeaders writes run_id = dataset_id, so a sample whose events
    # all come from one dataset has ONE distinct Run -- and this function would
    # then flip a single coin and send every event to train or every event to
    # test.  CORSIKA is safe (Run is the shower), MuonGun may not be.
    if len(uniq) < 20:
        print("  [!] split_by_shower: only %d distinct Run value(s) for %d "
              "events." % (len(uniq), len(runs)))
        print("      Too few to split on -- falling back to an EVENT-level "
              "split.  That is correct when the sample has no oversampling "
              "(MuonGun), and would leak copies if it does (CORSIKA).")
        return rng_.random(len(runs)) < frac
    keep = set(uniq[rng_.random(len(uniq)) < frac].tolist())
    return np.array([r in keep for r in runs], dtype=bool)


def build_dataset(tag, sig, bg, features, sig_istrain, bg_istrain, outdir,
                  provenance=None):
    """
    Write one .npz for one classifier -> its path.

    `provenance`, when given, is stored as a JSON string under the key
    "provenance", and train_L4_classifier.py copies it into the model's
    sidecar: which production, samples, weighting and noise cut made this
    set.  The notebook passes none, so its .npz files are what they were.
    """
    ws, wb = sig["w_phys"].copy(), bg["w_phys"].copy()
    for w in (ws, wb):
        w[~np.isfinite(w) | (w < 0)] = 0.0
    # equalise the class sums, then rescale into [0, 1]
    if ws.sum() > 0: ws /= ws.sum()
    if wb.sum() > 0: wb /= wb.sum()
    scale = max(ws.max(), wb.max())
    if scale > 0:
        ws /= scale; wb /= scale

    cols = {f: np.concatenate([np.asarray(sig[f], dtype=np.float64),
                               np.asarray(bg[f],  dtype=np.float64)])
            for f in features}
    cols["label"]   = np.r_[np.ones(len(ws)), np.zeros(len(wb))]
    cols["weight"]  = np.r_[ws, wb]
    cols["w_phys"]  = np.r_[np.asarray(sig["w_phys"], dtype=np.float64),
                            np.asarray(bg["w_phys"],  dtype=np.float64)]
    cols["istrain"] = np.r_[sig_istrain, bg_istrain]
    cols["features"] = np.array(features)
    if provenance is not None:
        cols["provenance"] = np.array(json.dumps(provenance, sort_keys=True))

    path = os.path.join(outdir, "L4_%s_dataset.npz" % tag)
    np.savez_compressed(path, **cols)
    print("  signal     train %7d / test %7d" % (sig_istrain.sum(), (~sig_istrain).sum()))
    print("  background train %7d / test %7d" % (bg_istrain.sum(), (~bg_istrain).sum()))
    print("  -> %s" % path)
    return path


def load_noise_prob(model_file):
    """
    The trained noise model as a function -> (noise_prob, features).

    noise_prob(d) is P(neutrino) for a stacked dict, read in the BOOSTER's own
    feature order, not NOISE_FEATURES: if the two ever disagreed, indexing by
    our list would silently feed the columns in the wrong order and the scores
    would be wrong without any error.
    """
    import lightgbm as lgb

    booster = lgb.Booster(model_file=model_file)
    features = list(booster.feature_name())

    def noise_prob(d):
        """P(neutrino) from the trained noise model, for a stacked dict."""
        X = np.column_stack([np.asarray(d[f], dtype=np.float64)
                             for f in features])
        return booster.predict(X)

    return noise_prob, features


def sha256_of(path):
    """Hex sha256 of a file -- what identifies a model in the provenance."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
