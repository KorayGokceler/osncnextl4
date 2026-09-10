#!/usr/bin/env python
'''
Where does the efficiency gap come from?  Measurements that separate three
hypotheses.

Context: the first noise BDT keeps 53% of the signal at 99% background
rejection; the target from Table 13 of the technical note is ~96% at 99.2%.
At EQUAL SIGNAL EFFICIENCY our rejection is 67% against their 99.2% -- so the
problem is not a misplaced cut, it is SEPARATION POWER.

This script measures two hypotheses:

  A) DATA -- is the background statistics insufficient?
     Learning curve: train on 25/50/75/100% of the background, always
     evaluating on the SAME full test set.  If the curve is still climbing at
     100%, more data helps and the slope tells you roughly how much.  If it
     has flattened, data is not the constraint.

  C) VARIABLES -- is one of them computed wrong?
     Standalone separation power of each variable (both directions tried).
     A variable that separates nothing, or separates in the OPPOSITE
     direction from what is expected, shows up here.  Signal/background
     medians are printed too, for eyeball comparison against Fig 12/13.

  (B) ENGINE -- AdaBoost vs LightGBM -- is NOT in this script; that is a
     separate chain, via reference/train_L4_classifier.py.

CAVEAT: the efficiency-at-99%-rejection metric places its threshold using
roughly the top 10 background events, so the metric itself is noisy.  Read
the spread column before believing any trend.

Usage:

    python pybdt_diagnose.py --ds-dir L4_output/ds --tag L4_noise \
        --features NchCleaned,micro_count,iLineFit_speed,fill_ratio,FullTimeLengthRatio
'''

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from pybdt import ml, util
from pybdt_train import build_learner, RESERVED_COLS
from pybdt_scan import eff_at_rejection
from types import SimpleNamespace


def load_ds(ds_dir, tag):
    out = {}
    for part in ("sig_train", "bg_train", "sig_test", "bg_test"):
        p = os.path.join(ds_dir, "%s_%s.ds" % (tag, part))
        if not os.path.exists(p):
            sys.exit("not found: %s" % p)
        out[part] = util.load(p)
    return out


def wcol(ds):
    names = list(ds.names)
    key = "w_phys" if "w_phys" in names else "weight"
    return np.nan_to_num(np.asarray(ds[key]))


def single_variable_power(ds, features, target):
    '''
    Standalone power of each variable.

    A decision tree splits on "value >= threshold", but which SIDE is signal
    depends on the variable.  We try both directions and keep the better one;
    the winning direction is printed, because a direction that contradicts
    physical expectation means we compute the variable wrong.
    '''
    ws = wcol(ds["sig_test"])
    wb = wcol(ds["bg_test"])
    print("\n" + "=" * 78)
    print("C) STANDALONE POWER OF EACH VARIABLE  (efficiency at %.0f%% rejection)"
          % (100 * target))
    print("=" * 78)
    print("%-22s %8s %8s | %10s %10s | %s"
          % ("variable", "eff", "direction", "sig median", "bg median", "NaN"))
    print("-" * 78)
    rows = []
    for f in features:
        s = np.asarray(ds["sig_test"][f], dtype=float)
        b = np.asarray(ds["bg_test"][f], dtype=float)
        nan = int((~np.isfinite(s)).sum() + (~np.isfinite(b)).sum())
        best, best_dir = -1.0, "?"
        for sign, label in ((+1.0, "large=signal"), (-1.0, "small=signal")):
            _, eff, _ = eff_at_rejection(sign * s, ws, sign * b, wb, target)
            if np.isfinite(eff) and eff > best:
                best, best_dir = eff, label
        rows.append((f, best, best_dir))
        print("%-22s %7.1f%% %8s | %10.4g %10.4g | %d"
              % (f, 100 * best, best_dir,
                 np.nanmedian(s), np.nanmedian(b), nan))
    print("\n  Reading: around %.1f%% means the variable separates NOTHING on"
          % (100 * (1 - target)))
    print("  its own (that is what a random threshold would give).")
    print("  If the 'direction' column contradicts your expectation, we may be")
    print("  computing the variable wrong -- compare against Fig 12/13.")
    print("  CAVEAT: this is a ONE-SIDED threshold.  A variable whose noise")
    print("  sits in BOTH tails cannot be captured this way and will look")
    print("  dead here even when a tree could use it.")
    return rows


def learning_curve(ds, features, args, fracs=(0.25, 0.5, 0.75, 1.0)):
    '''
    Thin out the background; keep the signal and the TEST set fixed.

    The test set must not change, otherwise the points are not comparable.
    '''
    rng = np.random.default_rng(args.seed)
    names = list(ds["bg_train"].names)
    n_bg = len(ds["bg_train"][names[0]])
    ws = wcol(ds["sig_test"])
    wb = wcol(ds["bg_test"])

    a = SimpleNamespace(
        depth=args.depth, min_split=None, num_cuts=None,
        num_random_variables=None, nonlinear_cuts=False,
        num_trees=args.num_trees, beta=args.beta,
        frac_random_events=0.5, prune_strength=None, use_purity=True)

    print("\n" + "=" * 78)
    print("A) LEARNING CURVE  (background thinned, test set fixed)")
    print("=" * 78)
    print("  model: depth %d, %d trees, beta %g;  %d repeats per point"
          % (args.depth, args.num_trees, args.beta, args.repeat))
    print("\n%-12s %-10s | %-26s" % ("background", "events",
                                     "EFFICIENCY (mean +- half-spread)"))
    print("-" * 54)

    out = []
    for frac in fracs:
        k = max(2, int(round(frac * n_bg)))
        effs = []
        for _ in range(max(1, args.repeat)):
            idx = rng.choice(n_bg, size=k, replace=False)
            bg = ml.DataSet({c: np.asarray(ds["bg_train"][c])[idx] for c in names})
            learner = build_learner(features, args.weight_col, a)
            bdt = learner.train(ds["sig_train"], bg)
            s = np.asarray(bdt.score(ds["sig_test"], use_purity=True, quiet=True))
            b = np.asarray(bdt.score(ds["bg_test"], use_purity=True, quiet=True))
            _, eff, _ = eff_at_rejection(s, ws, b, wb, args.target_rejection)
            effs.append(eff)
        m, half = float(np.mean(effs)), (max(effs) - min(effs)) / 2
        out.append((frac, k, m, half))
        print("%-12s %-10d | %6.1f%%  +-%.1f"
              % ("%d%%" % (100 * frac), k, 100 * m, 100 * half))

    # Slope at the end of the curve -- only meaningful if it beats the spread.
    if len(out) >= 2:
        (f0, k0, e0, h0), (f1, k1, e1, h1) = out[-2], out[-1]
        d = 100 * (e1 - e0)
        noise = 100 * (h0 + h1)
        print("\n  Last step (%d -> %d events): %+.1f points, "
              "combined spread +-%.1f points" % (k0, k1, d, noise))
        if abs(d) < noise:
            print("  -> The step is SMALLER THAN THE SPREAD: this curve does not")
            print("     establish anything.  Raise --repeat, or use a more stable")
            print("     metric (--target-rejection 0.90 puts ~100 background")
            print("     events behind the threshold instead of ~10).")
        elif d > 0:
            print("  -> The curve is still climbing beyond the noise: background")
            print("     statistics is the binding constraint.")
        else:
            print("  -> The curve is flat or falling: more background data will")
            print("     not help this model.  The constraint is elsewhere.")
    return out


def signal_balance(ds, features, args):
    '''Thin the signal down toward the background size to probe the imbalance.'''
    rng = np.random.default_rng(args.seed + 1)
    names = list(ds["sig_train"].names)
    n_sig = len(ds["sig_train"][names[0]])
    n_bg = len(ds["bg_train"][list(ds["bg_train"].names)[0]])
    ws = wcol(ds["sig_test"])
    wb = wcol(ds["bg_test"])

    a = SimpleNamespace(
        depth=args.depth, min_split=None, num_cuts=None,
        num_random_variables=None, nonlinear_cuts=False,
        num_trees=args.num_trees, beta=args.beta,
        frac_random_events=0.5, prune_strength=None, use_purity=True)

    print("\n" + "=" * 78)
    print("EXTRA) CLASS IMBALANCE  (signal thinned, test set fixed)")
    print("=" * 78)
    print("%-16s %-12s | %-26s" % ("signal:background", "signal events",
                                   "EFFICIENCY (mean +- half-spread)"))
    print("-" * 58)
    for ratio in (1, 5, 20, None):
        k = n_sig if ratio is None else min(n_sig, ratio * n_bg)
        effs = []
        for _ in range(max(1, args.repeat)):
            idx = rng.choice(n_sig, size=k, replace=False)
            sig = ml.DataSet({c: np.asarray(ds["sig_train"][c])[idx] for c in names})
            learner = build_learner(features, args.weight_col, a)
            bdt = learner.train(sig, ds["bg_train"])
            s = np.asarray(bdt.score(ds["sig_test"], use_purity=True, quiet=True))
            b = np.asarray(bdt.score(ds["bg_test"], use_purity=True, quiet=True))
            _, eff, _ = eff_at_rejection(s, ws, b, wb, args.target_rejection)
            effs.append(eff)
        label = ("full (%d:1)" % (n_sig // max(n_bg, 1))) if ratio is None \
            else ("%d:1" % ratio)
        print("%-16s %-12d | %6.1f%%  +-%.1f"
              % (label, k, 100 * np.mean(effs), 100 * (max(effs) - min(effs)) / 2))
    print("\n  If efficiency rises CLEARLY (beyond the spread) as the classes")
    print("  balance, the extreme imbalance is hurting AdaBoost.  If it does")
    print("  not move, the imbalance is not the problem.")


def main():
    ap = argparse.ArgumentParser(
        description="Diagnose the noise BDT efficiency gap")
    ap.add_argument("--ds-dir", required=True)
    ap.add_argument("--tag", default="L4_noise")
    ap.add_argument("--features", default=None)
    ap.add_argument("--weight-col", default="weight")
    ap.add_argument("--target-rejection", type=float, default=0.99,
                    help="0.90 gives a far more stable metric than 0.99: the "
                         "threshold then sits behind ~100 background events "
                         "instead of ~10")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--num-trees", type=int, default=500)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--skip-balance", action="store_true")
    args = ap.parse_args()

    ds = load_ds(args.ds_dir, args.tag)
    if args.features:
        features = [f.strip() for f in args.features.split(",") if f.strip()]
    else:
        features = [n for n in ds["sig_train"].names
                    if n not in RESERVED_COLS and n != args.weight_col]

    n_sig = len(ds["sig_train"][features[0]])
    n_bg = len(ds["bg_train"][features[0]])
    print("=== %s diagnosis ===" % args.tag)
    print("  train  signal %8d   background %6d   (ratio %d:1)"
          % (n_sig, n_bg, n_sig // max(n_bg, 1)))
    print("  test   signal %8d   background %6d"
          % (len(ds["sig_test"][features[0]]), len(ds["bg_test"][features[0]])))
    print("  target: signal efficiency at %.1f%% background rejection"
          % (100 * args.target_rejection))

    single_variable_power(ds, features, args.target_rejection)
    learning_curve(ds, features, args)
    if not args.skip_balance:
        signal_balance(ds, features, args)

    print("\n" + "=" * 78)
    print("The engine hypothesis (AdaBoost vs LightGBM) is NOT covered here --")
    print("compare against reference/train_L4_classifier.py with the same 5")
    print("variables.")
    print("=" * 78)


if __name__ == "__main__":
    main()
