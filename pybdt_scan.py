#!/usr/bin/env python
'''
Hyperparameter scan for the pybdt/AdaBoost oscNext L4 classifiers.

WHY A SEPARATE SCRIPT:
  pybdt_train.py trains ONE model, validates it, plots it and saves it.
  Scanning is a different job: train many models and pick among them.  The
  learner setup is NOT duplicated -- build_learner / score_expr come
  straight from pybdt_train, so there is still a single implementation.

WHAT MAKES IT FAST:
  The .ds files are read ONCE.  Training uses the full data, but the metrics
  are computed on a SUBSAMPLE of the signal (--max-sig, default 20000):
  scoring 165k events for every configuration was what set the wall time,
  and 20k events measure the efficiency to ~0.3%.

THE METRIC:
  A single number: SIGNAL EFFICIENCY AT A TARGET BACKGROUND REJECTION
  (--target-rejection, default 0.99 -- Table 13 of the note quotes 99.24%
  for noise).
  Overtraining filter: two-sample KS test on the train/test score
  distributions.  Configurations with p_KS < --ks-min are REJECTED.

  The KS test used here is pybdt's OWN WEIGHTED, binned test, imported from
  pybdt.validate -- the same statistic pybdt_train.py reports.  (An earlier
  version used scipy's unweighted ks_2samp; a configuration that passed the
  scan then came out OVERTRAINED in pybdt_train.  Different statistic,
  different answer.)

  Training is STOCHASTIC (frac_random_events, and pybdt exposes no seed), so
  a single run does not reproduce.  --repeat trains each configuration
  several times: rejection uses the WORST p_KS, ranking uses the MEAN
  efficiency, and the spread between repeats is printed.  If that spread is
  large, the scan is measuring randomness rather than hyperparameters.

SELECTION BIAS -- IMPORTANT:
  Configurations are selected on the test set, so the efficiency the scan
  prints is an optimistic estimate.  We do not have enough background events
  to afford a third split (see CLAUDE.md 5e).  Treat the number as an upper
  bound; the real one settles once more background has been processed.

Usage:

    python pybdt_scan.py --ds-dir L4_output/ds --tag L4_noise \
        --features NchCleaned,micro_count,iLineFit_speed,fill_ratio,FullTimeLengthRatio

    # custom grid
    python pybdt_scan.py ... --depth 2,3,4 --num-trees 200,500 \
        --min-split 20,100 --prune-strength none,10 --beta 0.5
'''

import os
import sys
import time
import argparse
import itertools
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from pybdt import ml, util
# Use the SAME statistic as pybdt_train.py: pybdt's KS is weighted and
# binned, scipy's ks_2samp is neither.  They do not agree.
from pybdt.validate import kolmogorov_smirnov_probability
from pybdt_train import build_learner, effective_params, score_expr, RESERVED_COLS


def parse_list(text, cast):
    '''"2,3,4" -> [2,3,4];  "none,10" -> [None, 10.0]'''
    out = []
    for piece in str(text).split(","):
        piece = piece.strip()
        out.append(None if piece.lower() in ("none", "-", "") else cast(piece))
    return out


def subsample(ds, n, rng):
    '''Take at most n events from a DataSet (for metric computation only).'''
    names = list(ds.names)
    total = len(ds[names[0]])
    if n <= 0 or total <= n:
        return ds
    idx = rng.choice(total, size=n, replace=False)
    return ml.DataSet({k: np.asarray(ds[k])[idx] for k in names})


def eff_at_rejection(s, ws, b, wb, target):
    '''
    Find the threshold that rejects `target` of the background and return the
    signal efficiency there.

    The threshold comes from a quantile of the background WEIGHT, not of the
    event count, because the rate comparison is weighted.
    '''
    if wb.sum() <= 0 or ws.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(b)
    cw = np.cumsum(wb[order]) / wb.sum()
    k = int(np.searchsorted(cw, target))
    k = min(k, len(b) - 1)
    cut = float(b[order][k])
    eff = float(ws[s >= cut].sum() / ws.sum())
    rej = float(1.0 - wb[b >= cut].sum() / wb.sum())
    return cut, eff, rej


def main():
    ap = argparse.ArgumentParser(description="pybdt hyperparameter scan")
    ap.add_argument("--ds-dir", required=True, help="directory holding the .ds files")
    ap.add_argument("--tag", default="L4_noise",
                    help="prefix of <tag>_{sig,bg}_{train,test}.ds")
    ap.add_argument("--features", default=None,
                    help="comma separated; defaults to everything but RESERVED_COLS")
    ap.add_argument("--weight-col", default="weight")

    ap.add_argument("--target-rejection", type=float, default=0.99)
    ap.add_argument("--ks-min", type=float, default=0.01,
                    help="configurations with p_KS below this are rejected")
    ap.add_argument("--max-sig", type=int, default=20000,
                    help="signal events used for the metrics (0=all).  CAREFUL: "
                         "fewer events weaken the KS test, so overtraining can "
                         "go UNDETECTED.  Use 0 for the final pass.")
    ap.add_argument("--repeat", type=int, default=3,
                    help="how many times to train each configuration.  pybdt's "
                         "training is random (frac_random_events) and takes no "
                         "seed, so a single run does not reproduce.  Rejection "
                         "uses the WORST p_KS, ranking the MEAN efficiency.")
    ap.add_argument("--seed", type=int, default=12345)

    # grid
    ap.add_argument("--depth", default="2,3,4")
    ap.add_argument("--num-trees", default="200,500")
    ap.add_argument("--min-split", default="20,100")
    ap.add_argument("--prune-strength", default="none,10")
    ap.add_argument("--beta", default="0.5")
    ap.add_argument("--frac-random-events", default="0.5")
    ap.add_argument("--num-cuts", default="none")
    ap.add_argument("--no-purity", action="store_true",
                    help="turn use_purity OFF (it is on by default)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # --- data: read ONCE ----------------------------------------------------
    ds = {}
    for part in ("sig_train", "bg_train", "sig_test", "bg_test"):
        p = os.path.join(args.ds_dir, "%s_%s.ds" % (args.tag, part))
        if not os.path.exists(p):
            sys.exit("not found: %s" % p)
        ds[part] = util.load(p)

    if args.features:
        features = [f.strip() for f in args.features.split(",") if f.strip()]
    else:
        features = [n for n in ds["sig_train"].names
                    if n not in RESERVED_COLS and n != args.weight_col]

    n_sig = len(ds["sig_train"][features[0]])
    n_bg = len(ds["bg_train"][features[0]])
    print("=== %s hyperparameter scan ===" % args.tag)
    print("  train  signal %8d   background %6d" % (n_sig, n_bg))
    print("  test   signal %8d   background %6d"
          % (len(ds["sig_test"][features[0]]), len(ds["bg_test"][features[0]])))
    print("  %d variables: %s" % (len(features), ", ".join(features)))
    print("  metric: signal efficiency at %.1f%% background rejection;  p_KS >= %.3f"
          % (100 * args.target_rejection, args.ks_min))

    # Subsample for the metrics; the background is small already, leave it be.
    m = {"sig_train": subsample(ds["sig_train"], args.max_sig, rng),
         "sig_test":  subsample(ds["sig_test"],  args.max_sig, rng),
         "bg_train":  ds["bg_train"],
         "bg_test":   ds["bg_test"]}
    if args.max_sig and n_sig > args.max_sig:
        print("  (metrics from %d signal events -- training uses all of them)"
              % args.max_sig)

    w = {k: np.asarray(m[k]["w_phys"]) if "w_phys" in m[k].names
         else np.asarray(m[k][args.weight_col]) for k in m}
    for k in w:
        w[k] = np.nan_to_num(w[k])

    # --- grid ---------------------------------------------------------------
    grid = list(itertools.product(
        parse_list(args.depth, int),
        parse_list(args.num_trees, int),
        parse_list(args.min_split, int),
        parse_list(args.prune_strength, float),
        parse_list(args.beta, float),
        parse_list(args.frac_random_events, float),
        parse_list(args.num_cuts, int)))
    print("  %d configurations x %d repeats = %d trainings\n"
          % (len(grid), args.repeat, len(grid) * args.repeat))

    hdr = ("%-6s %-7s %-6s %-7s %-5s | %-8s %-8s | %-9s %-9s"
           % ("depth", "trees", "split", "prune", "beta",
              "p_KS sig", "p_KS bg", "cut", "EFF +-half-spread"))
    print(hdr)
    print("-" * len(hdr))

    rows, t0 = [], time.time()

    for depth, trees, split, prune, beta, frac, ncuts in grid:
        a = SimpleNamespace(
            depth=depth, min_split=split, num_cuts=ncuts,
            num_random_variables=None, nonlinear_cuts=False,
            num_trees=trees, beta=beta, frac_random_events=frac,
            prune_strength=prune, use_purity=not args.no_purity)

        ps, pb, es, cs = [], [], [], []
        for _ in range(max(1, args.repeat)):
            learner = build_learner(features, args.weight_col, a)
            bdt = learner.train(ds["sig_train"], ds["bg_train"])
            sc = {k: np.asarray(bdt.score(m[k], use_purity=not args.no_purity,
                                          quiet=True)) for k in m}
            ps.append(float(kolmogorov_smirnov_probability(
                sc["sig_train"], w["sig_train"], sc["sig_test"], w["sig_test"])))
            pb.append(float(kolmogorov_smirnov_probability(
                sc["bg_train"], w["bg_train"], sc["bg_test"], w["bg_test"])))
            cut, eff, rej = eff_at_rejection(sc["sig_test"], w["sig_test"],
                                             sc["bg_test"], w["bg_test"],
                                             args.target_rejection)
            es.append(eff)
            cs.append(cut)

        # Reject on the WORST run so one lucky run cannot carry a configuration.
        p_sig, p_bg = min(ps), min(pb)
        eff, cut = float(np.mean(es)), float(np.mean(cs))
        spread = (max(es) - min(es)) if len(es) > 1 else 0.0

        ok = min(p_sig, p_bg) >= args.ks_min
        rows.append(dict(depth=depth, trees=trees, split=split, prune=prune,
                         beta=beta, frac=frac, ncuts=ncuts,
                         p_sig=p_sig, p_bg=p_bg, cut=cut, eff=eff,
                         spread=spread, ok=ok))
        print("%-6s %-7s %-6s %-7s %-5s | %-8.4f %-8.4f | %-9.3f %6.1f%% +-%4.1f%% %s"
              % (depth, trees, split,
                 "-" if prune is None else ("%g" % prune), beta,
                 p_sig, p_bg, cut, 100 * eff, 100 * spread / 2,
                 "" if ok else "  <-- OVERTRAINED, rejected"))

    print("\n%d configurations, %.0f s" % (len(grid), time.time() - t0))

    good = [r for r in rows if r["ok"] and np.isfinite(r["eff"])]
    if not good:
        print("\nNo configuration reached p_KS >= %.3f." % args.ks_min)
        print("Lower the capacity (--depth 1,2  --num-trees 50,100), or process")
        print("more BACKGROUND events -- that is the real constraint.")
        return

    good.sort(key=lambda r: -r["eff"])
    b = good[0]
    print("\n=== Best (among those that survived the overtraining filter) ===")
    print("  efficiency %.1f%%  (rejection %.2f%%, cut %.3f)"
          % (100 * b["eff"], 100 * args.target_rejection, b["cut"]))
    print("  p_KS  signal %.4f  background %.4f" % (b["p_sig"], b["p_bg"]))
    print("\nPaste into the notebook:\n")
    hyper = ['"--num-trees", "%d"' % b["trees"],
             '"--depth", "%d"' % b["depth"],
             '"--beta", "%g"' % b["beta"],
             '"--min-split", "%d"' % b["split"],
             '"--frac-random-events", "%g"' % b["frac"]]
    if b["prune"] is not None:
        hyper.append('"--prune-strength", "%g"' % b["prune"])
    if not args.no_purity:
        hyper.append('"--use-purity"')
    print("HYPER = [" + ", ".join(hyper) + "]")

    if b["spread"] > 0.05:
        print("\n  [!] Repeats of this configuration spread by %.1f points."
              % (100 * b["spread"]))
        print("      The selection is largely measuring RANDOMNESS, not the")
        print("      hyperparameters.  The real constraint is background")
        print("      statistics; process more noise files.")
    print("\nSelection was done on the TEST set -> this efficiency is optimistic.")
    print("Retrain the chosen configuration with pybdt_train.py; because")
    print("training is random, its p_KS will differ somewhat.")


if __name__ == "__main__":
    main()
