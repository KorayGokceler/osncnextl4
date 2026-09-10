#!/usr/bin/env python
'''
Train several models and compare their cut curves.

EFFICIENCY IS COUNTED IN EVENTS, NOT RATES.  Every efficiency and rejection
number here is a plain event count on the test set:

    signal efficiency = (signal events kept) / (signal events total)
    background rejection = 1 - (background events kept) / (background total)

w_phys is deliberately NOT used.  Rate-weighted numbers depend on our
invented E^-3 flux model, which is a separate open question; counting events
removes that from the comparison.  (Note: for noise the two agree anyway --
vuvuzela weights are uniform, see CLAUDE.md 5d.)

WHAT IT PRODUCES
  1. A table: for each model, the signal efficiency at fixed background
     rejection levels, with the raw counts next to it.  The "background kept"
     column shows how few events define the high-rejection points -- at
     99.9% rejection of ~1000 test events that is ONE event.
  2. A figure:
     - one panel per model: cut value on x, signal efficiency and background
       rejection on y  ("where do I cut, what do I keep, what do I throw")
     - one overlay panel: efficiency vs rejection for every model, which is
       the only fair way to compare models whose score scales differ.

OVERTRAINING is still reported with pybdt's own weighted KS, the same
statistic pybdt_train.py prints, so the numbers stay comparable to earlier
runs.  Efficiency is unweighted; the KS test is not.

TRAINING IS DETERMINISTIC.  --only-determinism measured it: the same
configuration trained twice gives BIT-IDENTICAL scores.  So there is nothing
to average over -- each curve below is exact for the given .ds files.  What
IS uncertain is the finite test set: at high rejection only a handful of
background events sit above the cut, so the vertical marker lines show where
100, 10 and 1 background events remain.  Anything to the right of the "10
events" line is not a measurement.

MODEL SPEC
    --model "label:key=value,key=value"
  keys: depth, trees, beta, min_split, prune, frac, cuts, purity
  e.g.  --model "d2t500:depth=2,trees=500"
        --model "d4t500p10:depth=4,trees=500,prune=10"
  With no --model given, a default set spanning the capacity range is used.

Usage:

    python compare_models.py --ds-dir L4_output/ds --tag L4_noise \
        --features NchCleaned,micro_count,iLineFit_speed,fill_ratio,FullTimeLengthRatio \
        --outdir L4_output/plots
'''

import os
import sys
import argparse
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from pybdt import util
from pybdt.validate import kolmogorov_smirnov_probability
from pybdt_train import build_learner, RESERVED_COLS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Rejection levels reported in the table.
REJ_LEVELS = [0.90, 0.95, 0.99, 0.995, 0.999]

# Grid the curves are averaged on.  Dense near 1 because that is where the
# interesting (and least stable) part of the curve lives.
REJ_GRID = np.concatenate([np.linspace(0.0, 0.98, 99),
                           1 - np.logspace(np.log10(0.02), np.log10(1e-4), 60)])

DEFAULT_MODELS = [
    "stump:depth=1,trees=300",
    "d2t500:depth=2,trees=500",
    "d3t300:depth=3,trees=300",
    "d4t500:depth=4,trees=500",
    "d6t500:depth=6,trees=500",
    "d2t500p10:depth=2,trees=500,prune=10",
]

_KEYS = {"depth": int, "trees": int, "beta": float, "min_split": int,
         "prune": float, "frac": float, "cuts": int, "purity": int}


def parse_model(spec):
    '''"label:depth=2,trees=500" -> (label, SimpleNamespace)'''
    if ":" in spec:
        label, rest = spec.split(":", 1)
    else:
        label, rest = spec, spec
    cfg = dict(depth=None, trees=None, beta=None, min_split=None,
               prune=None, frac=0.5, cuts=None, purity=1)
    for piece in rest.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            sys.exit("bad model spec piece (expected key=value): %r" % piece)
        k, v = piece.split("=", 1)
        k = k.strip()
        if k not in _KEYS:
            sys.exit("unknown model key %r (known: %s)"
                     % (k, ", ".join(sorted(_KEYS))))
        cfg[k] = None if v.strip().lower() in ("none", "-") else _KEYS[k](v)
    return label.strip(), SimpleNamespace(
        depth=cfg["depth"], min_split=cfg["min_split"], num_cuts=cfg["cuts"],
        num_random_variables=None, nonlinear_cuts=False,
        num_trees=cfg["trees"], beta=cfg["beta"],
        frac_random_events=cfg["frac"], prune_strength=cfg["prune"],
        use_purity=bool(cfg["purity"]))


def curve(s, b):
    '''
    Cut scan on event counts.

    Returns (cuts, efficiency, rejection).  Thresholds are the distinct
    background scores, so every point corresponds to a real place a cut could
    sit -- no interpolation artefacts at the high-rejection end.
    '''
    cuts = np.unique(np.concatenate([b, [b.min() - 1.0, b.max() + 1.0]]))
    eff = np.array([(s >= c).mean() for c in cuts])
    rej = np.array([1.0 - (b >= c).mean() for c in cuts])
    return cuts, eff, rej


def eff_on_grid(eff, rej, grid):
    '''Signal efficiency at each target rejection (step-wise, no smoothing).'''
    order = np.argsort(rej)
    r, e = rej[order], eff[order]
    out = np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        ok = np.flatnonzero(r >= g)
        if len(ok):
            out[i] = e[ok[0]]       # first cut that reaches this rejection
    return out


def kept_counts(s, b, target):
    '''Counts at the first cut reaching `target` rejection -- the raw numbers.'''
    cuts, eff, rej = curve(s, b)
    ok = np.flatnonzero(rej >= target)
    if not len(ok):
        return None
    i = ok[np.argmax(eff[ok])]
    c = cuts[i]
    return dict(cut=float(c), eff=float(eff[i]), rej=float(rej[i]),
                sig_kept=int((s >= c).sum()), sig_total=len(s),
                bg_kept=int((b >= c).sum()), bg_total=len(b))


def determinism_check(ds, features, weight_col, spec, target=0.99):
    """
    Do two trainings with IDENTICAL hyperparameters really differ?

    We have been ASSUMING pybdt's training is stochastic (frac_random_events
    samples events per tree, and pybdt exposes no seed setter), and blaming
    the +-10..17 point spread on that.  This measures it instead of assuming
    it: train the same configuration twice and compare the two score arrays
    element by element.

    If the scores are bit-identical, training is deterministic and the spread
    we saw came from somewhere else entirely -- which would invalidate a
    conclusion we have been leaning on.
    """
    label, a = parse_model(spec)
    print("=" * 92)
    print("DETERMINISM CHECK -- same hyperparameters, trained twice")
    print("=" * 92)
    print("  config: %s (depth=%s, trees=%s, beta=%s, frac_random_events=%s)"
          % (label, a.depth, a.num_trees, a.beta, a.frac_random_events))

    runs = []
    for _ in range(2):
        learner = build_learner(features, weight_col, a)
        bdt = learner.train(ds["sig_train"], ds["bg_train"])
        runs.append(dict(
            n_trees=len(bdt),
            sig=np.asarray(bdt.score(ds["sig_test"], use_purity=a.use_purity,
                                     quiet=True)),
            bg=np.asarray(bdt.score(ds["bg_test"], use_purity=a.use_purity,
                                    quiet=True))))

    same_trees = runs[0]["n_trees"] == runs[1]["n_trees"]
    d_sig = np.abs(runs[0]["sig"] - runs[1]["sig"])
    d_bg = np.abs(runs[0]["bg"] - runs[1]["bg"])
    n_diff = int((d_sig > 0).sum() + (d_bg > 0).sum())
    n_tot = len(d_sig) + len(d_bg)

    print("\n  trees:            %d vs %d%s"
          % (runs[0]["n_trees"], runs[1]["n_trees"],
             "" if same_trees else "   <-- differ"))
    print("  events with a different score: %d / %d  (%.1f%%)"
          % (n_diff, n_tot, 100.0 * n_diff / n_tot))
    print("  max |score difference|:        %.6g" % max(d_sig.max(), d_bg.max()))

    for i, r in enumerate(runs):
        k = kept_counts(r["sig"], r["bg"], target)
        print("  run %d: efficiency at %.0f%% rejection = %5.1f%%  "
              "(signal %d/%d, background %d/%d, cut %.4f)"
              % (i, 100 * target, 100 * k["eff"], k["sig_kept"], k["sig_total"],
                 k["bg_kept"], k["bg_total"], k["cut"]))

    if n_diff == 0 and same_trees:
        print("\n  -> IDENTICAL.  Training is deterministic; the spread seen")
        print("     earlier does NOT come from training randomness.  Look at")
        print("     the metric instead (the 99%% rejection threshold sits")
        print("     behind ~10 background events) or at the .ds regeneration.")
    else:
        print("\n  -> DIFFERENT.  Training is stochastic, as assumed: the same")
        print("     configuration gives a different model each time.  Any A/B")
        print("     comparison needs repeats.")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Train several models and compare their cut curves "
                    "(efficiency counted in events)")
    ap.add_argument("--ds-dir", required=True)
    ap.add_argument("--tag", default="L4_noise")
    ap.add_argument("--features", default=None)
    ap.add_argument("--weight-col", default="weight")
    ap.add_argument("--model", action="append", default=None,
                    help='"label:depth=2,trees=500"; repeatable')
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--check-determinism", action="store_true",
                    help="train the first configuration twice and report "
                         "whether the two models are identical, then continue")
    ap.add_argument("--only-determinism", action="store_true",
                    help="run just that check and stop")
    args = ap.parse_args()

    specs = args.model if args.model else DEFAULT_MODELS
    models = [parse_model(s) for s in specs]

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

    n_sig_te = len(ds["sig_test"][features[0]])
    n_bg_te = len(ds["bg_test"][features[0]])
    print("=== %s : model comparison ===" % args.tag)
    print("  train  signal %8d   background %6d"
          % (len(ds["sig_train"][features[0]]), len(ds["bg_train"][features[0]])))
    print("  test   signal %8d   background %6d" % (n_sig_te, n_bg_te))
    print("  %d variables: %s" % (len(features), ", ".join(features)))
    print("  efficiency and rejection are EVENT COUNTS (w_phys not used)")
    print("  %d models (training is deterministic, one run each)\n" % len(models))

    if args.check_determinism or args.only_determinism:
        determinism_check(ds, features, args.weight_col, specs[0])
        if args.only_determinism:
            return

    # KS still uses the training weights, to stay comparable with pybdt_train.
    w = {k: np.nan_to_num(np.asarray(ds[k][args.weight_col]))
         for k in ("sig_train", "sig_test", "bg_train", "bg_test")}

    results = []
    for label, a in models:
        learner = build_learner(features, args.weight_col, a)
        bdt = learner.train(ds["sig_train"], ds["bg_train"])
        sc = {k: np.asarray(bdt.score(ds[k], use_purity=a.use_purity, quiet=True))
              for k in ("sig_train", "sig_test", "bg_train", "bg_test")}
        ks_sig = float(kolmogorov_smirnov_probability(
            sc["sig_train"], w["sig_train"], sc["sig_test"], w["sig_test"]))
        ks_bg = float(kolmogorov_smirnov_probability(
            sc["bg_train"], w["bg_train"], sc["bg_test"], w["bg_test"]))
        cuts, eff, rej = curve(sc["sig_test"], sc["bg_test"])
        results.append(dict(label=label, cfg=a, scores=sc,
                            eff_grid=eff_on_grid(eff, rej, REJ_GRID),
                            ks_sig=ks_sig, ks_bg=ks_bg))
        print("  trained %-12s  depth=%-4s trees=%-5s prune=%-5s  "
              "p_KS sig=%.3f bg=%.3f%s"
              % (label, a.depth, a.num_trees,
                 "-" if a.prune_strength is None else a.prune_strength,
                 ks_sig, ks_bg,
                 "  <-- OVERTRAINED" if min(ks_sig, ks_bg) < 0.01 else ""))

    # ---- table -------------------------------------------------------------
    print("\n" + "=" * 92)
    print("SIGNAL EFFICIENCY AT FIXED BACKGROUND REJECTION  (event counts, "
          "test set, run 0)")
    print("=" * 92)
    hdr = "%-12s" % "model" + "".join("%15s" % ("rej %.1f%%" % (100 * t))
                                      for t in REJ_LEVELS)
    print(hdr)
    print("-" * len(hdr))
    for res in results:
        row = "%-12s" % res["label"]
        for t in REJ_LEVELS:
            k = kept_counts(res["scores"]["sig_test"], res["scores"]["bg_test"], t)
            row += "%14s " % ("-" if k is None else "%.1f%%" % (100 * k["eff"]))
        print(row)

    print("\nraw counts at each level (run 0):")
    for res in results:
        print("  %-12s" % res["label"], end="")
        for t in REJ_LEVELS:
            k = kept_counts(res["scores"]["sig_test"], res["scores"]["bg_test"], t)
            if k is None:
                print("  %-22s" % "-", end="")
            else:
                print("  sig %6d/%d bg %3d/%d" % (k["sig_kept"], k["sig_total"],
                                                  k["bg_kept"], k["bg_total"]),
                      end="")
        print()
    print("\n  Note how few background events define the right-hand columns:")
    print("  at 99.9%% rejection of %d test events, that is ~%d event(s)."
          % (n_bg_te, max(1, int(round(0.001 * n_bg_te)))))

    # ---- figure ------------------------------------------------------------
    n = len(results)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol)) + 1
    fig = plt.figure(figsize=(4.8 * ncol, 3.4 * nrow))

    for i, res in enumerate(results):
        ax = fig.add_subplot(nrow, ncol, i + 1)
        cuts, eff, rej = curve(res["scores"]["sig_test"], res["scores"]["bg_test"])
        ax.plot(cuts, 100 * eff, color="tab:blue", lw=1.4,
                label="signal kept")
        ax.plot(cuts, 100 * rej, color="tab:red", lw=1.4,
                label="background rejected")
        ax.set_xlabel("cut on BDT score")
        ax.set_ylabel("% of events")
        ax.set_ylim(0, 102)
        ax.grid(alpha=.3)
        ax.set_title("%s  (depth %s, %s trees)"
                     % (res["label"], res["cfg"].depth, res["cfg"].num_trees),
                     fontsize=9)
        if i == 0:
            ax.legend(fontsize=8, loc="center left")

    ax = fig.add_subplot(nrow, 1, nrow)
    for res in results:
        ax.plot(100 * REJ_GRID, 100 * res["eff_grid"], lw=1.6, label=res["label"])

    # Where the test-set statistics runs out: how many background events are
    # still above the cut.  Right of the "10 events" line this is not a
    # measurement any more, whatever the curve suggests.
    for n_left, style in ((100, "-."), (10, "--"), (1, ":")):
        r = 100 * (1.0 - n_left / float(n_bg_te))
        ax.axvline(r, color="0.4", ls=style, lw=1)
        ax.text(r, 4, " %d bg events left" % n_left, rotation=90,
                fontsize=7, color="0.3", va="bottom")

    # Table 13 target of the technical note.
    ax.plot([99.2], [96.0], marker="*", ms=14, color="k", zorder=5)
    ax.text(99.2, 96.0, "  note target", fontsize=8, va="center")

    ax.set_xlabel("background rejection [%]")
    ax.set_ylabel("signal efficiency [%]")
    ax.set_xlim(80, 100)
    ax.set_ylim(0, 102)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    ax.set_title("All models on one axis.  Curves are exact (training is "
                 "deterministic); the grey lines mark where the test-set "
                 "background runs out.", fontsize=9)

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "%s_model_comparison.png" % args.tag)
    fig.suptitle("%s - cut curves, efficiency counted in events" % args.tag,
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=130)
    print("\n  -> %s" % out)

    print("\n" + "=" * 92)
    print("HOW TO READ THIS")
    print("=" * 92)
    print("* Per-model panels answer 'where do I cut, what do I keep/throw'.")
    print("  The score scale differs between models, so these panels are NOT")
    print("  comparable to each other -- use the bottom overlay for that.")
    print("* The overlay is the comparison: higher curve = better model.")
    print("* Right of the '10 bg events left' line the curve is defined by a")
    print("  handful of events -- differences there are not measurements.")
    print("* Compare against the single best variable: a plain cut on")
    print("  NchCleaned alone reached ~72% at 99% rejection.  A model that")
    print("  does not clearly beat that is not earning its keep.")


if __name__ == "__main__":
    main()
