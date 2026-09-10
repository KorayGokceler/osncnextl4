#!/usr/bin/env python
'''
AdaBoost (pybdt) vs LightGBM -- a FAIR head-to-head.

WHY
  The technical note (v00.074 sec. 3.6.1) trains the L4 classifiers with
  LightGBM; the hyperparameters in Table 10 are LightGBM's native names.
  Using pybdt/AdaBoost in this project was a DELIBERATE deviation -- but
  its cost was never measured.  The first noise BDT kept 53% of the signal
  at 99% background rejection; the target is 96%.  This script says whether
  that gap comes from the engine.

WHAT IS HELD IDENTICAL (the comparison is only fair because of these)
  * the same events -- the same .ds files, the same train/test split
  * the same features
  * the same training weights (the `weight` column; classes balanced)
  * the same measurement: signal efficiency at a target background
    rejection, COUNTED IN EVENTS, on the test set only.  The helpers
    (curve / eff_on_grid / kept_counts) are IMPORTED from compare_models.py,
    not reimplemented, so the number printed here is produced by exactly the
    same code as the pybdt numbers in CLAUDE.md 5h/5k.
  * the same overtraining criterion: the train/test efficiency GAP.

  pybdt scores live in [-1, 1], LightGBM outputs [0, 1] -- but the metric is
  insensitive to a monotone transform (the cut comes from a background
  quantile), so the different scales do not spoil the comparison.

WHY EVENT COUNTS AND NOT w_phys
  Same reason as compare_models.py: rate-weighted numbers depend on our
  invented E^-3 flux model, which is a separate open question.  A
  w_phys-weighted cross-check is printed at the end, clearly separated, so
  the main table stays comparable to the pybdt tables.

WHAT IS NECESSARILY DIFFERENT
  * the engine: AdaBoost vs gradient boosting
  * the hyperparameters: each engine has its own API.  The LightGBM side is
    READ WITH AST from PARAMS in reference/train_L4_classifier.py (not
    imported -- that file imports pandas/sklearn at module level and cannot
    be imported inside IceTray).  The values are straight from Table 10.
  * NaN: LightGBM routes missing values itself, whereas the pybdt path drops
    NaN events in the notebook (make_datasets.drop_nan).  We read the same
    .ds files, so there are no NaN events here either -- no difference is
    created, but be aware of it.
  * early stopping: LightGBM has it, pybdt does not.  The reference script
    early-stops on the TEST set; we do NOT (that would leak).  A validation
    slice is split off the TRAINING set instead (--valid-frac) and the test
    set is never touched during training.

READ THE RESULT WITH CARE
  Table 10 sets min_data_in_leaf=500 and it was tuned for the reference's
  (much larger) noise statistics.  With ~1000 background training events
  (CLAUDE.md 5e) that constraint alone can decide the outcome, so the script
  warns when the background sample is small relative to it.  A weak LightGBM
  result under those conditions says "not enough background", not
  "gradient boosting loses".

PLOTS (with --outdir)
  The same set the AdaBoost side produces:
    <tag>_lightgbm_overtrain.png   train vs test score shapes
    <tag>_lightgbm_dist.png        score distribution, linear and log
    <tag>_lightgbm_rate.png        weight surviving each cut
  (the three pybdt_train.py draws through pybdt's Validator, redrawn here
   because the Validator only takes pybdt BDT objects), plus
    <tag>_lightgbm_cuts.png        cut curves + efficiency vs rejection,
                                   the compare_models.py figure for one engine
  --pybdt-model puts an AdaBoost curve on that last panel, trained on the
  same .ds files through the same build_learner compare_models.py uses.

Usage:

    python lightgbm_compare.py --ds-dir L4_output/ds --tag L4_noise \
        --features NchCleaned,micro_count,iLineFit_speed,fill_ratio,FullTimeLengthRatio \
        --outdir L4_output/plots \
        --pybdt-model "d2t500:depth=2,trees=500" \
        --pybdt-model "d4t500:depth=4,trees=500"
'''

import os
import sys
import ast
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from pybdt import util
from pybdt.validate import kolmogorov_smirnov_probability
from pybdt_train import RESERVED_COLS, build_learner
from pybdt_scan import eff_at_rejection
# The measurement itself comes from compare_models so that both engines are
# scored by one implementation.  REJ_LEVELS keeps the table rows identical
# to the pybdt tables as well, and parse_model lets --pybdt-model take the
# very same spec strings compare_models.py takes.
from compare_models import (curve, eff_on_grid, kept_counts, parse_model,
                            REJ_LEVELS, REJ_GRID)

import matplotlib
matplotlib.use("Agg")           # headless -- same guard pybdt_train.py needs
import matplotlib.pyplot as plt

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("lightgbm is not available -- this comparison cannot run.\n"
             "Inside IceTray: python -c 'import lightgbm'")

REFERENCE_TRAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "reference", "train_L4_classifier.py")


def read_reference_params(path=REFERENCE_TRAIN):
    """
    Read the Table 10 values out of reference/train_L4_classifier.py with AST.

    WHY NOT IMPORT: that file does `import pandas` and
    `from sklearn.metrics import ...` at module level -- even though its own
    docstring says "sklearn and joblib are NOT available inside IceTray".  So
    it cannot be imported there.  Reading the values from the source beats
    copying them here and keeping two copies in sync (l4_data.check_feature_map
    reads FEATURE_MAP the same way).

    A missing or changed file is a hard error -- silently falling back to some
    other value would make the comparison against Table 10 meaningless.
    """
    if not os.path.exists(path):
        sys.exit("reference not found: %s" % path)
    tree = ast.parse(open(path, encoding="utf-8").read())

    def as_dict(node):
        # dict(a=1, b=2) -> {"a": 1, "b": 2}
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "dict":
            return {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
        return ast.literal_eval(node)

    params, early = None, None
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        for t in stmt.targets:
            name = getattr(t, "id", None)
            if name == "PARAMS" and isinstance(stmt.value, ast.Dict):
                params = {ast.literal_eval(k): as_dict(v)
                          for k, v in zip(stmt.value.keys, stmt.value.values)}
            elif name == "EARLY_STOPPING":
                early = ast.literal_eval(stmt.value)
    if params is None:
        sys.exit("PARAMS not found in %s (the file may have changed)" % path)
    return params, (early if early is not None else 100)


PARAMS, EARLY_STOPPING = read_reference_params()


def load_side(ds_dir, tag, part, features):
    '''Read one .ds file: feature matrix, training weight, physical weight.'''
    p = os.path.join(ds_dir, "%s_%s.ds" % (tag, part))
    if not os.path.exists(p):
        sys.exit("not found: %s" % p)
    ds = util.load(p)
    names = list(ds.names)
    missing = [f for f in features if f not in names]
    if missing:
        sys.exit("%s does not contain: %s\n  available: %s"
                 % (p, ", ".join(missing), ", ".join(sorted(names))))
    X = np.column_stack([np.asarray(ds[f], dtype=float) for f in features])
    w = np.nan_to_num(np.asarray(ds["weight"], dtype=float))
    wp = np.nan_to_num(np.asarray(ds["w_phys"], dtype=float)) \
        if "w_phys" in names else w.copy()
    return X, w, wp


def report_table(s, b, levels=REJ_LEVELS):
    '''
    Signal efficiency at fixed background rejection -- the same table shape
    compare_models.py prints, so the rows line up with the pybdt results.

    The raw counts are printed next to each row on purpose: at high rejection
    only a handful of background events sit above the cut, and a row backed by
    fewer than 10 events is not a measurement (CLAUDE.md 5h).
    '''
    cuts, eff, rej = curve(s, b)
    effs = eff_on_grid(eff, rej, levels)
    print("\n%-10s | %9s | %8s | %10s | %s"
          % ("rejection", "eff", "cut", "sig kept", "bg kept"))
    print("-" * 62)
    for lvl, e in zip(levels, effs):
        k = kept_counts(s, b, lvl)
        if k is None or not np.isfinite(e):
            print("%-10.1f%% | %9s | %8s | %10s | %s"
                  % (100 * lvl, "n/a", "-", "-", "cut unreachable"))
            continue
        mark = "  <-- NOT A MEASUREMENT" if k["bg_kept"] < 10 else ""
        print("%-10.1f%% | %8.1f%% | %8.4f | %5d/%5d | %d/%d%s"
              % (100 * lvl, 100 * e, k["cut"],
                 k["sig_kept"], k["sig_total"],
                 k["bg_kept"], k["bg_total"], mark))


# ---------------------------------------------------------------------------
# Plots -- the LightGBM counterparts of what the AdaBoost side produces.
#
#   pybdt_train.py  ->  <name>_overtrain.png, <name>_dist.png, <name>_rate.png
#                       (drawn by pybdt's own Validator)
#   compare_models.py -> <tag>_model_comparison.png  (cut curves + overlay)
#
# The Validator only takes pybdt BDT objects, so the first three are redrawn
# here with matplotlib against the same definitions -- same four series, same
# weighting, same dual linear/log layout.
# ---------------------------------------------------------------------------

SERIES = [("train_sig", "signal (train)", "tab:blue"),
          ("test_sig", "signal (test)", "tab:cyan"),
          ("train_bg", "background (train)", "tab:red"),
          ("test_bg", "background (test)", "tab:orange")]


def _range(scores, pad=0.02):
    vals = np.concatenate([np.asarray(v) for v in scores.values()])
    vals = vals[np.isfinite(vals)]
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        return (0.0, 1.0)
    span = hi - lo
    return (lo - pad * span, hi + pad * span)


def plot_overtrain(scores, weights, tag, outdir, p_sig, p_bg, bins=60):
    """
    Train vs test score shapes -- the counterpart of pybdt's
    create_overtrain_check_plot.

    Normalised to unit area so the comparison is about SHAPE: the training and
    test sets do not have the same size, and a size difference is not
    overtraining.  Training sets are drawn filled, test sets as points, which
    is how the pybdt plot reads too.
    """
    rng = _range(scores)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key, label, color in SERIES:
        h, edges = np.histogram(scores[key], bins=bins, range=rng,
                                weights=weights[key], density=True)
        mid = 0.5 * (edges[1:] + edges[:-1])
        if key.startswith("train"):
            ax.hist(mid, bins=edges, weights=h, histtype="stepfilled",
                    alpha=.35, color=color, label=label)
        else:
            ax.plot(mid, h, "o", ms=3, color=color, label=label)
    ax.set_xlabel("LightGBM score")
    ax.set_ylabel("normalised weight / bin")
    ax.set_yscale("log")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    ax.set_title("%s - overtraining check   (p_KS signal %.3f, background %.3f)"
                 % (tag, p_sig, p_bg), fontsize=10)
    return _save(fig, outdir, "%s_lightgbm_overtrain.png" % tag)


def plot_dist(scores, weights, tag, outdir, weight_name, bins=60):
    """Score distribution, weighted count per bin -- linear and log."""
    rng = _range(scores)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, logy in zip(axes, (False, True)):
        for key, label, color in SERIES:
            ax.hist(scores[key], bins=bins, range=rng, weights=weights[key],
                    histtype="step", lw=1.4, color=color, label=label)
        ax.set_xlabel("LightGBM score")
        ax.set_ylabel("%s / bin" % weight_name)
        ax.grid(alpha=.3)
        if logy:
            ax.set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.suptitle("%s - score distribution (weight column: %s)"
                 % (tag, weight_name), fontsize=11)
    return _save(fig, outdir, "%s_lightgbm_dist.png" % tag, suptitle=True)


def plot_rate(scores, weights, tag, outdir, weight_name, bins=200):
    """
    Weight surviving each cut -- 'if I cut here, what is left'.  The
    counterpart of pybdt_train's _rate plot.
    """
    rng = _range(scores)
    grid = np.linspace(rng[0], rng[1], bins)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, logy in zip(axes, (False, True)):
        for key, label, color in SERIES:
            v, w = np.asarray(scores[key]), np.asarray(weights[key])
            above = np.array([w[v >= c].sum() for c in grid])
            ax.plot(grid, above, lw=1.4, color=color, label=label)
        ax.set_xlabel("cut on LightGBM score")
        ax.set_ylabel("%s above the cut" % weight_name)
        ax.grid(alpha=.3)
        if logy:
            ax.set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.suptitle("%s - weight surviving the cut (weight column: %s)"
                 % (tag, weight_name), fontsize=11)
    return _save(fig, outdir, "%s_lightgbm_rate.png" % tag, suptitle=True)


def plot_cuts(s_te, b_te, tag, outdir, n_bg_te, extra=None):
    """
    The compare_models.py figure for one engine: where to cut on the left,
    efficiency vs rejection on the right.

    `extra` is a list of (label, s_test, b_test) from other engines -- the
    right-hand panel is the only fair way to put them on one axis, because
    the score scales differ.  The grey lines mark where the test-set
    background runs out; right of "10 events" nothing here is a measurement.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    cuts, eff, rej = curve(s_te, b_te)
    ax = axes[0]
    ax.plot(cuts, 100 * eff, color="tab:blue", lw=1.5, label="signal kept")
    ax.plot(cuts, 100 * rej, color="tab:red", lw=1.5, label="background rejected")
    ax.set_xlabel("cut on LightGBM score")
    ax.set_ylabel("% of events")
    ax.set_ylim(0, 102)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="center left")
    ax.set_title("LightGBM (Table 10) - where do I cut", fontsize=10)

    ax = axes[1]
    ax.plot(100 * REJ_GRID, 100 * eff_on_grid(eff, rej, REJ_GRID), lw=1.8,
            color="tab:green", label="LightGBM")
    for label, es, eb in (extra or []):
        c2, e2, r2 = curve(es, eb)
        ax.plot(100 * REJ_GRID, 100 * eff_on_grid(e2, r2, REJ_GRID), lw=1.5,
                ls="--", label=label)

    for n_left, style in ((100, "-."), (10, "--"), (1, ":")):
        r = 100 * (1.0 - n_left / float(n_bg_te))
        ax.axvline(r, color="0.4", ls=style, lw=1)
        ax.text(r, 4, " %d bg events left" % n_left, rotation=90,
                fontsize=7, color="0.3", va="bottom")

    # Table 13 target of the technical note -- same marker compare_models uses.
    ax.plot([99.2], [96.0], marker="*", ms=14, color="k", zorder=5)
    ax.text(99.2, 96.0, "  note target", fontsize=8, va="center")

    ax.set_xlabel("background rejection [%]")
    ax.set_ylabel("signal efficiency [%]")
    ax.set_xlim(80, 100)
    ax.set_ylim(0, 102)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Efficiency vs rejection - the only comparable axis",
                 fontsize=10)

    fig.suptitle("%s - cut curves, efficiency counted in events" % tag,
                 fontsize=12)
    return _save(fig, outdir, "%s_lightgbm_cuts.png" % tag, suptitle=True)


def _save(fig, outdir, name, suptitle=False):
    fig.tight_layout(rect=[0, 0, 1, 0.94] if suptitle else None)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("  -> %s" % path)
    return path


def train_pybdt(ds_dir, tag, features, specs, weight_col="weight"):
    """
    Train the given pybdt models on the SAME .ds files, for the overlay.

    Deliberately reuses parse_model and build_learner rather than
    reimplementing: the overlay is only worth drawing if the AdaBoost curve on
    it is the same curve compare_models.py would draw.
    """
    parts = {}
    for part in ("sig_train", "bg_train", "sig_test", "bg_test"):
        parts[part] = util.load(os.path.join(ds_dir, "%s_%s.ds" % (tag, part)))
    out = []
    for spec in specs:
        label, cfg = parse_model(spec)
        print("  training pybdt %s ..." % label)
        learner = build_learner(features, weight_col, cfg)
        bdt = learner.train(parts["sig_train"], parts["bg_train"])
        out.append((
            "pybdt %s" % label,
            np.asarray(bdt.score(parts["sig_test"], use_purity=cfg.use_purity,
                                 quiet=True)),
            np.asarray(bdt.score(parts["bg_test"], use_purity=cfg.use_purity,
                                 quiet=True))))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="AdaBoost (pybdt) vs LightGBM on identical data")
    ap.add_argument("--ds-dir", required=True)
    ap.add_argument("--tag", default="L4_noise")
    ap.add_argument("--features", required=True,
                    help="comma separated -- must be the SAME as the pybdt run")
    ap.add_argument("--params", default=None,
                    help="PARAMS key: noise|muon.  Derived from --tag if unset.")
    ap.add_argument("--target-rejection", type=float, default=0.99,
                    help="rejection the headline number is quoted at")
    ap.add_argument("--gap-at", type=float, default=0.90,
                    help="rejection the train/test efficiency gap is measured "
                         "at.  0.90 by default because ~100 background events "
                         "sit above that cut; at 0.99 it is ~10 and the gap is "
                         "mostly noise (same default as compare_models.py).")
    ap.add_argument("--valid-frac", type=float, default=0.2,
                    help="slice split off the TRAINING set for early stopping. "
                         "The test set is never used during training.")
    ap.add_argument("--no-early-stopping", action="store_true")
    ap.add_argument("--num-boost-round", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", default=None,
                    help="if given, the model (.txt) and the plots are written there")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip the plots even with --outdir")
    ap.add_argument("--plot-weight", choices=("weight", "w_phys"),
                    default="weight",
                    help="weighting for the dist/rate/overtrain plots. "
                         "'weight' (default) is the class-balanced training "
                         "weight, the same one pybdt_train.py plots with, so "
                         "the figures are comparable side by side.")
    ap.add_argument("--pybdt-model", action="append", default=[],
                    metavar="SPEC",
                    help="also train this pybdt model on the same .ds files "
                         "and draw it on the efficiency-vs-rejection panel. "
                         "Same spec syntax as compare_models.py, repeatable, "
                         'e.g. --pybdt-model "d2t500:depth=2,trees=500"')
    args = ap.parse_args()

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    bad = [f for f in features if f in RESERVED_COLS]
    if bad:
        sys.exit("these columns are not BDT inputs: %s" % ", ".join(bad))

    key = args.params or ("muon" if "muon" in args.tag.lower() else "noise")
    if key not in PARAMS:
        sys.exit("unknown PARAMS key %r (available: %s)"
                 % (key, ", ".join(sorted(PARAMS))))
    p = dict(PARAMS[key])
    n_rounds = p.pop("num_boost_round", 2000)
    if args.num_boost_round is not None:
        n_rounds = args.num_boost_round
    if args.learning_rate is not None:
        p["learning_rate"] = args.learning_rate
    p["seed"] = args.seed

    # --- data ---------------------------------------------------------------
    Xs, ws_tr, wps_tr = load_side(args.ds_dir, args.tag, "sig_train", features)
    Xb, wb_tr, wpb_tr = load_side(args.ds_dir, args.tag, "bg_train", features)
    Xst, ws_te, wps_te = load_side(args.ds_dir, args.tag, "sig_test", features)
    Xbt, wb_te, wpb_te = load_side(args.ds_dir, args.tag, "bg_test", features)

    print("=== %s : LightGBM (Table 10) ===" % args.tag)
    print("  train  signal %8d   background %6d" % (len(Xs), len(Xb)))
    print("  test   signal %8d   background %6d" % (len(Xst), len(Xbt)))
    print("  %d features: %s" % (len(features), ", ".join(features)))
    print("  parameters (%s): %s" % (key, ", ".join(
        "%s=%s" % (k, v) for k, v in sorted(p.items())
        if k not in ("objective", "metric", "verbosity", "deterministic"))))
    print("  efficiency and rejection below are EVENT COUNTS (w_phys not used)")

    X = np.vstack([Xs, Xb])
    y = np.r_[np.ones(len(Xs)), np.zeros(len(Xb))]
    w = np.r_[ws_tr, wb_tr]

    rng = np.random.default_rng(args.seed)
    if args.no_early_stopping or args.valid_frac <= 0:
        tr = np.ones(len(y), dtype=bool)
        va = None
    else:
        # The validation slice comes out of the TRAINING set -- the test set is
        # left alone.  Split per class so the class ratio is preserved: the
        # background is scarce and a blind random split could thin it further.
        tr = np.ones(len(y), dtype=bool)
        for cls in (0.0, 1.0):
            idx = np.flatnonzero(y == cls)
            k = max(1, int(round(args.valid_frac * len(idx))))
            tr[rng.choice(idx, size=k, replace=False)] = False
        va = ~tr
        print("  early stopping: %.0f%% validation slice off the TRAINING set "
              "(signal %d, background %d), metric %s"
              % (100 * args.valid_frac, int((va & (y == 1)).sum()),
                 int((va & (y == 0)).sum()), p.get("metric", "auc")))

    # Table 10 was tuned on the reference's (much larger) background sample.
    # With ~1000 background events a leaf-size floor of 500 is a hard cap on
    # what any tree can carve out, and it would dominate the result.
    n_bg_fit = int((y[tr] == 0).sum())
    min_leaf = p.get("min_data_in_leaf")
    if min_leaf and n_bg_fit < 10 * min_leaf:
        print("\n  !! WARNING: min_data_in_leaf=%d but only %d background "
              "events are fitted." % (min_leaf, n_bg_fit))
        print("     Table 10 assumes far more background statistics "
              "(CLAUDE.md 5e).  A weak result here means 'not enough")
        print("     background', NOT 'gradient boosting loses'.  Re-run with "
              "--params noise and a lowered min_data_in_leaf to separate the two.")

    dtrain = lgb.Dataset(X[tr], label=y[tr], weight=w[tr],
                         feature_name=features, free_raw_data=False)
    callbacks = [lgb.log_evaluation(0)]
    valid_sets = None
    if va is not None:
        valid_sets = [lgb.Dataset(X[va], label=y[va], weight=w[va],
                                  feature_name=features, reference=dtrain)]
        callbacks.append(lgb.early_stopping(EARLY_STOPPING, verbose=False))

    print("\nTraining ...")
    booster = lgb.train(p, dtrain, num_boost_round=n_rounds,
                        valid_sets=valid_sets, callbacks=callbacks)
    # predict() defaults to best_iteration when early stopping fired, so that
    # is the model being scored -- report it, not the number of trees built.
    best = booster.best_iteration or booster.num_trees()
    print("  done -- scoring with %d trees (%d built, limit %d)"
          % (best, booster.num_trees(), n_rounds))
    if va is not None and best >= n_rounds:
        print("  note: early stopping never fired; the round limit was hit, "
              "so the model may still be improving.")

    # --- evaluation: the SAME measurement as the pybdt side -----------------
    s_tr = booster.predict(Xs)
    b_tr = booster.predict(Xb)
    s_te = booster.predict(Xst)
    b_te = booster.predict(Xbt)

    print("\n--- Signal efficiency vs background rejection (TEST set, events) ---")
    report_table(s_te, b_te)

    k_te = kept_counts(s_te, b_te, args.target_rejection)
    print("\n--- HEADLINE ---")
    if k_te is None:
        print("  %.1f%% rejection is not reachable on this test set."
              % (100 * args.target_rejection))
    else:
        print("  signal efficiency at %.1f%% background rejection: %.1f%%"
              % (100 * args.target_rejection, 100 * k_te["eff"]))
        print("  (cut %.4f, %d/%d background events left)"
              % (k_te["cut"], k_te["bg_kept"], k_te["bg_total"]))
        if k_te["bg_kept"] < 10:
            print("  !! fewer than 10 background events define this point -- "
                  "it is not a measurement (CLAUDE.md 5h).")

    # --- overtraining -------------------------------------------------------
    # The gap is the criterion; p_KS is printed for continuity with
    # pybdt_train.py but is NOT used to judge (CLAUDE.md 5i).
    g_te = kept_counts(s_te, b_te, args.gap_at)
    g_tr = kept_counts(s_tr, b_tr, args.gap_at)
    print("\n--- Overtraining ---")
    if g_te is None or g_tr is None:
        print("  gap at %.0f%% rejection: not reachable" % (100 * args.gap_at))
    else:
        gap = g_tr["eff"] - g_te["eff"]
        print("  eff at %.0f%% rejection: train %.1f%%  test %.1f%%  gap %+.1f%s"
              % (100 * args.gap_at, 100 * g_tr["eff"], 100 * g_te["eff"],
                 100 * gap, "  <-- MEMORISING" if gap > 0.05 else ""))
    p_sig = float(kolmogorov_smirnov_probability(s_tr, ws_tr, s_te, ws_te))
    p_bg = float(kolmogorov_smirnov_probability(b_tr, wb_tr, b_te, wb_te))
    print("  p_KS signal %.4f / background %.4f "
          "(informational only -- see CLAUDE.md 5i)" % (p_sig, p_bg))

    # --- rate-weighted cross-check ------------------------------------------
    # Kept separate from the table above because w_phys depends on our invented
    # E^-3 flux model.  For noise the two agree anyway (CLAUDE.md 5d).
    cut, eff, rej = eff_at_rejection(s_te, wps_te, b_te, wpb_te,
                                     args.target_rejection)
    print("\n--- Rate-weighted cross-check (w_phys, not the headline) ---")
    if np.isfinite(eff):
        print("  efficiency %.1f%% at %.2f%% rejection (cut %.4f)"
              % (100 * eff, 100 * rej, cut))
    else:
        print("  not computable (zero total weight)")

    print("\n--- Feature importance (gain) ---")
    imp = booster.feature_importance(importance_type="gain",
                                     iteration=best)
    tot = imp.sum() or 1.0
    for f, v in sorted(zip(features, imp), key=lambda x: -x[1]):
        print("  %-22s %8.1f  (%.1f%%)" % (f, v, 100 * v / tot))

    # --- outputs ------------------------------------------------------------
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        path = os.path.join(args.outdir, "%s_lightgbm.txt" % args.tag)
        booster.save_model(path, num_iteration=best)
        print("\n  -> %s" % path)

        if not args.no_plots:
            # The same figures the AdaBoost side produces: three from
            # pybdt_train.py's Validator, one from compare_models.py.
            scores = {"train_sig": s_tr, "test_sig": s_te,
                      "train_bg": b_tr, "test_bg": b_te}
            if args.plot_weight == "w_phys":
                weights = {"train_sig": wps_tr, "test_sig": wps_te,
                           "train_bg": wpb_tr, "test_bg": wpb_te}
            else:
                weights = {"train_sig": ws_tr, "test_sig": ws_te,
                           "train_bg": wb_tr, "test_bg": wb_te}

            extra = []
            if args.pybdt_model:
                print("\n--- AdaBoost overlay (same .ds, same features) ---")
                extra = train_pybdt(args.ds_dir, args.tag, features,
                                    args.pybdt_model)

            print("\n--- Plots ---")
            try:
                plot_overtrain(scores, weights, args.tag, args.outdir,
                               p_sig, p_bg)
                plot_dist(scores, weights, args.tag, args.outdir,
                          args.plot_weight)
                plot_rate(scores, weights, args.tag, args.outdir,
                          args.plot_weight)
                plot_cuts(s_te, b_te, args.tag, args.outdir, len(b_te),
                          extra=extra)
            except Exception as exc:
                # Same guard pybdt_train.py grew after a matplotlib error took
                # a whole run down: the numbers above are already printed and
                # the model is already saved, so a drawing failure must not
                # lose them.
                print("  [!] plotting failed (%s: %s)" % (type(exc).__name__, exc))
                print("      The numbers above and the saved model are unaffected.")
            if not args.pybdt_model:
                print("\n  hint: add --pybdt-model \"d2t500:depth=2,trees=500\" to put")
                print("        the AdaBoost curve on the same axis.")
    else:
        print("\n  (no --outdir given: no model and no plots were written)")

    print("\n" + "=" * 70)
    print("Compare the HEADLINE line against the same row of the pybdt run")
    print("(compare_models.py, or the tables in CLAUDE.md 5h/5k) -- both are")
    print("event counts on the test set, produced by the same helpers.")
    print("Close    -> the gap is not the engine (data or features).")
    print("LightGBM clearly ahead -> AdaBoost is not enough for this problem.")
    print("Either way, check the min_data_in_leaf warning first.")
    print("=" * 70)


if __name__ == "__main__":
    main()
