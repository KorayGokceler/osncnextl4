#!/usr/bin/env python
'''
oscNext L4 classifier training -- LightGBM, the method of the technical note.

    L4_<tag>_dataset.npz  -->  train_L4_classifier.py  -->  L4_<tag>_model.txt
                                                            L4_<tag>_model.json
                                                            plots

The notebook writes the .npz; this script owns the training.  Keeping the two
apart means there is exactly one implementation of the training, and the
notebook stays a thin interface.

DEPENDENCIES: numpy + lightgbm (+ matplotlib for the plots).  Deliberately NO
pandas, NO sklearn, NO joblib -- none of them exist in the IceTray environment
(py3-v4.4.2), and this script has to be runnable there.

MODEL FORMAT: LightGBM's own text format (.txt) plus a JSON sidecar, not
joblib/pickle.  The text format is stable across versions and readable; a
pickle is neither, and joblib is not installed on the application side.
Application side: oscnext_l4/classifier.py (lgb.Booster + numpy, nothing else).

HOW IT IS EVALUATED
  Signal efficiency at a target background rejection, COUNTED IN EVENTS, on
  the TEST SET ONLY.  Event counts rather than rates because rate weights
  depend on our invented E^-3 flux model; the w_phys-weighted number is
  printed separately as a cross-check.

  Overtraining is judged by the train/test efficiency GAP, not by a KS test
  on the score distributions.  Measured on this data, KS flagged the best
  model as overtrained and the worst as clean -- with ~165k signal events it
  finds differences that are statistically significant and physically
  irrelevant.  The gap compares what we actually care about.

  Watch the "bg kept" column.  With ~1000 background test events, 99%
  rejection is defined by 10 of them and 99.9% by one.  Rows backed by fewer
  than 10 events are marked; they are not measurements.

Usage:

    python train_L4_classifier.py --tag noise \
        --dataset L4_output/ds/L4_noise_dataset.npz \
        --outdir L4_output/models
'''

import os
import sys
import json
import argparse
import datetime

import numpy as np
import lightgbm as lgb


# ===========================================================================
# Hyperparameters -- technical note Table 10
# ===========================================================================
#
# The native API is used, so the Table 10 names apply directly.  (The sklearn
# API renames these to min_child_samples / colsample_bytree / reg_alpha /
# reg_lambda / min_split_gain -- we are NOT using it.)
#
# NOTE min_data_in_leaf=500 was tuned by the reference on much larger noise
# statistics than we have.  With ~1000 background training events it is a hard
# cap on what any tree can carve out; the script warns when that is the case.

PARAMS = {
    "noise": dict(
        objective="binary", metric="auc", verbosity=-1,
        max_depth=6,
        num_leaves=25,
        max_bin=32,
        min_data_in_leaf=500,
        feature_fraction=0.8,       # noise
        lambda_l1=2.0,
        lambda_l2=1.0,
        min_gain_to_split=2.0,
        is_unbalance=False,         # classes are balanced through `weight`
        learning_rate=0.05,
        num_boost_round=2000,       # early stopping cuts this short
        seed=12345, deterministic=True,
    ),
    "muon": dict(
        objective="binary", metric="auc", verbosity=-1,
        max_depth=6,
        num_leaves=25,
        max_bin=32,
        min_data_in_leaf=500,
        feature_fraction=0.7,       # muon
        lambda_l1=2.0,
        lambda_l2=1.0,
        min_gain_to_split=2.0,
        is_unbalance=False,
        learning_rate=0.05,
        num_boost_round=2000,
        seed=12345, deterministic=True,
    ),
}

EARLY_STOPPING = 100

# Reference cut values, v00.07 pass2.  These are the note's LightGBM
# probabilities, so they carry over to a LightGBM model -- unlike a pybdt
# score, which lives on a different scale entirely.
DEFAULT_CUT = {"noise": 0.70, "muon": 0.65}

# Rejection levels the results table reports.
REJ_LEVELS = [0.90, 0.95, 0.99, 0.995, 0.999]

# Columns in the .npz that are NOT model inputs.
RESERVED_COLS = {"w_phys", "weight", "istrain", "features",
                 "livetime", "Run", "Event", "SubEvent"}


# ===========================================================================
# Data
# ===========================================================================

def load_dataset(path, features=None):
    '''
    Read the .npz the notebook wrote.

    Expected arrays: one per feature, plus
      weight   training weight (classes balanced)
      w_phys   physical weight (rate), for the cross-check only
      label    1 = signal, 0 = background
      istrain  True = training set
      features the feature names, so the model and the file cannot drift apart
    '''
    if not os.path.exists(path):
        sys.exit("dataset not found: %s" % path)
    z = np.load(path, allow_pickle=False)
    names = list(z.files)

    if features is None:
        if "features" in names:
            features = [str(f) for f in z["features"]]
        else:
            features = sorted(n for n in names
                              if n not in RESERVED_COLS and n != "label")
    missing = [f for f in features if f not in names]
    if missing:
        sys.exit("%s does not contain: %s\n  available: %s"
                 % (path, ", ".join(missing), ", ".join(sorted(names))))
    bad = [f for f in features if f in RESERVED_COLS or f == "label"]
    if bad:
        sys.exit("these are not model inputs: %s" % ", ".join(bad))

    X = np.column_stack([np.asarray(z[f], dtype=float) for f in features])
    y = np.asarray(z["label"], dtype=float)
    w = np.nan_to_num(np.asarray(z["weight"], dtype=float))
    wp = np.nan_to_num(np.asarray(z["w_phys"], dtype=float)) \
        if "w_phys" in names else w.copy()
    tr = np.asarray(z["istrain"], dtype=bool)

    nan_rows = int((~np.isfinite(X)).any(axis=1).sum())
    if nan_rows:
        # Not fatal: LightGBM routes missing values itself.  Said out loud so
        # it is a decision rather than a surprise.
        print("  note: %d of %d events have a non-finite input; LightGBM will "
              "route them itself." % (nan_rows, len(y)))
    return X, y, w, wp, tr, features


# ===========================================================================
# Measurement
# ===========================================================================

def curve(s, b):
    '''
    Cut scan on event counts -> (cuts, efficiency, rejection).

    Thresholds are the distinct background scores, so every point is a real
    place a cut could sit -- no interpolation artefacts where it matters most.
    '''
    cuts = np.unique(np.concatenate([b, [b.min() - 1.0, b.max() + 1.0]]))
    eff = np.array([(s >= c).mean() for c in cuts])
    rej = np.array([1.0 - (b >= c).mean() for c in cuts])
    return cuts, eff, rej


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


def eff_at_rejection_weighted(s, ws, b, wb, target):
    '''
    Same idea, but weighted -- the cut comes from a quantile of the background
    WEIGHT.  Used only for the rate cross-check.
    '''
    if wb.sum() <= 0 or ws.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(b)
    cw = np.cumsum(wb[order]) / wb.sum()
    k = min(int(np.searchsorted(cw, target)), len(b) - 1)
    cut = float(b[order][k])
    return (cut,
            float(ws[s >= cut].sum() / ws.sum()),
            float(1.0 - wb[b >= cut].sum() / wb.sum()))


def report_table(s, b, levels=REJ_LEVELS):
    '''Signal efficiency at fixed background rejection, with the raw counts.'''
    cuts, eff, rej = curve(s, b)
    print("\n%-10s | %9s | %8s | %12s | %s"
          % ("rejection", "eff", "cut", "sig kept", "bg kept"))
    print("-" * 66)
    rows = []
    for lvl in levels:
        k = kept_counts(s, b, lvl)
        if k is None:
            print("%-9.1f%% | %9s | %8s | %12s | %s"
                  % (100 * lvl, "n/a", "-", "-", "cut unreachable"))
            continue
        mark = "  <-- NOT A MEASUREMENT" if k["bg_kept"] < 10 else ""
        print("%-9.1f%% | %8.1f%% | %8.4f | %5d/%5d | %d/%d%s"
              % (100 * lvl, 100 * k["eff"], k["cut"],
                 k["sig_kept"], k["sig_total"],
                 k["bg_kept"], k["bg_total"], mark))
        rows.append(dict(rejection=lvl, **k))
    return rows


# ===========================================================================
# Plots
# ===========================================================================

def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def make_plots(scores, weights, tag, outdir, weight_name, n_bg_te):
    '''
    Three figures:
      <tag>_overtrain.png  train vs test score shapes
      <tag>_dist.png       score distribution, linear and log
      <tag>_cuts.png       cut curve and efficiency vs rejection
    '''
    plt = _setup_matplotlib()
    series = [("train_sig", "signal (train)", "tab:blue"),
              ("test_sig", "signal (test)", "tab:cyan"),
              ("train_bg", "background (train)", "tab:red"),
              ("test_bg", "background (test)", "tab:orange")]
    made = []

    def save(fig, name, suptitle=False):
        fig.tight_layout(rect=[0, 0, 1, 0.94] if suptitle else None)
        p = os.path.join(outdir, name)
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p)
        print("  -> %s" % p)

    rng_ = (0.0, 1.0)          # LightGBM outputs a probability

    # --- overtraining: shapes, normalised, so a size difference is not read
    #     as overtraining -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key, label, color in series:
        h, edges = np.histogram(scores[key], bins=60, range=rng_,
                                weights=weights[key], density=True)
        mid = 0.5 * (edges[1:] + edges[:-1])
        if key.startswith("train"):
            ax.hist(mid, bins=edges, weights=h, histtype="stepfilled",
                    alpha=.35, color=color, label=label)
        else:
            ax.plot(mid, h, "o", ms=3, color=color, label=label)
    ax.set_xlabel("P(signal)")
    ax.set_ylabel("normalised weight / bin")
    ax.set_yscale("log")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    ax.set_title("%s - train vs test score shape" % tag, fontsize=10)
    save(fig, "%s_overtrain.png" % tag)

    # --- score distribution ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, logy in zip(axes, (False, True)):
        for key, label, color in series:
            ax.hist(scores[key], bins=60, range=rng_, weights=weights[key],
                    histtype="step", lw=1.4, color=color, label=label)
        ax.set_xlabel("P(signal)")
        ax.set_ylabel("%s / bin" % weight_name)
        ax.grid(alpha=.3)
        if logy:
            ax.set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.suptitle("%s - score distribution (weight: %s)" % (tag, weight_name),
                 fontsize=11)
    save(fig, "%s_dist.png" % tag, suptitle=True)

    # --- cut curves ------------------------------------------------------
    s_te, b_te = scores["test_sig"], scores["test_bg"]
    cuts, eff, rej = curve(s_te, b_te)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    ax.plot(cuts, 100 * eff, color="tab:blue", lw=1.5, label="signal kept")
    ax.plot(cuts, 100 * rej, color="tab:red", lw=1.5,
            label="background rejected")
    if tag in DEFAULT_CUT:
        ax.axvline(DEFAULT_CUT[tag], color="k", ls=":", lw=1)
        ax.text(DEFAULT_CUT[tag], 50, " note cut %.2f" % DEFAULT_CUT[tag],
                fontsize=8, rotation=90, va="center")
    ax.set_xlabel("cut on P(signal)")
    ax.set_ylabel("% of events")
    ax.set_ylim(0, 102)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="center left")
    ax.set_title("where do I cut", fontsize=10)

    ax = axes[1]
    ax.plot(100 * rej, 100 * eff, lw=1.8, color="tab:green")
    # Where the test-set statistics runs out.  Right of "10 events left"
    # nothing on this curve is a measurement.
    for n_left, style in ((100, "-."), (10, "--"), (1, ":")):
        r = 100 * (1.0 - n_left / float(n_bg_te))
        ax.axvline(r, color="0.4", ls=style, lw=1)
        ax.text(r, 4, " %d bg events left" % n_left, rotation=90,
                fontsize=7, color="0.3", va="bottom")
    if tag == "noise":
        # Table 13 target of the technical note.
        ax.plot([99.2], [96.0], marker="*", ms=14, color="k", zorder=5)
        ax.text(99.2, 96.0, "  note target", fontsize=8, va="center")
    ax.set_xlabel("background rejection [%]")
    ax.set_ylabel("signal efficiency [%]")
    ax.set_xlim(80, 100)
    ax.set_ylim(0, 102)
    ax.grid(alpha=.3)
    ax.set_title("efficiency vs rejection", fontsize=10)

    fig.suptitle("%s - cut curves, efficiency counted in events" % tag,
                 fontsize=12)
    save(fig, "%s_cuts.png" % tag, suptitle=True)
    return made


# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="Train an L4 classifier (LightGBM)")
    ap.add_argument("--tag", required=True, choices=sorted(PARAMS),
                    help="which classifier: noise or muon")
    ap.add_argument("--dataset", required=True, help=".npz written by the notebook")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--features", default=None,
                    help="comma separated; defaults to the 'features' array "
                         "stored in the .npz")
    ap.add_argument("--target-rejection", type=float, default=None,
                    help="rejection the headline number is quoted at "
                         "(default 0.99 for noise, 0.94 for muon)")
    ap.add_argument("--gap-at", type=float, default=0.90,
                    help="rejection the train/test efficiency gap is measured "
                         "at.  0.90 by default because ~100 background events "
                         "sit above that cut; at 0.99 it is ~10 and the gap "
                         "would be mostly noise.")
    ap.add_argument("--valid-frac", type=float, default=0.2,
                    help="slice split off the TRAINING set for early stopping. "
                         "The test set is never used during training.")
    ap.add_argument("--no-early-stopping", action="store_true")
    ap.add_argument("--num-boost-round", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--min-data-in-leaf", type=int, default=None,
                    help="override Table 10; useful when the background "
                         "sample is small (see the warning this prints)")
    ap.add_argument("--plot-weight", choices=("weight", "w_phys"),
                    default="weight")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    target = args.target_rejection
    if target is None:
        target = 0.99 if args.tag == "noise" else 0.94

    X, y, w, wp, tr, features = load_dataset(args.dataset, args.features)
    te = ~tr
    sig, bg = y == 1, y == 0

    p = dict(PARAMS[args.tag])
    n_rounds = p.pop("num_boost_round", 2000)
    if args.num_boost_round is not None:
        n_rounds = args.num_boost_round
    if args.learning_rate is not None:
        p["learning_rate"] = args.learning_rate
    if args.min_data_in_leaf is not None:
        p["min_data_in_leaf"] = args.min_data_in_leaf
    p["seed"] = args.seed

    print("=== L4 %s classifier : LightGBM (Table 10) ===" % args.tag)
    print("  train  signal %8d   background %6d"
          % (int((tr & sig).sum()), int((tr & bg).sum())))
    print("  test   signal %8d   background %6d"
          % (int((te & sig).sum()), int((te & bg).sum())))
    print("  %d features: %s" % (len(features), ", ".join(features)))
    print("  parameters: %s" % ", ".join(
        "%s=%s" % (k, v) for k, v in sorted(p.items())
        if k not in ("objective", "metric", "verbosity", "deterministic")))
    print("  efficiency and rejection are EVENT COUNTS (w_phys not used)")

    # --- early stopping on a slice of TRAIN, never on test -----------------
    rng = np.random.default_rng(args.seed)
    fit = tr.copy()
    va = None
    if not args.no_early_stopping and args.valid_frac > 0:
        # Split per class so the class ratio is preserved: the background is
        # scarce and a blind random split could thin it further.
        for cls in (0.0, 1.0):
            idx = np.flatnonzero(tr & (y == cls))
            k = max(1, int(round(args.valid_frac * len(idx))))
            fit[rng.choice(idx, size=k, replace=False)] = False
        va = tr & ~fit
        print("  early stopping: %.0f%% validation slice off the TRAINING set "
              "(signal %d, background %d), metric %s"
              % (100 * args.valid_frac, int((va & sig).sum()),
                 int((va & bg).sum()), p.get("metric", "auc")))

    n_bg_fit = int((fit & bg).sum())
    min_leaf = p.get("min_data_in_leaf")
    if min_leaf and n_bg_fit < 10 * min_leaf:
        print("\n  !! WARNING: min_data_in_leaf=%d but only %d background "
              "events are fitted." % (min_leaf, n_bg_fit))
        print("     Table 10 assumes far more background statistics than we "
              "have.  If the result looks weak, re-run with")
        print("     --min-data-in-leaf lowered to tell 'not enough background' "
              "apart from 'the parameters do not fit our sample'.")

    dtrain = lgb.Dataset(X[fit], label=y[fit], weight=w[fit],
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
    # predict() uses best_iteration when early stopping fired, so that is the
    # model being scored -- report it, not the number of trees built.
    best = booster.best_iteration or booster.num_trees()
    print("  done -- scoring with %d trees (%d built, limit %d)"
          % (best, booster.num_trees(), n_rounds))
    if va is not None and best >= n_rounds:
        print("  note: early stopping never fired; the round limit was hit, "
              "so the model may still be improving.")

    # --- evaluation, TEST ONLY --------------------------------------------
    def score(mask):
        return booster.predict(X[mask])

    s_tr, b_tr = score(tr & sig), score(tr & bg)
    s_te, b_te = score(te & sig), score(te & bg)

    print("\n--- Signal efficiency vs background rejection (TEST set) ---")
    rows = report_table(s_te, b_te)

    k_te = kept_counts(s_te, b_te, target)
    print("\n--- HEADLINE ---")
    if k_te is None:
        print("  %.1f%% rejection is not reachable on this test set."
              % (100 * target))
    else:
        print("  signal efficiency at %.1f%% background rejection: %.1f%%"
              % (100 * target, 100 * k_te["eff"]))
        print("  (cut %.4f, %d/%d background events left)"
              % (k_te["cut"], k_te["bg_kept"], k_te["bg_total"]))
        if k_te["bg_kept"] < 10:
            print("  !! fewer than 10 background events define this point -- "
                  "it is not a measurement.")

    g_tr = kept_counts(s_tr, b_tr, args.gap_at)
    g_te = kept_counts(s_te, b_te, args.gap_at)
    gap = float("nan")
    print("\n--- Overtraining ---")
    if g_tr is None or g_te is None:
        print("  gap at %.0f%% rejection: not reachable" % (100 * args.gap_at))
    else:
        gap = g_tr["eff"] - g_te["eff"]
        print("  eff at %.0f%% rejection: train %.1f%%  test %.1f%%  gap %+.1f%s"
              % (100 * args.gap_at, 100 * g_tr["eff"], 100 * g_te["eff"],
                 100 * gap, "  <-- MEMORISING" if gap > 0.05 else ""))

    cut_w, eff_w, rej_w = eff_at_rejection_weighted(
        s_te, wp[te & sig], b_te, wp[te & bg], target)
    print("\n--- Rate-weighted cross-check (w_phys, not the headline) ---")
    if np.isfinite(eff_w):
        print("  efficiency %.1f%% at %.2f%% rejection (cut %.4f)"
              % (100 * eff_w, 100 * rej_w, cut_w))
    else:
        print("  not computable (zero total weight)")

    # The note's own cut, on the note's own scale -- directly comparable
    # because this is a LightGBM probability.
    if args.tag in DEFAULT_CUT:
        c = DEFAULT_CUT[args.tag]
        print("\n--- At the note's cut (%.2f) ---" % c)
        print("  signal efficiency %.1f%%, background rejection %.2f%% "
              "(%d background events left)"
              % (100 * (s_te >= c).mean(), 100 * (1 - (b_te >= c).mean()),
                 int((b_te >= c).sum())))

    print("\n--- Feature importance (gain) ---")
    imp = booster.feature_importance(importance_type="gain", iteration=best)
    tot = imp.sum() or 1.0
    importance = {}
    for f, v in sorted(zip(features, imp), key=lambda x: -x[1]):
        importance[f] = float(v)
        print("  %-22s %8.1f  (%.1f%%)" % (f, v, 100 * v / tot))

    # --- outputs -----------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    model_path = os.path.join(args.outdir, "L4_%s_model.txt" % args.tag)
    booster.save_model(model_path, num_iteration=best)
    print("\n  -> %s" % model_path)

    plots = []
    if not args.no_plots:
        scores = {"train_sig": s_tr, "test_sig": s_te,
                  "train_bg": b_tr, "test_bg": b_te}
        src = wp if args.plot_weight == "w_phys" else w
        weights = {"train_sig": src[tr & sig], "test_sig": src[te & sig],
                   "train_bg": src[tr & bg], "test_bg": src[te & bg]}
        print("\n--- Plots ---")
        try:
            plots = make_plots(scores, weights, args.tag, args.outdir,
                               args.plot_weight, len(b_te))
        except Exception as exc:
            # The numbers are printed and the model is saved by now; a drawing
            # failure must not take those down with it.
            print("  [!] plotting failed (%s: %s)" % (type(exc).__name__, exc))
            print("      The model and the numbers above are unaffected.")

    meta = dict(
        tag=args.tag,
        features=features,
        params=p,
        num_boost_round_limit=n_rounds,
        n_trees=int(best),
        trained_at=datetime.datetime.now().isoformat(timespec="seconds"),
        lightgbm_version=lgb.__version__,
        dataset=os.path.abspath(args.dataset),
        n_train_sig=int((tr & sig).sum()), n_train_bg=int((tr & bg).sum()),
        n_test_sig=int((te & sig).sum()), n_test_bg=int((te & bg).sum()),
        target_rejection=target,
        metrics=dict(
            eff_at_target=(None if k_te is None else k_te["eff"]),
            cut_at_target=(None if k_te is None else k_te["cut"]),
            bg_kept_at_target=(None if k_te is None else k_te["bg_kept"]),
            gap_at=args.gap_at,
            gap=(None if not np.isfinite(gap) else float(gap)),
            eff_weighted_at_target=(None if not np.isfinite(eff_w)
                                    else float(eff_w)),
            table=rows,
        ),
        importance_gain=importance,
        default_cut=DEFAULT_CUT.get(args.tag),
        plots=plots,
    )
    meta_path = os.path.join(args.outdir, "L4_%s_model.json" % args.tag)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print("  -> %s" % meta_path)


if __name__ == "__main__":
    main()
