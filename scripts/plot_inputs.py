#!/usr/bin/env python
'''
BDT input variable distributions, signal vs background -- our counterpart to
Figure 13 of the technical note.

WHY THIS IS SEPARATE FROM THE TRAINER
  train_L4_classifier.py plots what the MODEL does (score shapes, cut curves).
  This plots what the model is FED, which is the figure a write-up needs first
  and the only one that can be compared with the note directly.  It reads the
  .npz, so it needs neither the model nor a reload of the HDF5 -- it can be run
  while a training is still going.

WHAT IT SHOWS PER PANEL
  Shapes normalised to unit area, so the 164:1 class imbalance is not read as
  separation.  Signal is a filled step and background a plain line: identity is
  carried by the mark as well as the colour, so the panels survive a colour-blind
  reader and a greyscale print.  Each title carries the single-variable AUC --
  what that one variable separates on its own, which is the honest companion to
  the model's feature importance (a variable can have high importance because it
  is used late in the tree while separating little by itself).

  The NaN fraction is printed per panel.  LightGBM routes missing values itself,
  so a variable that is entirely NaN trains without complaint and contributes
  nothing; that is invisible in every other plot we make.

USAGE
    python scripts/plot_inputs.py <dataset.npz> --outdir <dir>
    python scripts/plot_inputs.py <dataset.npz> --outdir <dir> --weight none
'''
from __future__ import print_function

import argparse
import os

import numpy as np

# Signal blue / background red: the repo's existing series colours.  Checked
# rather than assumed -- the pair passes the lightness band, the chroma floor,
# a protanopia separation of dE 21.1 (target 8) and 3:1 contrast on white.
C_SIG, C_BG = "tab:blue", "tab:red"

# How each variable wants to be drawn.  Taken from the note where it says so:
# Figure 13's iLineFit axis is logarithmic over 1e-3..1e3 m/ns, and the two
# ratios are bounded by construction.  Anything not listed falls back to a
# robust percentile range, so a new variable plots sensibly with no edit here.
AXES = {
    "iLineFit_speed":      dict(logx=True, lo=1e-3, hi=1e3),
    "fill_ratio":          dict(lo=0.0, hi=1.0),
    "FullTimeLengthRatio": dict(lo=0.0, hi=1.0),
    "NchCleaned":          dict(integer=True),
    "micro_count":         dict(integer=True),
    "ICVetoHits":          dict(integer=True),
    "NAbove200Hits":       dict(integer=True),
    "RTVeto250Hits":       dict(integer=True),
    "VICH_nch":            dict(integer=True),
}


def single_auc(s, b, rng, cap=200000):
    """Rank-based AUC of one variable, on a subsample so it stays cheap."""
    s = s[np.isfinite(s)]
    b = b[np.isfinite(b)]
    if len(s) == 0 or len(b) == 0:
        return float("nan")
    if len(s) > cap:
        s = rng.choice(s, cap, replace=False)
    if len(b) > cap:
        b = rng.choice(b, cap, replace=False)
    z = np.concatenate([s, b])
    r = np.empty(len(z))
    r[np.argsort(z, kind="mergesort")] = np.arange(1, len(z) + 1)
    # average ranks over ties, so a discrete variable is not flattered
    _, inv, cnt = np.unique(z, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    r = (sums / cnt)[inv]
    n_s, n_b = len(s), len(b)
    auc = (r[:n_s].sum() - n_s * (n_s + 1) / 2.0) / (float(n_s) * n_b)
    return max(auc, 1.0 - auc)      # direction is irrelevant for separation


def edges_for(name, vals, nbins):
    cfg = AXES.get(name, {})
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return np.linspace(0, 1, nbins + 1), False
    logx = bool(cfg.get("logx"))
    lo = cfg.get("lo")
    hi = cfg.get("hi")
    if lo is None or hi is None:
        # robust range: the tails of a low-energy distribution are long and a
        # single outlier would otherwise squeeze every panel into one bin
        lo = np.percentile(finite, 0.5) if lo is None else lo
        hi = np.percentile(finite, 99.5) if hi is None else hi
    if hi <= lo:
        hi = lo + 1.0
    if logx:
        pos = finite[finite > 0]
        if len(pos) == 0:
            logx = False
        else:
            lo = max(lo, pos.min())
            return np.logspace(np.log10(lo), np.log10(hi), nbins + 1), True
    if cfg.get("integer"):
        n = int(min(nbins, max(2, round(hi - lo) + 1)))
        return np.arange(int(np.floor(lo)), int(np.floor(lo)) + n + 1) - 0.5, False
    return np.linspace(lo, hi, nbins + 1), False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help=".npz written by section 6 of the notebook")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default=None,
                    help="name for the output file; taken from the .npz name "
                         "when not given")
    ap.add_argument("--weight", choices=("w_phys", "weight", "none"),
                    default="w_phys",
                    help="w_phys (default) gives the rate-weighted shape the "
                         "note's figures show; none gives raw event counts")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                         "axes.grid": True, "grid.alpha": 0.25,
                         "axes.axisbelow": True})

    d = np.load(args.dataset, allow_pickle=False)
    features = [str(f) for f in d["features"]]
    label = d["label"]
    sig, bg = label == 1, label == 0
    tag = args.tag or os.path.basename(args.dataset).replace("L4_", "") \
                                                    .replace("_dataset.npz", "")

    if args.weight == "none":
        w = np.ones(len(label))
        wname = "events"
    else:
        w = np.asarray(d[args.weight], dtype=np.float64).copy()
        w[~np.isfinite(w) | (w < 0)] = 0.0
        wname = args.weight

    rng = np.random.default_rng(args.seed)
    ncol = 3 if len(features) > 4 else len(features)
    nrow = int(np.ceil(len(features) / float(ncol)))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.2 * nrow),
                             squeeze=False)

    print("%-24s %8s  %9s  %9s" % ("variable", "AUC", "NaN sig", "NaN bg"))
    for k, name in enumerate(features):
        ax = axes[k // ncol][k % ncol]
        v = np.asarray(d[name], dtype=np.float64)
        e, logx = edges_for(name, v, args.bins)

        for mask, colour, lab, filled in ((sig, C_SIG, "signal", True),
                                          (bg, C_BG, "background", False)):
            h, _ = np.histogram(v[mask], bins=e, weights=w[mask])
            tot = h.sum()
            if tot > 0:
                h = h / tot / np.diff(e)     # unit area, density
            mid = 0.5 * (e[1:] + e[:-1])
            if filled:
                ax.hist(mid, bins=e, weights=h, histtype="stepfilled",
                        color=colour, alpha=0.30, lw=1.4, edgecolor=colour,
                        label=lab)
            else:
                ax.hist(mid, bins=e, weights=h, histtype="step",
                        color=colour, lw=1.7, label=lab)

        nan_s = float(np.mean(~np.isfinite(v[sig]))) * 100
        nan_b = float(np.mean(~np.isfinite(v[bg]))) * 100
        auc = single_auc(v[sig], v[bg], rng)
        print("%-24s %8.4f  %8.2f%%  %8.2f%%" % (name, auc, nan_s, nan_b))

        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(name)
        ax.set_ylabel("normalised %s / bin" % wname)
        title = "AUC %.3f" % auc
        if nan_s > 0.01 or nan_b > 0.01:
            title += "   NaN %.1f/%.1f%%" % (nan_s, nan_b)
        ax.set_title(title, fontsize=8.5, color="0.35")

    for k in range(len(features), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    axes[0][0].legend(fontsize=8, frameon=False)

    fig.suptitle("L4 %s BDT inputs -- shapes, %s-weighted" % (tag, wname),
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "%s_inputs.png" % tag)
    fig.savefig(out)
    print("\n  -> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
