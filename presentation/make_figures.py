#!/usr/bin/env python
'''
Produce the presentation figures that need nothing but files already on disk.

    python presentation/make_figures.py

    python presentation/make_figures.py --production pass2

Reads (<suf> is "" for pass3 and "_pass2" for pass2 -- the same rule the
notebook's cell 3 applies, so the two productions never overwrite each other)
    L4_output/models<suf>/L4_noise_model.json   the trained model's metadata
    L4_output/ds<suf>/L4_noise_dataset.npz      the training set
    L4_output/models<suf>/noise_*.png           the plots train_L4_classifier wrote

Writes into presentation/figures/
    4.png                       input distributions, signal vs background,
                                the figure to hold beside the note's Fig 13
    feature_importance.png      gain split                       (slide 10)
    input_correlation_signal.png      rank correlation of the 5 inputs,
    input_correlation_background.png  one file per class               (slide 7)
    lightgbm_cuts.png           copied from noise_cuts.png        (slide 9)
    lightgbm_score_dist.png     copied from noise_dist.png        (slide 8)

Dependencies: numpy + matplotlib.  No lightgbm, no icetray -- this runs in a
plain python as well as inside env-shell.

If a source file is missing the script says which one and carries on, so it is
safe to run before the training has been redone.
'''

import os
import sys
import json
import shutil
import argparse
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def default_out_root():
    return os.environ.get("OSCNEXT_OUT_ROOT", os.path.join(REPO, "L4_output"))


# ---------------------------------------------------------------------------
# 1. feature importance
# ---------------------------------------------------------------------------

def feature_importance(json_path, out_path):
    '''Horizontal bar chart of the gain split, read from the model sidecar.'''
    with open(json_path) as fh:
        meta = json.load(fh)
    imp = meta.get("importance_gain")
    if not imp:
        print("  [!] no importance_gain in %s" % json_path)
        return False

    total = sum(imp.values()) or 1.0
    names, vals = zip(*sorted(imp.items(), key=lambda kv: kv[1]))
    pct = [100.0 * v / total for v in vals]

    fig, ax = plt.subplots(figsize=(5.0, 2.5))
    bars = ax.barh(range(len(names)), pct, color="tab:green", height=0.65)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("gain [%]", fontsize=9)
    ax.set_xlim(0, max(pct) * 1.18)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", alpha=.3)
    ax.set_axisbelow(True)
    for b, p in zip(bars, pct):
        ax.text(b.get_width() + max(pct) * 0.015, b.get_y() + b.get_height() / 2,
                "%.1f%%" % p, va="center", fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print("  -> %s" % out_path)
    return True


# ---------------------------------------------------------------------------
# 2. input correlation
# ---------------------------------------------------------------------------

def _rank(a):
    '''Ranks with ties averaged -- enough for a Spearman coefficient.'''
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average the ranks of tied values (micro_count and NchCleaned are integers,
    # so ties are the rule here, not the exception)
    vals = a[order]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def _spearman(X):
    '''Rank-correlation matrix.

    Pearson would be the wrong choice: iLineFit_speed spans three decades,
    micro_count and NchCleaned are small integers, and fill_ratio piles up near
    zero.  A rank correlation measures monotone association without caring
    about those shapes.
    '''
    R = np.column_stack([_rank(X[:, j]) for j in range(X.shape[1])])
    return np.corrcoef(R.T)


def input_correlation(npz_path, out_dir):
    """Rank correlation of the BDT inputs, one PNG per class.

    Separately, because a pair can be correlated in one class and not the
    other -- and the question this plot answers (is a low-gain variable
    redundant, or just weak?) is about the training data as the model sees it.

    One file per class rather than two panels in one figure: side by side the
    tick labels are longer than the axis they sit on, and `FullTimeLengthRatio`
    ran into its neighbour.  Alone, each panel has room for readable labels.
    """
    z = np.load(npz_path, allow_pickle=False)
    names = [str(f) for f in z["features"]]
    X = np.column_stack([np.asarray(z[n], dtype=float) for n in names])
    y = np.asarray(z["label"], dtype=float)

    finite = np.isfinite(X).all(axis=1)
    n_drop = int((~finite).sum())
    if n_drop:
        print("  (%d of %d events dropped: a non-finite input)"
              % (n_drop, len(y)))

    made = []
    for label, mask in (("signal", finite & (y == 1)),
                        ("background", finite & (y == 0))):
        out_path = os.path.join(out_dir, "input_correlation_%s.png" % label)
        if mask.sum() < 10:
            print("  [!] %s: only %d events, skipped" % (label, int(mask.sum())))
            continue
        C = _spearman(X[mask])

        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right",
                           rotation_mode="anchor", fontsize=9)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title("%s  (%d events)" % (label, int(mask.sum())), fontsize=11)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, "%.2f" % C[i, j], ha="center", va="center",
                        fontsize=8.5,
                        color="white" if abs(C[i, j]) > 0.55 else "black")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("Spearman rank correlation", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print("  -> %s" % out_path)
        made.append(out_path)
    return len(made)


def input_distributions(npz_path, figures, tag):
    """Delegate to scripts/plot_inputs.py rather than reimplement it.

    That script owns the axis choices taken from the note (Figure 13's
    logarithmic iLineFit axis, the bounded ratios) and the per-panel AUC and
    NaN annotations.  A second copy here would drift from it, and this figure
    is precisely the one the deck holds beside Figure 13 -- it has to be the
    same plot the analysis looked at, not a lookalike.
    """
    script = os.path.join(REPO, "scripts", "plot_inputs.py")
    if not os.path.exists(script):
        print("  [!] not found: %s" % script)
        return False
    r = subprocess.run([sys.executable, script, npz_path,
                        "--outdir", figures, "--tag", tag],
                       capture_output=True, text=True)
    sys.stdout.write("".join("    " + l + "\n"
                             for l in r.stdout.strip().split("\n") if l))
    if r.returncode != 0:
        print("  [!] plot_inputs.py failed:\n%s" % (r.stderr or "")[-1500:])
        return False
    src = os.path.join(figures, "%s_inputs.png" % tag)
    if not os.path.exists(src):
        print("  [!] plot_inputs.py wrote nothing at %s" % src)
        return False
    # The deck includes it as 4.png (FIGURES.md: the numeric names are
    # historical and carry no meaning); keep the descriptive name too.
    shutil.copyfile(src, os.path.join(figures, "4.png"))
    print("  -> %s" % os.path.join(figures, "4.png"))
    return True


def copy_as(src, dst):
    if not os.path.exists(src):
        print("  [!] not found: %s" % src)
        print("      run: python scripts/train_L4_classifier.py --tag noise \\")
        print("               --dataset <.npz> --outdir <models dir>")
        return False
    shutil.copyfile(src, dst)
    print("  -> %s   (from %s)" % (dst, os.path.basename(src)))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=None,
                    help="L4_output directory (default: $OSCNEXT_OUT_ROOT, "
                         "else <repo>/L4_output)")
    ap.add_argument("--tag", default="noise", help="classifier tag (noise|muon)")
    ap.add_argument("--production", default="pass3", choices=("pass2", "pass3"),
                    help="which production's output tree to read.  pass3 keeps "
                         "the bare directory names it has always used; pass2 "
                         "reads ds_pass2/ and models_pass2/, exactly as cell 3 "
                         "of the notebook writes them.")
    ap.add_argument("--figures", default=os.path.join(HERE, "figures"),
                    help="where the figures are written")
    args = ap.parse_args()

    out_root = args.out_root or default_out_root()
    suf = "" if args.production == "pass3" else "_" + args.production
    models = os.path.join(out_root, "models" + suf)
    ds = os.path.join(out_root, "ds" + suf, "L4_%s_dataset.npz" % args.tag)
    meta = os.path.join(models, "L4_%s_model.json" % args.tag)
    os.makedirs(args.figures, exist_ok=True)

    print("production  : %s" % args.production)
    print("output root : %s" % out_root)
    print("models      : %s" % models)
    print("dataset     : %s" % ds)
    print("figures     : %s" % args.figures)
    print()

    ok = 0

    print("1. input distributions (slide 5, beside the note's Figure 13)")
    if os.path.exists(ds):
        ok += input_distributions(ds, args.figures, args.tag)
    else:
        print("  [!] not found: %s" % ds)
        print("      section 6 of the notebook writes it")

    print("\n2. feature importance")
    if os.path.exists(meta):
        ok += feature_importance(meta, os.path.join(args.figures,
                                                    "feature_importance.png"))
    else:
        print("  [!] not found: %s" % meta)
        print("      the training writes it -- run train_L4_classifier.py first")

    print("\n3. input correlation")
    if os.path.exists(ds):
        ok += input_correlation(ds, args.figures)
    else:
        print("  [!] not found: %s" % ds)
        print("      section 6 of the notebook writes it")

    print("\n4. plots written by the training")
    ok += copy_as(os.path.join(models, "%s_cuts.png" % args.tag),
                  os.path.join(args.figures, "lightgbm_cuts.png"))
    ok += copy_as(os.path.join(models, "%s_dist.png" % args.tag),
                  os.path.join(args.figures, "lightgbm_score_dist.png"))

    print("\n%d of 6 figures produced." % ok)
    if ok < 6:
        print("The deck compiles either way -- a missing figure becomes a box "
              "naming the file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
