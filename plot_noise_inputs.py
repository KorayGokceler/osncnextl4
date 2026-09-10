#!/usr/bin/env python
'''
Distributions of the noise BDT inputs -- for comparison against Figures
12/13 of the technical note.

For each variable: x axis is the variable's value, y axis is the RATE
[Hz/bin] (weighted by w_phys), with two curves -- neutrino (nue+numu) and
noise.  Same layout as Figures 12/13 in the note.

WHY: we verified the code path of all 5 inputs (the note's definition plus
the original pass2 code) but never compared the VALUES THEY PRODUCE against
the reference.  "We call the right module with the right parameters" and
"we produce the same numbers as the reference" are different claims.

iLineFit_speed is the prime suspect: in the diagnostic run its standalone
separation power came out at 0.1% (a random cut gives 1%) and the signal /
noise medians are nearly identical (0.137 / 0.144).  In Figure 13 this
variable spans a log x axis from 10^-3 to 10^3 -- SIX DECADES.  If ours is
squeezed into one decade, we compute it differently.

Reads the .ds files, i.e. exactly the data the BDT saw; train and test are
merged (splitting them is meaningless for a distribution).  Signal is
nue+numu combined (.ds carries no sample label).

Usage:

    python plot_noise_inputs.py --ds-dir L4_output/ds --tag L4_noise \
        --features NchCleaned,micro_count,iLineFit_speed,fill_ratio,FullTimeLengthRatio \
        --outdir L4_output/plots
'''

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from pybdt import util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Axis ranges as given by the note.  lo/hi None -> derived from the data.
#   Figure 13: L4_iLineFit.speed  -> log, 10^-3 .. 10^3
#              L4_fill_ratio.fillratio_from_mean -> 0.0 .. 1.0
#              IC2018_LE_L3_Vars.FullTimeLengthRatio -> 0.0 .. 1.0
#   Figure 12: NchCleaned, micro_count -> the axis range could not be read
#              reliably from the PDF text, so it is derived from the data.
AXES = {
    "iLineFit_speed":      dict(log=True,  lo=1e-3, hi=1e3,
                                note="Fig 13: log, 10^-3 .. 10^3 m/ns"),
    "fill_ratio":          dict(log=False, lo=0.0, hi=1.0,
                                note="Fig 13: 0.0 .. 1.0"),
    "FullTimeLengthRatio": dict(log=False, lo=0.0, hi=1.0,
                                note="Fig 13: 0.0 .. 1.0"),
    "NchCleaned":          dict(log=False, lo=None, hi=None,
                                note="Fig 12 (range not readable from the note)"),
    "micro_count":         dict(log=False, lo=None, hi=None,
                                note="Fig 12 (range not readable from the note)"),
}

PCTS = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]


def load_side(ds_dir, tag, side, features):
    '''Merge train + test -- splitting them is meaningless for a distribution.'''
    xs, ws = [], []
    for split in ("train", "test"):
        p = os.path.join(ds_dir, "%s_%s_%s.ds" % (tag, side, split))
        if not os.path.exists(p):
            sys.exit("not found: %s" % p)
        ds = util.load(p)
        names = list(ds.names)
        xs.append(np.column_stack([np.asarray(ds[f], dtype=float)
                                   for f in features]))
        wk = "w_phys" if "w_phys" in names else "weight"
        ws.append(np.nan_to_num(np.asarray(ds[wk], dtype=float)))
    return np.vstack(xs), np.concatenate(ws)


def describe(name, s, ws, b, wb, cfg):
    '''Percentiles plus range checks -- the numbers behind the plot.'''
    print("\n" + "-" * 78)
    print("%s   (%s)" % (name, cfg["note"]))
    print("-" * 78)
    print("%-10s %s" % ("percentile",
                        "".join("%10s" % ("%g%%" % p) for p in PCTS)))
    for tag, v in (("neutrino", s), ("noise", b)):
        q = np.nanpercentile(v[np.isfinite(v)], PCTS)
        print("%-10s %s" % (tag, "".join("%10.4g" % x for x in q)))

    for tag, v in (("neutrino", s), ("noise", b)):
        fin = np.isfinite(v)
        n_bad = int((~fin).sum())
        n_zero = int((v[fin] == 0).sum())
        n_neg = int((v[fin] < 0).sum())
        pos = fin & (v > 0)
        span = (np.log10(np.nanmax(v[pos]) / np.nanmin(v[pos]))
                if pos.sum() > 1 else float("nan"))
        print("  %-9s NaN/inf=%-6d  exactly zero=%-6d  negative=%-6d  "
              "positive values span %.1f decades"
              % (tag, n_bad, n_zero, n_neg, span))

    if cfg["lo"] is not None:
        for tag, v in (("neutrino", s), ("noise", b)):
            fin = np.isfinite(v)
            out = int(((v[fin] < cfg["lo"]) | (v[fin] > cfg["hi"])).sum())
            flag = "" if out == 0 else "   <-- outside the note's axis"
            print("  %-9s outside the note's range [%g, %g]: %d / %d%s"
                  % (tag, cfg["lo"], cfg["hi"], out, int(fin.sum()), flag))


def panel(ax, name, s, ws, b, wb, cfg, bins):
    lo, hi = cfg["lo"], cfg["hi"]
    both = np.concatenate([s[np.isfinite(s)], b[np.isfinite(b)]])
    if lo is None:
        lo, hi = np.percentile(both, [0.1, 99.9])
        if hi <= lo:
            lo, hi = both.min(), both.max() + 1e-9

    if cfg["log"]:
        lo = max(lo, 1e-12)
        edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
        ax.set_xscale("log")
    else:
        edges = np.linspace(lo, hi, bins + 1)

    for v, w, lab, color in ((b, wb, "noise", "tab:red"),
                             (s, ws, "neutrino", "tab:blue")):
        m = np.isfinite(v)
        ax.hist(v[m], bins=edges, weights=w[m], histtype="step",
                lw=1.5, color=color, label=lab)

    ax.set_yscale("log")
    ax.set_xlabel(name)
    ax.set_ylabel("rate [Hz / bin]")
    ax.grid(alpha=.3)
    ax.set_title(cfg["note"], fontsize=8)


def main():
    ap = argparse.ArgumentParser(
        description="Rate histograms of the noise BDT inputs (Fig 12/13 style)")
    ap.add_argument("--ds-dir", required=True)
    ap.add_argument("--tag", default="L4_noise")
    ap.add_argument("--features", required=True, help="comma separated")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--bins", type=int, default=60)
    args = ap.parse_args()

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    Xs, ws = load_side(args.ds_dir, args.tag, "sig", features)
    Xb, wb = load_side(args.ds_dir, args.tag, "bg", features)

    print("=== %s input distributions ===" % args.tag)
    print("  neutrino %d events,  total rate %.4e Hz (%.1f mHz)"
          % (len(Xs), ws.sum(), 1e3 * ws.sum()))
    print("  noise    %d events,  total rate %.4e Hz (%.1f mHz)"
          % (len(Xb), wb.sum(), 1e3 * wb.sum()))
    print("  (train + test merged; weighted by w_phys)")

    ncol = 3
    nrow = int(np.ceil(len(features) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for i, f in enumerate(features):
        cfg = AXES.get(f, dict(log=False, lo=None, hi=None, note=""))
        s, b = Xs[:, i], Xb[:, i]
        describe(f, s, ws, b, wb, cfg)
        panel(axes[i], f, s, ws, b, wb, cfg, args.bins)

    axes[0].legend(fontsize=9)
    for j in range(len(features), len(axes)):
        axes[j].axis("off")

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "%s_input_distributions.png" % args.tag)
    fig.suptitle("%s inputs - rate vs variable (technical note Fig 12/13 style)"
                 % args.tag, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)
    print("\n  -> %s" % out)

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("* 'positive values span N decades': Figure 13 shows SIX decades for")
    print("  iLineFit_speed.  If ours spans 1-2, we compute it differently.")
    print("* 'exactly zero' count: if failed fits arrive as zero, that is")
    print("  precisely how a variable loses its separation power.")
    print("* 'outside the note's range': a systematic shift shows up here.")
    print("* If the two curves overlap everywhere, that variable separates")
    print("  nothing.")


if __name__ == "__main__":
    main()
