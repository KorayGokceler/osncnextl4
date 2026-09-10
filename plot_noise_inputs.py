#!/usr/bin/env python
'''
Noise BDT girdilerinin dagilimlari -- teknik not Sekil 12/13 ile
karsilastirmak icin.

Her degisken icin: x ekseni degiskenin degeri, y ekseni ORAN [Hz/bin]
(w_phys ile agirlikli), iki egri -- notrino (nue+numu) ve gurultu.
Notun Sekil 12/13'undeki bicimin aynisi.

NEDEN: 5 girdinin kod yolunu dogruladik (nottaki tanim + orijinal pass2
kodu) ama URETTIKLERI DEGERLERI referansla hic karsilastirmadik.  "Dogru
modulu dogru parametrelerle cagiriyoruz" ile "referansla ayni sayiyi
uretiyoruz" farkli seyler.

Ozellikle iLineFit_speed suphede: teshis kosusunda tek basina ayirt etme
gucu %0.1 cikti (rastgele kesim %1 verir) ve sinyal/gurultu medyanlari
neredeyse ayni (0.137 / 0.144).  Sekil 13'te bu degiskenin x ekseni
log olcekte 10^-3 .. 10^3, yani ALTI KADEME.  Bizimki tek kademeye
sikismissa hesap farkli demektir.

Girdi olarak .ds dosyalarini okur -- yani BDT'nin gordugu tam veriyi;
train ve test birlestirilir (dagilim icin ayirmanin anlami yok).
Sinyal nue+numu birlesik (.ds ornek etiketi tasimiyor).

Kullanim:

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


# Notun verdigi eksen araliklari.  lo/hi None ise veriden turetilir.
#   Sekil 13: L4_iLineFit.speed  -> log, 10^-3 .. 10^3
#             L4_fill_ratio.fillratio_from_mean -> 0.0 .. 1.0
#             IC2018_LE_L3_Vars.FullTimeLengthRatio -> 0.0 .. 1.0
#   Sekil 12: NchCleaned, micro_count -> eksen araligi PDF metninden
#             guvenilir okunamadi, veriden turetiliyor.
AXES = {
    "iLineFit_speed":      dict(log=True,  lo=1e-3, hi=1e3,
                                note="Sekil 13: log, 10^-3 .. 10^3 m/ns"),
    "fill_ratio":          dict(log=False, lo=0.0, hi=1.0,
                                note="Sekil 13: 0.0 .. 1.0"),
    "FullTimeLengthRatio": dict(log=False, lo=0.0, hi=1.0,
                                note="Sekil 13: 0.0 .. 1.0"),
    "NchCleaned":          dict(log=False, lo=None, hi=None,
                                note="Sekil 12 (aralik nottan okunamadi)"),
    "micro_count":         dict(log=False, lo=None, hi=None,
                                note="Sekil 12 (aralik nottan okunamadi)"),
}

PCTS = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]


def load_side(ds_dir, tag, side, features):
    '''train + test birlestir -- dagilim icin ayirmanin anlami yok.'''
    xs, ws = [], []
    for split in ("train", "test"):
        p = os.path.join(ds_dir, "%s_%s_%s.ds" % (tag, side, split))
        if not os.path.exists(p):
            sys.exit("bulunamadi: %s" % p)
        ds = util.load(p)
        names = list(ds.names)
        xs.append(np.column_stack([np.asarray(ds[f], dtype=float)
                                   for f in features]))
        wk = "w_phys" if "w_phys" in names else "weight"
        ws.append(np.nan_to_num(np.asarray(ds[wk], dtype=float)))
    return np.vstack(xs), np.concatenate(ws)


def describe(name, s, ws, b, wb, cfg):
    '''Yuzdelikler + notun araligina uyum -- grafigin sayisal karsiligi.'''
    print("\n" + "-" * 78)
    print("%s   (%s)" % (name, cfg["note"]))
    print("-" * 78)
    print("%-10s %s" % ("yuzdelik", "".join("%10s" % ("%g%%" % p) for p in PCTS)))
    for tag, v in (("notrino", s), ("gurultu", b)):
        q = np.nanpercentile(v[np.isfinite(v)], PCTS)
        print("%-10s %s" % (tag, "".join("%10.4g" % x for x in q)))

    for tag, v in (("notrino", s), ("gurultu", b)):
        fin = np.isfinite(v)
        n_bad = int((~fin).sum())
        n_zero = int((v[fin] == 0).sum())
        n_neg = int((v[fin] < 0).sum())
        span = (np.log10(np.nanmax(v[fin & (v > 0)]) /
                         np.nanmin(v[fin & (v > 0)]))
                if (fin & (v > 0)).sum() > 1 else float("nan"))
        print("  %-8s NaN/inf=%-6d  tam sifir=%-6d  negatif=%-6d  "
              "pozitif deger araligi=%.1f kademe"
              % (tag, n_bad, n_zero, n_neg, span))

    if cfg["lo"] is not None:
        for tag, v in (("notrino", s), ("gurultu", b)):
            fin = np.isfinite(v)
            out = int(((v[fin] < cfg["lo"]) | (v[fin] > cfg["hi"])).sum())
            flag = "" if out == 0 else "   <-- notun ekseni disinda"
            print("  %-8s notun araligi [%g, %g] disinda: %d / %d%s"
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

    for v, w, lab, color in ((b, wb, "gurultu", "tab:red"),
                             (s, ws, "notrino", "tab:blue")):
        m = np.isfinite(v)
        ax.hist(v[m], bins=edges, weights=w[m], histtype="step",
                lw=1.5, color=color, label=lab)

    ax.set_yscale("log")
    ax.set_xlabel(name)
    ax.set_ylabel("oran [Hz / bin]")
    ax.grid(alpha=.3)
    ax.set_title(cfg["note"], fontsize=8)


def main():
    ap = argparse.ArgumentParser(
        description="Noise BDT girdilerinin oran histogramlari (Sekil 12/13 bicimi)")
    ap.add_argument("--ds-dir", required=True)
    ap.add_argument("--tag", default="L4_noise")
    ap.add_argument("--features", required=True, help="virgulle ayrilmis")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--bins", type=int, default=60)
    args = ap.parse_args()

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    Xs, ws = load_side(args.ds_dir, args.tag, "sig", features)
    Xb, wb = load_side(args.ds_dir, args.tag, "bg", features)

    print("=== %s girdi dagilimlari ===" % args.tag)
    print("  notrino %d olay,  toplam oran %.4e Hz (%.1f mHz)"
          % (len(Xs), ws.sum(), 1e3 * ws.sum()))
    print("  gurultu %d olay,  toplam oran %.4e Hz (%.1f mHz)"
          % (len(Xb), wb.sum(), 1e3 * wb.sum()))
    print("  (train + test birlesik; agirlik w_phys)")

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
    out = os.path.join(args.outdir, "%s_girdi_dagilimlari.png" % args.tag)
    fig.suptitle("%s girdileri — oran vs degisken (teknik not Sekil 12/13 bicimi)"
                 % args.tag, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)
    print("\n  -> %s" % out)

    print("\n" + "=" * 78)
    print("NASIL OKUNACAK")
    print("=" * 78)
    print("* 'pozitif deger araligi ... kademe' satiri: Sekil 13 iLineFit_speed")
    print("  icin ALTI kademe gosteriyor.  Bizimki 1-2 kademedeyse hesap farkli.")
    print("* 'tam sifir' sayisi: basarisiz fitler sifir olarak geliyorsa")
    print("  degiskenin ayirt etme gucu boyle kaybolur.")
    print("* 'notun araligi disinda' satiri: sistematik bir kayma varsa gorunur.")
    print("* Iki egri her yerde ust uste biniyorsa o degisken ayirmiyor demektir.")


if __name__ == "__main__":
    main()
