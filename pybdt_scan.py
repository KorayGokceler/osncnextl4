#!/usr/bin/env python
'''
Hiperparametre taramasi -- pybdt/AdaBoost, oscNext L4 siniflandiricilari.

NEDEN AYRI BIR SCRIPT:
  pybdt_train.py tek bir modeli egitir, dogrular, grafik cizer ve kaydeder.
  Tarama farkli bir is: onlarca modeli egitip aralarindan secmek.  Ogrenici
  kurulumu TEKRARLANMIYOR -- build_learner/score_expr dogrudan
  pybdt_train'den import ediliyor, tek implementasyon kaliyor.

NASIL HIZLI:
  .ds dosyalari BIR KEZ okunuyor.  Egitim tam veri uzerinde yapiliyor ama
  metrikler icin sinyal ORNEKLENIYOR (--max-sig, varsayilan 20 000):
  165 000 olayi her konfigurasyon icin skorlamak taramanin suresini
  belirleyen adimdi ve 20 000 olay verimi ~%0.3 hassasiyetle olcmeye zaten
  yetiyor.

OLCUT:
  Tek sayi: HEDEF ARKAPLAN REDDINDE SINYAL VERIMI (--target-rejection,
  varsayilan 0.99 -- teknik not Tablo 13'te gurultu icin %99.24).
  Overtraining elemesi: train/test skor dagilimlarinin iki-ornekli KS testi
  (scipy).  p_KS < --ks-min olan konfigurasyonlar ELENIR.

  NOT: buradaki KS scipy'nin agirliksiz testi; pybdt'nin kendi
  get_kolmogorov_smirnov_probability'si ile birebir ayni sayiyi vermez.
  Tarama bir ON ELEME -- secilen konfigurasyon yine pybdt_train.py ile
  egitilmeli, asil KS oradan okunmali.

DIKKAT -- SECIM YANLILIGI:
  Konfigurasyon test seti uzerinden seciliyor, o yuzden taramanin bastigi
  verim iyimser bir tahmindir.  Ucuncu bir ayrim yapacak kadar arkaplan
  olayimiz yok (bkz. CLAUDE.md 5e).  Secilen degeri "ust sinir" olarak
  okuyun; asil sayi daha cok arkaplan islendikten sonra netlesir.

Kullanim:

    python pybdt_scan.py --ds-dir L4_output/ds --tag L4_noise \
        --features NchCleaned,micro_count,iLineFit_speed,fill_ratio,FullTimeLengthRatio

    # kendi izgaran
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
from pybdt_train import build_learner, effective_params, score_expr, RESERVED_COLS

try:
    from scipy.stats import ks_2samp
except ImportError:
    ks_2samp = None


def parse_list(text, cast):
    '''"2,3,4" -> [2,3,4];  "none,10" -> [None, 10.0]'''
    out = []
    for piece in str(text).split(","):
        piece = piece.strip()
        out.append(None if piece.lower() in ("none", "-", "") else cast(piece))
    return out


def subsample(ds, n, rng):
    '''DataSet'ten en fazla n olay sec (metrik hesabi icin).'''
    names = list(ds.names)
    total = len(ds[names[0]])
    if n <= 0 or total <= n:
        return ds
    idx = rng.choice(total, size=n, replace=False)
    return ml.DataSet({k: np.asarray(ds[k])[idx] for k in names})


def eff_at_rejection(s, ws, b, wb, target):
    '''
    Arkaplanin `target` kadarini reddeden kesimi bul, oradaki sinyal verimini
    dondur.  Kesim arkaplan AGIRLIGININ kuantilinden geliyor -- olay sayisi
    degil, cunku oran karsilastirmasi agirlikli yapiliyor.
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
    ap = argparse.ArgumentParser(description="pybdt hiperparametre taramasi")
    ap.add_argument("--ds-dir", required=True, help=".ds dosyalarinin dizini")
    ap.add_argument("--tag", default="L4_noise",
                    help="<tag>_{sig,bg}_{train,test}.ds on eki")
    ap.add_argument("--features", default=None,
                    help="virgulle ayrilmis; verilmezse RESERVED_COLS disi hepsi")
    ap.add_argument("--weight-col", default="weight")

    ap.add_argument("--target-rejection", type=float, default=0.99)
    ap.add_argument("--ks-min", type=float, default=0.01,
                    help="p_KS bunun altindaysa konfigurasyon elenir")
    ap.add_argument("--max-sig", type=int, default=20000,
                    help="metrik hesabinda kullanilacak sinyal olayi (0=hepsi)")
    ap.add_argument("--seed", type=int, default=12345)

    # izgara
    ap.add_argument("--depth", default="2,3,4")
    ap.add_argument("--num-trees", default="200,500")
    ap.add_argument("--min-split", default="20,100")
    ap.add_argument("--prune-strength", default="none,10")
    ap.add_argument("--beta", default="0.5")
    ap.add_argument("--frac-random-events", default="0.5")
    ap.add_argument("--num-cuts", default="none")
    ap.add_argument("--no-purity", action="store_true",
                    help="use_purity'yi KAPAT (varsayilan acik)")
    args = ap.parse_args()

    if ks_2samp is None:
        sys.exit("scipy yok -- overtraining elemesi yapilamaz.")

    rng = np.random.default_rng(args.seed)

    # --- veri: BIR KEZ oku --------------------------------------------------
    ds = {}
    for part in ("sig_train", "bg_train", "sig_test", "bg_test"):
        p = os.path.join(args.ds_dir, "%s_%s.ds" % (args.tag, part))
        if not os.path.exists(p):
            sys.exit("bulunamadi: %s" % p)
        ds[part] = util.load(p)

    if args.features:
        features = [f.strip() for f in args.features.split(",") if f.strip()]
    else:
        features = [n for n in ds["sig_train"].names
                    if n not in RESERVED_COLS and n != args.weight_col]

    n_sig = len(ds["sig_train"][features[0]])
    n_bg = len(ds["bg_train"][features[0]])
    print("=== %s hiperparametre taramasi ===" % args.tag)
    print("  train  sinyal %8d   arkaplan %6d" % (n_sig, n_bg))
    print("  test   sinyal %8d   arkaplan %6d"
          % (len(ds["sig_test"][features[0]]), len(ds["bg_test"][features[0]])))
    print("  degisken %d: %s" % (len(features), ", ".join(features)))
    print("  olcut: %%%.1f arkaplan reddinde sinyal verimi;  p_KS >= %.3f"
          % (100 * args.target_rejection, args.ks_min))

    # metrik icin ornekleme (arkaplan zaten kucuk, dokunmuyoruz)
    m = {"sig_train": subsample(ds["sig_train"], args.max_sig, rng),
         "sig_test":  subsample(ds["sig_test"],  args.max_sig, rng),
         "bg_train":  ds["bg_train"],
         "bg_test":   ds["bg_test"]}
    if args.max_sig and n_sig > args.max_sig:
        print("  (metrikler %d sinyal olayi uzerinden -- egitim tam veriyle)"
              % args.max_sig)

    w = {k: np.asarray(m[k]["w_phys"]) if "w_phys" in m[k].names
         else np.asarray(m[k][args.weight_col]) for k in m}
    for k in w:
        w[k] = np.nan_to_num(w[k])

    # --- izgara -------------------------------------------------------------
    grid = list(itertools.product(
        parse_list(args.depth, int),
        parse_list(args.num_trees, int),
        parse_list(args.min_split, int),
        parse_list(args.prune_strength, float),
        parse_list(args.beta, float),
        parse_list(args.frac_random_events, float),
        parse_list(args.num_cuts, int)))
    print("  %d konfigurasyon\n" % len(grid))

    hdr = ("%-6s %-7s %-6s %-7s %-5s | %-8s %-8s | %-9s %-9s"
           % ("depth", "trees", "split", "prune", "beta",
              "p_KS sig", "p_KS bg", "kesim", "VERIM"))
    print(hdr)
    print("-" * len(hdr))

    expr = score_expr(not args.no_purity)
    rows, t0 = [], time.time()

    for depth, trees, split, prune, beta, frac, ncuts in grid:
        a = SimpleNamespace(
            depth=depth, min_split=split, num_cuts=ncuts,
            num_random_variables=None, nonlinear_cuts=False,
            num_trees=trees, beta=beta, frac_random_events=frac,
            prune_strength=prune, use_purity=not args.no_purity)
        learner = build_learner(features, args.weight_col, a)
        bdt = learner.train(ds["sig_train"], ds["bg_train"])

        sc = {k: np.asarray(bdt.score(m[k], use_purity=not args.no_purity,
                                      quiet=True)) for k in m}

        p_sig = float(ks_2samp(sc["sig_train"], sc["sig_test"]).pvalue)
        p_bg = float(ks_2samp(sc["bg_train"], sc["bg_test"]).pvalue)
        cut, eff, rej = eff_at_rejection(sc["sig_test"], w["sig_test"],
                                         sc["bg_test"], w["bg_test"],
                                         args.target_rejection)

        ok = min(p_sig, p_bg) >= args.ks_min
        rows.append(dict(depth=depth, trees=trees, split=split, prune=prune,
                         beta=beta, frac=frac, ncuts=ncuts,
                         p_sig=p_sig, p_bg=p_bg, cut=cut, eff=eff, rej=rej,
                         ok=ok, n_trees=len(bdt)))
        print("%-6s %-7s %-6s %-7s %-5s | %-8.4f %-8.4f | %-9.3f %6.1f%% %s"
              % (depth, trees, split,
                 "-" if prune is None else ("%g" % prune), beta,
                 p_sig, p_bg, cut, 100 * eff,
                 "" if ok else "  <-- OVERTRAIN, elendi"))

    print("\n%d konfigurasyon, %.0f s" % (len(grid), time.time() - t0))

    good = [r for r in rows if r["ok"] and np.isfinite(r["eff"])]
    if not good:
        print("\nHicbir konfigurasyon p_KS >= %.3f esigini gecmedi." % args.ks_min)
        print("Kapasiteyi dusurun (--depth 1,2  --num-trees 50,100) ya da")
        print("daha cok ARKAPLAN olayi isleyin -- asil kisit orasi.")
        return

    good.sort(key=lambda r: -r["eff"])
    b = good[0]
    print("\n=== En iyi (overtraining elemesini gecenler arasinda) ===")
    print("  verim %.1f%%  (red %.2f%%, kesim %.3f)"
          % (100 * b["eff"], 100 * b["rej"], b["cut"]))
    print("  p_KS  sinyal %.4f  arkaplan %.4f" % (b["p_sig"], b["p_bg"]))
    print("\nNotebook'a yapistir:\n")
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
    print("\nSecim TEST seti uzerinden yapildi -> bu verim iyimser bir tahmin.")
    print("Secilen konfigurasyonu pybdt_train.py ile yeniden egitin.")


if __name__ == "__main__":
    main()
