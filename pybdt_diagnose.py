#!/usr/bin/env python
'''
Verim acigi nereden geliyor?  Uc hipotezi ayirt eden olcumler.

Baglam: ilk noise BDT'si %99 arkaplan reddinde %53 sinyal verimi veriyor;
teknik not Tablo 13'te hedef %99.2 redde ~%96.  Aynı VERIMDE bizim
reddimiz %67, onlarinki %99.2 -- yani sorun kesimin yerini kacirmak degil,
AYIRT ETME GUCU.

Bu script iki hipotezi olcer:

  A) VERI  -- arkaplan istatistigi yetersiz mi?
     Ogrenme egrisi: arkaplanin %25/%50/%75/%100'uyle egit, HEP AYNI tam
     test setinde olc.  Egri %100'de hala tirmaniyorsa veri eklemek
     kazandirir ve ne kadar kazandiracagi kabaca okunur.  Duzlestiyse
     kisit veri degildir.

  C) DEGISKENLER -- birini yanlis mi hesapliyoruz?
     Her degiskenin TEK BASINA ayirt etme gucu (iki yonde de denenir).
     Bir degisken hic ayirmiyorsa ya da BEKLENENIN TERSI yonde ayiriyorsa
     burada gorunur.  Ayrica her degiskenin sinyal/arkaplan medyanlari
     basilir -- Sekil 12-13 ile goz karsilastirmasi icin.

  (B) MOTOR hipotezi -- AdaBoost yerine LightGBM -- bu scriptte YOK;
     ayri bir zincir, reference/train_L4_classifier.py uzerinden.

Kullanim:

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
            sys.exit("bulunamadi: %s" % p)
        out[part] = util.load(p)
    return out


def cols(ds, names):
    return {k: np.asarray(ds[k]) for k in names}


def wcol(ds):
    names = list(ds.names)
    key = "w_phys" if "w_phys" in names else "weight"
    return np.nan_to_num(np.asarray(ds[key]))


def single_variable_power(ds, features, target):
    '''
    Her degiskenin tek basina gucu.

    Karar agaci "deger >= kesim" seklinde ayirir, ama hangi TARAFIN sinyal
    oldugu degiskene gore degisir.  Iki yonu de deneyip iyisini aliyoruz;
    hangi yonun kazandigi da basiliyor -- beklenenin tersi cikarsa
    degiskeni yanlis hesapliyoruz demektir.
    '''
    ws = wcol(ds["sig_test"])
    wb = wcol(ds["bg_test"])
    print("\n" + "=" * 78)
    print("C) DEGISKENLERIN TEK BASINA GUCU  (%%%.0f arkaplan reddinde verim)"
          % (100 * target))
    print("=" * 78)
    print("%-22s %8s %8s | %10s %10s | %s"
          % ("degisken", "verim", "yon", "sig medyan", "bg medyan", "NaN"))
    print("-" * 78)
    rows = []
    for f in features:
        s = np.asarray(ds["sig_test"][f], dtype=float)
        b = np.asarray(ds["bg_test"][f], dtype=float)
        nan = int((~np.isfinite(s)).sum() + (~np.isfinite(b)).sum())
        best, best_dir = -1.0, "?"
        for sign, label in ((+1.0, "buyuk=sinyal"), (-1.0, "kucuk=sinyal")):
            _, eff, _ = eff_at_rejection(sign * s, ws, sign * b, wb, target)
            if np.isfinite(eff) and eff > best:
                best, best_dir = eff, label
        rows.append((f, best, best_dir))
        print("%-22s %7.1f%% %8s | %10.4g %10.4g | %d"
              % (f, 100 * best, best_dir,
                 np.nanmedian(s), np.nanmedian(b), nan))
    print("\n  Yorum: %5.1f%% civari = o degisken TEK BASINA hicbir sey"
          % (100 * (1 - target)))
    print("         ayirmiyor demektir (rastgele kesimin verecegi deger).")
    print("         'yon' sutunu beklentine uymuyorsa degiskeni yanlis")
    print("         hesapliyor olabiliriz -- Sekil 12-13 ile karsilastir.")
    return rows


def learning_curve(ds, features, args, fracs=(0.25, 0.5, 0.75, 1.0)):
    '''
    Arkaplani seyrelt, sinyali ve TEST setini sabit tut.

    Test seti degismiyor -- yoksa noktalar karsilastirilamaz.
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
    print("A) OGRENME EGRISI  (arkaplan seyreltiliyor, test seti sabit)")
    print("=" * 78)
    print("  model: derinlik %d, %d agac, beta %g;  her nokta %d tekrar"
          % (args.depth, args.num_trees, args.beta, args.repeat))
    print("\n%-12s %-10s | %-22s" % ("arkaplan", "olay", "VERIM (ort +- yariyayilim)"))
    print("-" * 52)

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
              % ("%%%d" % (100 * frac), k, 100 * m, 100 * half))

    # egrinin sonundaki egim: son iki nokta
    if len(out) >= 2:
        (f0, k0, e0, _), (f1, k1, e1, _) = out[-2], out[-1]
        d = 100 * (e1 - e0)
        print("\n  Son adimda (%d -> %d olay) kazanc: %+.1f puan" % (k0, k1, d))
        if d > 2:
            print("  -> Egri HALA TIRMANIYOR: arkaplan istatistigi bagliyici.")
            print("     Kabaca ayni egimle devam ederse arkaplani iki katina")
            print("     cikarmak %+.0f puan daha getirir." % d)
        elif d > 0.5:
            print("  -> Egri yavasliyor: veri eklemek az miktarda kazandirir.")
        else:
            print("  -> Egri DUZLESMIS: daha cok arkaplan verisi bu modelle")
            print("     kazandirmaz.  Kisit baska yerde (degiskenler ya da motor).")
    return out


def signal_balance(ds, features, args):
    '''Sinyali arkaplan mertebesine indirip dengesizligin etkisini olc.'''
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
    print("EK) SINIF DENGESIZLIGI  (sinyal seyreltiliyor, test seti sabit)")
    print("=" * 78)
    print("%-16s %-10s | %-22s" % ("sinyal/arkaplan", "sinyal olayi", "VERIM"))
    print("-" * 52)
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
        print("%-16s %-10d | %6.1f%%  +-%.1f"
              % ("tam (%d:1)" % (n_sig // n_bg) if ratio is None else "%d:1" % ratio,
                 k, 100 * np.mean(effs), 100 * (max(effs) - min(effs)) / 2))
    print("\n  Denge duzelince verim BELIRGIN artiyorsa 159:1 dengesizligi")
    print("  AdaBoost'u zorluyor demektir; degismiyorsa dengesizlik sorun degil.")


def main():
    ap = argparse.ArgumentParser(description="noise BDT verim acigi teshisi")
    ap.add_argument("--ds-dir", required=True)
    ap.add_argument("--tag", default="L4_noise")
    ap.add_argument("--features", default=None)
    ap.add_argument("--weight-col", default="weight")
    ap.add_argument("--target-rejection", type=float, default=0.99)
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
    print("=== %s teshis ===" % args.tag)
    print("  train  sinyal %8d   arkaplan %6d   (oran %d:1)"
          % (n_sig, n_bg, n_sig // max(n_bg, 1)))
    print("  test   sinyal %8d   arkaplan %6d"
          % (len(ds["sig_test"][features[0]]), len(ds["bg_test"][features[0]])))
    print("  hedef  %%%.1f arkaplan reddinde sinyal verimi"
          % (100 * args.target_rejection))

    single_variable_power(ds, features, args.target_rejection)
    learning_curve(ds, features, args)
    if not args.skip_balance:
        signal_balance(ds, features, args)

    print("\n" + "=" * 78)
    print("Motor hipotezi (AdaBoost vs LightGBM) bu scriptte YOK --")
    print("ayni 5 degiskenle reference/train_L4_classifier.py ile karsilastirin.")
    print("=" * 78)


if __name__ == "__main__":
    main()
