#!/usr/bin/env python
'''
AdaBoost (pybdt) vs LightGBM -- ADIL kafa kafaya karsilastirma.

NEDEN:
  Teknik not (v00.074 §3.6.1) L4 siniflandiricilarini LightGBM ile
  egitiyor; Tablo 10'daki hiperparametreler LightGBM'in native isimleri.
  Bu projede pybdt/AdaBoost kullanmak BILINCLI bir sapmaydi -- ama
  bedeli hic olculmedi.  Ilk noise BDT'si %99 arkaplan reddinde %53
  sinyal verimi verdi; hedef %96.  Bu script farkin motordan gelip
  gelmedigini kesin olarak soyler.

NEYIN AYNI OLDUGU (karsilastirmanin adil olmasi bunlara bagli):
  * ayni olaylar -- ayni .ds dosyalari, ayni train/test ayrimi
  * ayni 5 degisken
  * ayni egitim agirligi (weight kolonu; siniflar esitlenmis)
  * ayni degerlendirme: w_phys ile agirlikli, hedef arkaplan reddinde
    sinyal verimi.  eff_at_rejection pybdt_scan'den IMPORT ediliyor,
    yeniden yazilmiyor.
  * ayni overtraining olcutu: pybdt'nin kendi AGIRLIKLI KS'i

  pybdt skoru [-1,1], LightGBM ciktisi [0,1] -- ama olcut monoton
  donusume DUYARSIZ (kesim arkaplan kuantilinden geliyor), o yuzden
  farkli olcekler karsilastirmayi bozmuyor.

NEYIN ZORUNLU OLARAK FARKLI OLDUGU:
  * motor: AdaBoost vs gradient boosting
  * hiperparametreler: her motorun kendi API'si.  LightGBM tarafi
    reference/train_L4_classifier.py'deki PARAMS'tan AST ile OKUNUYOR
    (import degil -- o dosya pandas/sklearn import ediyor, IceTray
    ortaminda import edilemez).  Degerler dogrudan Tablo 10.
  * NaN: LightGBM eksik degeri kendisi yonlendirir; pybdt yolunda ise
    notebook NaN'li olaylari DUSURUYOR (make_datasets.drop_nan).  Ayni
    .ds dosyalarini okudugumuz icin burada da NaN'li olay yok -- yani
    fark olusmuyor, ama bilinsin.
  * early stopping: LightGBM'de var, pybdt'de yok.  Referans script
    early stopping icin TEST setini kullaniyor; burada bunu YAPMIYORUZ
    (test sizmasi olurdu).  Egitim setinden ayrilan bir dogrulama
    dilimi kullaniliyor (--valid-frac), test seti hic dokunulmadan
    kaliyor.

Kullanim:

    python lightgbm_compare.py --ds-dir L4_output/ds --tag L4_noise \
        --features NchCleaned,micro_count,iLineFit_speed,fill_ratio,FullTimeLengthRatio
'''

import os
import sys
import ast
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from pybdt import util
from pybdt.validate import kolmogorov_smirnov_probability
from pybdt_train import RESERVED_COLS
from pybdt_scan import eff_at_rejection

try:
    import lightgbm as lgb
except ImportError:
    sys.exit("lightgbm yok -- bu karsilastirma yapilamaz.\n"
             "IceTray ortaminda: python -c 'import lightgbm'")

REFERENCE_TRAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "reference", "train_L4_classifier.py")


def read_reference_params(path=REFERENCE_TRAIN):
    """
    Tablo 10 degerlerini reference/train_L4_classifier.py'den AST ile oku.

    NEDEN IMPORT DEGIL: o dosya modul seviyesinde `import pandas` ve
    `from sklearn.metrics import ...` yapiyor -- kendi docstring'i
    "IceTray ortaminda sklearn ve joblib YOK" dedigi halde.  Yani IceTray
    ortaminda import EDILEMEZ.  Degerleri kaynaktan okumak, onlari buraya
    kopyalayip iki yerde tutmaktan iyi (l4_data.check_feature_map de
    FEATURE_MAP'i ayni sebeple AST ile okuyor).

    Dosya yoksa/degistiyse net hata verilir -- sessizce baska bir degere
    dusmek Tablo 10 ile karsilastirmayi anlamsiz kilardi.
    """
    if not os.path.exists(path):
        sys.exit("referans bulunamadi: %s" % path)
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
        sys.exit("%s icinde PARAMS bulunamadi (dosya degismis olabilir)" % path)
    return params, (early if early is not None else 100)


PARAMS, EARLY_STOPPING = read_reference_params()


def load_side(ds_dir, tag, part, features):
    p = os.path.join(ds_dir, "%s_%s.ds" % (tag, part))
    if not os.path.exists(p):
        sys.exit("bulunamadi: %s" % p)
    ds = util.load(p)
    names = list(ds.names)
    X = np.column_stack([np.asarray(ds[f], dtype=float) for f in features])
    w = np.nan_to_num(np.asarray(ds["weight"], dtype=float))
    wp = np.nan_to_num(np.asarray(ds["w_phys"], dtype=float)) \
        if "w_phys" in names else w.copy()
    return X, w, wp


def report_curve(s, ws, b, wb, cuts=None):
    '''pybdt tarafiyla ayni bicimde verim/red tablosu.'''
    lo, hi = min(s.min(), b.min()), max(s.max(), b.max())
    cuts = cuts if cuts is not None else np.linspace(lo, hi, 9)[1:-1]
    print("\n%-9s | %8s %8s | %6s" % ("kesim", "verim", "red", "bg olay"))
    print("-" * 40)
    for c in cuts:
        print("%-9.4f | %7.1f%% %7.2f%% | %6d"
              % (c, 100 * ws[s >= c].sum() / ws.sum(),
                 100 * (1 - wb[b >= c].sum() / wb.sum()), int((b >= c).sum())))


def main():
    ap = argparse.ArgumentParser(
        description="AdaBoost (pybdt) vs LightGBM, ayni veriyle")
    ap.add_argument("--ds-dir", required=True)
    ap.add_argument("--tag", default="L4_noise")
    ap.add_argument("--features", required=True,
                    help="virgulle ayrilmis -- pybdt kosusuyla AYNI olmali")
    ap.add_argument("--params", default=None,
                    help="PARAMS anahtari: noise|muon.  Varsayilan tag'den turer.")
    ap.add_argument("--target-rejection", type=float, default=0.99)
    ap.add_argument("--valid-frac", type=float, default=0.2,
                    help="early stopping icin EGITIM setinden ayrilan dilim. "
                         "Test seti hic kullanilmaz.")
    ap.add_argument("--no-early-stopping", action="store_true")
    ap.add_argument("--num-boost-round", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--outdir", default=None,
                    help="verilirse model .txt olarak yazilir")
    args = ap.parse_args()

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    bad = [f for f in features if f in RESERVED_COLS]
    if bad:
        sys.exit("BDT girdisi olmayan kolon verilmis: %s" % ", ".join(bad))

    key = args.params or ("muon" if "muon" in args.tag.lower() else "noise")
    p = dict(PARAMS[key])
    n_rounds = args.num_boost_round or p.pop("num_boost_round")
    p.pop("num_boost_round", None)
    if args.learning_rate:
        p["learning_rate"] = args.learning_rate
    p["seed"] = args.seed

    # --- veri ---------------------------------------------------------------
    Xs, ws_tr, _ = load_side(args.ds_dir, args.tag, "sig_train", features)
    Xb, wb_tr, _ = load_side(args.ds_dir, args.tag, "bg_train", features)
    Xst, ws_te, wps_te = load_side(args.ds_dir, args.tag, "sig_test", features)
    Xbt, wb_te, wpb_te = load_side(args.ds_dir, args.tag, "bg_test", features)

    print("=== %s : LightGBM (Tablo 10) ===" % args.tag)
    print("  train  sinyal %8d   arkaplan %6d" % (len(Xs), len(Xb)))
    print("  test   sinyal %8d   arkaplan %6d" % (len(Xst), len(Xbt)))
    print("  degisken %d: %s" % (len(features), ", ".join(features)))
    print("  parametreler (%s): %s" % (key, ", ".join(
        "%s=%s" % (k, v) for k, v in sorted(p.items())
        if k not in ("objective", "metric", "verbosity", "deterministic"))))

    X = np.vstack([Xs, Xb])
    y = np.r_[np.ones(len(Xs)), np.zeros(len(Xb))]
    w = np.r_[ws_tr, wb_tr]

    rng = np.random.default_rng(args.seed)
    if args.no_early_stopping or args.valid_frac <= 0:
        tr = np.ones(len(y), dtype=bool)
        va = None
    else:
        # Dogrulama dilimi EGITIM setinden -- test setine dokunmuyoruz.
        # Sinif orani korunsun diye her sinifta ayri ayri ayiriyoruz:
        # arkaplan zaten az, rastgele ayirim onu daha da inceltebilirdi.
        tr = np.ones(len(y), dtype=bool)
        for cls in (0.0, 1.0):
            idx = np.flatnonzero(y == cls)
            k = max(1, int(round(args.valid_frac * len(idx))))
            tr[rng.choice(idx, size=k, replace=False)] = False
        va = ~tr
        print("  early stopping: egitim setinden %%%.0f dogrulama dilimi "
              "(sinyal %d, arkaplan %d)"
              % (100 * args.valid_frac, int((va & (y == 1)).sum()),
                 int((va & (y == 0)).sum())))

    dtrain = lgb.Dataset(X[tr], label=y[tr], weight=w[tr],
                         feature_name=features, free_raw_data=False)
    callbacks = [lgb.log_evaluation(0)]
    valid_sets = None
    if va is not None:
        valid_sets = [lgb.Dataset(X[va], label=y[va], weight=w[va],
                                  feature_name=features, reference=dtrain)]
        callbacks.append(lgb.early_stopping(EARLY_STOPPING, verbose=False))

    print("\nEgitim ...")
    booster = lgb.train(p, dtrain, num_boost_round=n_rounds,
                        valid_sets=valid_sets, callbacks=callbacks)
    print("  bitti -- %d agac (sinir %d)" % (booster.num_trees(), n_rounds))

    # --- degerlendirme: pybdt tarafiyla AYNI olcut --------------------------
    s_tr = booster.predict(Xs)
    b_tr = booster.predict(Xb)
    s_te = booster.predict(Xst)
    b_te = booster.predict(Xbt)

    p_sig = float(kolmogorov_smirnov_probability(s_tr, ws_tr, s_te, ws_te))
    p_bg = float(kolmogorov_smirnov_probability(b_tr, wb_tr, b_te, wb_te))
    print("\n--- Overtraining (pybdt agirlikli KS, pybdt_train ile ayni) ---")
    print("  p_KS sinyal   = %.4f%s" % (p_sig, "  <-- OVERTRAIN" if p_sig < 0.01 else ""))
    print("  p_KS arkaplan = %.4f%s" % (p_bg, "  <-- OVERTRAIN" if p_bg < 0.01 else ""))

    cut, eff, rej = eff_at_rejection(s_te, wps_te, b_te, wpb_te,
                                     args.target_rejection)
    print("\n--- ASIL SAYI ---")
    print("  %%%.1f arkaplan reddinde sinyal verimi: %.1f%%  (kesim %.4f, red %.2f%%)"
          % (100 * args.target_rejection, 100 * eff, cut, 100 * rej))

    report_curve(s_te, wps_te, b_te, wpb_te)

    print("\n--- Degisken onemi (gain) ---")
    imp = booster.feature_importance(importance_type="gain")
    tot = imp.sum() or 1.0
    for f, v in sorted(zip(features, imp), key=lambda x: -x[1]):
        print("  %-22s %8.1f  (%%%.1f)" % (f, v, 100 * v / tot))

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        path = os.path.join(args.outdir, "%s_lightgbm.txt" % args.tag)
        booster.save_model(path)
        print("\n  -> %s" % path)

    print("\n" + "=" * 70)
    print("Bu sayiyi pybdt kosusunun ayni satiriyla karsilastirin.")
    print("Yakinsa fark motordan DEGIL (veri ya da degiskenler);")
    print("LightGBM belirgin onde ise AdaBoost bu problemde yetmiyor.")
    print("=" * 70)


if __name__ == "__main__":
    main()
