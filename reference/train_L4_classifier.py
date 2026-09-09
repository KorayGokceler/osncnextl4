#!/usr/bin/env python
'''
oscNext L4 siniflandirici egitimi.

Notebook'un urettigi parquet dosyalarini okur, LightGBM modelini egitir,
degerlendirir ve I3Classifier'in yukleyebilecegi .joblib dosyasini yazar.

  L4_noise_training.parquet  --+
                               +--> train_L4_classifier.py --> L4_*_model.joblib
  L4_muon_training.parquet   --+                            \\-> rapor + grafikler

Kullanim:

  # Noise siniflandirici
  python train_L4_classifier.py --tag noise \\
      --input ./l4_training_data/L4_noise_training.parquet \\
      --outdir ./models

  # Muon siniflandirici
  python train_L4_classifier.py --tag muon \\
      --input ./l4_training_data/L4_muon_training.parquet \\
      --outdir ./models

  # Sadece degerlendirme (yeniden egitmeden)
  python train_L4_classifier.py --tag noise --input ... --outdir ./models --eval-only

MODEL FORMATI: LightGBM'in kendi metin formati (.txt) + JSON meta dosyasi.

Neden joblib/pickle degil:
  * IceTray ortaminda (py3-v4.4.2) sklearn ve joblib YOK, sadece lightgbm var.
    sklearn API'li bir model orada yuklenemez -- LGBMClassifier sinifi bile
    insa edilemez.
  * Native metin formati surumler arasi stabil; pickle degil.  Egitim ortami
    ile IceTray ortaminin lightgbm surumleri farkli olabilir.
  * Metin formati okunabilir -- agaclari gozle inceleyebilirsiniz.

Uygulama tarafi: l4_classifier_module.py  (lgb.Booster + numpy, baska yok)

--format joblib ile ek olarak pickle de yazilabilir (joblib kuruluysa).
'''

import os
import sys
import json
import argparse
import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.metrics import roc_curve, roc_auc_score


# ===========================================================================
# Hyperparametreler -- technical note Tablo 10
# ===========================================================================
#
# Native API kullaniliyor -> Tablo 10'daki isimler dogrudan gecerli.
# (sklearn API'de bunlar min_child_samples / colsample_bytree / reg_alpha /
#  reg_lambda / min_split_gain olarak yeniden adlandirilir; onu KULLANMIYORUZ.)

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
        is_unbalance=False,
        learning_rate=0.05,
        num_boost_round=2000,       # early stopping kesecek
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

# v00.07 referans kesim degerleri
DEFAULT_CUT = {"noise": 0.70, "muon": 0.65}

# I3Classifier'in class_key parametresi bu ismi arar
POSITIVE_CLASS = "neutrino"
NEGATIVE_CLASS = {"noise": "noise", "muon": "muon"}


# ===========================================================================
# Veri
# ===========================================================================

def load_data(path, features=None):
    df = pd.read_parquet(path)
    meta_cols = {"label", "w_train", "w_phys", "split", "sample",
                 "Run", "Event", "SubEvent"}
    if features is None:
        features = [c for c in df.columns if c not in meta_cols]

    missing = [f for f in features if f not in df.columns]
    if missing:
        sys.exit(f"Parquet'te eksik degisken: {missing}")

    X = df[features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    y = df["label"].to_numpy().astype(int)
    w = df["w_train"].to_numpy(float)
    wp = df["w_phys"].to_numpy(float) if "w_phys" in df else np.full(len(df), np.nan)
    split = (df["split"].to_numpy() if "split" in df
             else np.array(["train"] * len(df), dtype=object))

    print(f"  {len(df):,} olay | {len(features)} degisken")
    print(f"  sinyal={int(y.sum()):,}  arkaplan={int((1-y).sum()):,}")
    print(f"  train={int((split=='train').sum()):,}  test={int((split=='test').sum()):,}")
    return df, X, y, w, wp, split, features


# ===========================================================================
# Egitim
# ===========================================================================

def train(X, y, w, split, tag, params_override=None):
    '''
    Native LightGBM API ile egit (lgb.train -> Booster).

    sklearn API (LGBMClassifier) KULLANILMIYOR: IceTray ortaminda sklearn yok,
    dolayisiyla oradaki uygulama tarafi sklearn nesnesi yukleyemez.
    Booster ise sadece lightgbm'e bagimli.

    Binary objective ile booster.predict(X) dogrudan P(sinif=1) dondurur --
    predict_proba'ya ihtiyac yok.
    '''
    p = dict(PARAMS[tag])
    if params_override:
        p.update(params_override)

    tr, te = split == "train", split == "test"
    if te.sum() == 0:
        sys.exit("Test seti bos -- parquet'teki 'split' kolonunu kontrol edin.")

    n_rounds = p.pop("num_boost_round")
    dtrain = lgb.Dataset(X[tr], label=y[tr], weight=w[tr],
                         feature_name=list(X.columns), free_raw_data=False)
    dtest = lgb.Dataset(X[te], label=y[te], weight=w[te],
                        reference=dtrain, free_raw_data=False)

    print(f"\nEgitim ({tag}) ...")
    booster = lgb.train(
        p, dtrain, num_boost_round=n_rounds,
        valid_sets=[dtrain, dtest], valid_names=["train", "test"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(100)])
    print(f"  en iyi iterasyon: {booster.best_iteration}")
    return booster


def predict(booster, X):
    '''P(neutrino).  Booster.predict binary objective'de dogrudan olasilik verir.'''
    return booster.predict(X, num_iteration=booster.best_iteration)


# ===========================================================================
# Degerlendirme
# ===========================================================================

def evaluate(model, X, y, w, wp, split, tag, outdir, cut=None):
    cut = cut if cut is not None else DEFAULT_CUT[tag]
    tr, te = split == "train", split == "test"
    prob = predict(model, X)

    auc_tr = roc_auc_score(y[tr], prob[tr], sample_weight=w[tr])
    auc_te = roc_auc_score(y[te], prob[te], sample_weight=w[te])

    print(f"\n--- Performans ({tag}) ---")
    print(f"  AUC train = {auc_tr:.4f}")
    print(f"  AUC test  = {auc_te:.4f}")
    print(f"  fark      = {auc_tr - auc_te:+.4f}", end="  ")
    if auc_tr - auc_te > 0.01:
        print("<-- OVERTRAINING")
        if tag == "noise":
            print("      (v00.07'de de noise siniflandiricisinda bir miktar")
            print("       overtraining gozlendi; secim sonunda gurultu ihmal")
            print("       edilebilir seviyeye indigi icin etkisi kucuk sayildi)")
    else:
        print("(iyi)")

    # --- Sekil 15/22: train/test tahmin dagilimi ---
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    bins = np.linspace(0, 1, 41)
    for lab, cname, color in [(1, POSITIVE_CLASS, "tab:blue"),
                              (0, NEGATIVE_CLASS[tag], "tab:red")]:
        for sset, style in [(tr, dict(histtype="stepfilled", alpha=0.3)),
                            (te, dict(histtype="step", lw=1.6))]:
            m = sset & (y == lab)
            if m.sum() == 0:
                continue
            name = f"{cname} ({'train' if sset is tr else 'test'})"
            ax.hist(prob[m], bins=bins, weights=w[m], density=True,
                    color=color, label=name, **style)
    ax.axvline(cut, color="k", ls="--", lw=1, label=f"kesim = {cut}")
    ax.set_yscale("log"); ax.set_xlabel(f"P({POSITIVE_CLASS})")
    ax.set_ylabel("normalize yogunluk"); ax.legend(fontsize=7)
    ax.set_title(f"L4 {tag} siniflandirici — train/test uyumu", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{outdir}/L4_{tag}_train_test.png", dpi=130)
    plt.close(fig)

    # --- Sekil 16/23: kesim degerine karsi rate ---
    perf = None
    if np.isfinite(wp).any():
        cuts = np.linspace(0, 1, 101)
        sig_rate = [wp[(y == 1) & (prob >= c)].sum() for c in cuts]
        bkg_rate = [wp[(y == 0) & (prob >= c)].sum() for c in cuts]
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.plot(cuts, sig_rate, label=POSITIVE_CLASS, color="tab:blue")
        ax.plot(cuts, bkg_rate, label=NEGATIVE_CLASS[tag], color="tab:red")
        ax.axvline(cut, color="k", ls="--", lw=1)
        ax.set_yscale("log"); ax.set_xlabel("kesim degeri")
        ax.set_ylabel("Rate [1/s]"); ax.legend(fontsize=8)
        ax.set_title(f"L4 {tag} — kesim degerine karsi rate", fontsize=10)
        fig.tight_layout(); fig.savefig(f"{outdir}/L4_{tag}_rate_vs_cut.png", dpi=130)
        plt.close(fig)

        s0, s1 = wp[y == 1].sum(), wp[(y == 1) & (prob >= cut)].sum()
        b0, b1 = wp[y == 0].sum(), wp[(y == 0) & (prob >= cut)].sum()
        perf = dict(cut=float(cut),
                    signal_rate_before=float(s0), signal_rate_after=float(s1),
                    signal_efficiency=float(s1 / s0) if s0 > 0 else np.nan,
                    bkg_rate_before=float(b0), bkg_rate_after=float(b1),
                    bkg_rejection=float(1 - b1 / b0) if b0 > 0 else np.nan)
        print(f"\n--- Kesim {cut} performansi ---")
        print(f"  sinyal: {s0:.4e} -> {s1:.4e} Hz  (verim {100*s1/s0:.1f}%)")
        print(f"  arkaplan: {b0:.4e} -> {b1:.4e} Hz  (red {100*(1-b1/b0):.1f}%)")
        ref = {"noise": "v00.07: 36.6 mHz -> <0.3 mHz, ~%96 nu korundu",
               "muon":  "v00.07: muonlarin %94'u atildi, nu'larin %87'si korundu"}
        print(f"  referans -> {ref[tag]}")

    # --- ROC ---
    fig, ax = plt.subplots(figsize=(4.2, 4))
    for sset, name in [(tr, "train"), (te, "test")]:
        fpr, tpr, _ = roc_curve(y[sset], prob[sset], sample_weight=w[sset])
        a = auc_tr if name == "train" else auc_te
        ax.plot(fpr, tpr, lw=1.4, label=f"{name} (AUC={a:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("yanlis pozitif oran"); ax.set_ylabel("dogru pozitif oran")
    ax.legend(fontsize=8); ax.set_title(f"L4 {tag} ROC", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{outdir}/L4_{tag}_roc.png", dpi=130)
    plt.close(fig)

    # --- Feature importance ---
    imp = pd.DataFrame({
        "feature": list(X.columns),
        "gain": model.feature_importance("gain"),
        "split": model.feature_importance("split"),
    })
    imp["gain_%"] = 100 * imp["gain"] / max(imp["gain"].sum(), 1)
    imp = imp.sort_values("gain_%", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 0.32 * len(imp) + 1.6))
    ax.barh(imp.feature[::-1], imp["gain_%"][::-1], color="tab:blue")
    ax.set_xlabel("gain [%]"); ax.set_title(f"L4 {tag} feature importance", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{outdir}/L4_{tag}_importance.png", dpi=130)
    plt.close(fig)
    print("\n--- Feature importance ---")
    print(imp[["feature", "gain_%", "split"]].to_string(index=False))
    if (imp["gain_%"] < 1.0).any():
        weak = imp[imp["gain_%"] < 1.0].feature.tolist()
        print(f"\n  [!] Katkisi <%1 olan degiskenler: {weak}")
        print("      v00.07'de tum degiskenlerin belirgin katki sagladigi belirtiliyor.")
        print("      Bunlari cikarip yeniden egitmeyi degerlendirin.")

    return dict(auc_train=float(auc_tr), auc_test=float(auc_te),
                overtraining=float(auc_tr - auc_te),
                best_iteration=int(model.best_iteration or 0),
                cut_performance=perf,
                importance=imp.to_dict("records"))


# ===========================================================================
# Kaydetme
# ===========================================================================

def save_model(booster, features, tag, outdir, fmt, metrics, extra_meta=None):
    '''
    Modeli kaydet.

    Her zaman yazilan (birincil):
      L4_{tag}_model.txt   -- LightGBM native metin formati
      L4_{tag}_model.json  -- degisken listesi + sinif haritasi + meta

    Ikisi BIRLIKTE bir modeldir: .txt agaclari, .json ise hangi kolonun hangi
    sirada beslenecegini tutar.  l4_classifier_module.py ikisini de okur.

    fmt="joblib" veya "both" ise ek olarak pickle da yazilir (joblib kuruluysa).
    '''
    txt_path  = os.path.join(outdir, f"L4_{tag}_model.txt")
    json_path = os.path.join(outdir, f"L4_{tag}_model.json")

    booster.save_model(txt_path, num_iteration=booster.best_iteration)

    sidecar = dict(
        tag=tag,
        # SIRA ONEMLI: uygulama tarafi kolonlari bu sirada besler
        features=list(features),
        positive_class=POSITIVE_CLASS,
        negative_class=NEGATIVE_CLASS[tag],
        # booster.predict() P(positive_class) dondurur
        output_is_probability_of=POSITIVE_CLASS,
        default_cut=DEFAULT_CUT[tag],
        best_iteration=int(booster.best_iteration or 0),
        lightgbm_version=lgb.__version__,
        trained=datetime.datetime.now().isoformat(timespec="seconds"),
        params=PARAMS[tag],
        metrics={k: v for k, v in metrics.items() if k != "importance"},
        **(extra_meta or {}),
    )
    with open(json_path, "w") as fh:
        json.dump(sidecar, fh, indent=2, default=str)

    print(f"\nModel kaydedildi:")
    print(f"  {txt_path}   ({os.path.getsize(txt_path)/1024:.0f} KB)")
    print(f"  {json_path}")

    # --- geri yukleme testi: uygulama tarafini taklit et ---
    back = lgb.Booster(model_file=txt_path)
    assert back.num_feature() == len(features), \
        f"Degisken sayisi uyusmuyor: {back.num_feature()} vs {len(features)}"
    if list(back.feature_name()) != list(features):
        print("  [!] Booster'daki degisken isimleri json ile birebir ayni degil")
        print("      booster:", list(back.feature_name())[:5], "...")
        print("      json   :", list(features)[:5], "...")
    print("  geri yukleme testi: OK "
          f"({back.num_trees()} agac, {back.num_feature()} degisken)")

    if fmt in ("joblib", "both"):
        try:
            import joblib
            jp = os.path.join(outdir, f"L4_{tag}_model.joblib")
            joblib.dump({"model": booster, "features": list(features),
                         "classes": {POSITIVE_CLASS: 1, NEGATIVE_CLASS[tag]: 0}}, jp)
            print(f"  {jp}  (ek format)")
        except ImportError:
            print("  [!] joblib kurulu degil, pickle atlandi (native format yeterli)")

    with open(os.path.join(outdir, f"L4_{tag}_report.json"), "w") as fh:
        json.dump(dict(tag=tag, features=list(features), metrics=metrics),
                  fh, indent=2, default=str)
    return txt_path


# ===========================================================================

def main():
    p = argparse.ArgumentParser(description="oscNext L4 siniflandirici egitimi")
    p.add_argument("--tag", required=True, choices=["noise", "muon"])
    p.add_argument("--input", required=True, help="Notebook'un urettigi parquet")
    p.add_argument("--outdir", default="./models")
    p.add_argument("--features", nargs="+", default=None,
                   help="Kullanilacak degiskenler (varsayilan: parquet'teki hepsi)")
    p.add_argument("--cut", type=float, default=None,
                   help="Degerlendirmede kullanilacak kesim (varsayilan: v00.07)")
    p.add_argument("--format", default="native", choices=["native", "joblib", "both"],
                   help="native (.txt+.json, IceTray'de calisir) / joblib (ek pickle)")
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--eval-only", action="store_true",
                   help="Mevcut modeli yukleyip sadece degerlendir")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"=== L4 {args.tag} siniflandirici ===")
    print(f"Girdi: {args.input}")
    df, X, y, w, wp, split, features = load_data(args.input, args.features)

    if args.eval_only:
        path = os.path.join(args.outdir, f"L4_{args.tag}_model.txt")
        model = lgb.Booster(model_file=path)
        print(f"Model yuklendi: {path}")
    else:
        override = {}
        if args.learning_rate:
            override["learning_rate"] = args.learning_rate
        model = train(X, y, w, split, args.tag, override)

    metrics = evaluate(model, X, y, w, wp, split, args.tag, args.outdir, args.cut)

    if not args.eval_only:
        save_model(model, features, args.tag, args.outdir, args.format, metrics,
                   extra_meta=dict(input_parquet=os.path.abspath(args.input)))

    print("\nGrafikler:", args.outdir)


if __name__ == "__main__":
    main()
