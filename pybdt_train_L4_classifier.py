#!/usr/bin/env python
'''
oscNext L4 siniflandirici egitimi -- pybdt (AdaBoost) surumu.

train_L4_classifier.py'nin resmi (LightGBM, teknik nota uygun) surumune
BILINCLI bir ALTERNATIF. bkz. CLAUDE.md "BDT egitimi: pybdt kullanilacak"
bolumu -- bu resmi oscNext yontemi DEGIL, kullanicinin kendi tercihi.

Farklar (train_L4_classifier.py ile karsilastirmali):
  * Motor       : pybdt AdaBoost (LightGBM gradient boosting DEGIL)
  * Hiperparam. : Tablo 10'daki degerler DOGRUDAN TASINAMAZ -- pybdt'nin
                  kendi API'si farkli (num_trees/beta/depth/min_split/
                  prune_strength). Asagidaki PYBDT_PARAMS pybdt'nin kendi
                  ornek betiginden (resources/examples/train_sample_bdt.sh,
                  IC79 numu analizi) alinmis GENEL varsayilanlardir --
                  oscNext icin optimize EDILMEMISTIR.
  * Skor araligi: use_purity=True ile SAMME.R olasilik-benzeri skor
                  ([0,1]'e yakin) uretir ama LightGBM'in P(sinyal)
                  ciktisiyla BIREBIR ayni olcek degildir -- kesim degeri
                  (--cut) bu script icin ayrica belirlenmeli.
  * Model format: pybdt.util.save (pickle, sadece pybdt kurulu ortamda
                  okunur) -- LightGBM'deki .txt/.json fallback'i yok.

Veri yukleme (parquet -> DataFrame -> features/labels/weights/split)
train_L4_classifier.py ile ORTAK, oradan import edilir -- iki ayri
implementasyon senkron kayabilir diye.

CALISMA ORTAMI: SADECE pybdt'nin derlendigi ozel build'in env-shell'i
icinde calisir (pybdt py3-v4.4.2 dagitiminda YOK, kaynaktan derlendi --
bkz. CLAUDE.md):

    eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/setup.sh)
    cd /data/user/kgokcele/icetray_build/build && ./env-shell.sh
    cd ~/l4
    python pybdt_train_L4_classifier.py --tag noise \
        --input ./l4_training_data/L4_noise_training.parquet \
        --outdir ./models_pybdt
'''

import os
import sys
import json
import argparse
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

from train_L4_classifier import load_data, POSITIVE_CLASS, NEGATIVE_CLASS

try:
    from pybdt import ml, util
except ImportError:
    sys.exit(
        "icecube.pybdt import edilemedi.\n"
        "Bu betik SADECE pybdt'nin derlendigi ozel build'in env-shell'i "
        "icinde calisir:\n"
        "  eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/setup.sh)\n"
        "  cd /data/user/kgokcele/icetray_build/build && ./env-shell.sh")


# ===========================================================================
# Hiperparametreler -- pybdt'nin kendi ornek degerleri (bkz. modul docstring).
# DIKKAT: oscNext icin optimize edilmemis, sadece baslangic noktasi.
# ===========================================================================

PYBDT_PARAMS = {
    "noise": dict(
        num_trees=300, beta=0.7, max_depth=3, min_split=500,
        num_cuts=100, linear_cuts=True, num_random_variables=0,
        frac_random_events=0.5, use_purity=True, prune_strength=35.0,
    ),
    "muon": dict(
        num_trees=300, beta=0.7, max_depth=3, min_split=500,
        num_cuts=100, linear_cuts=True, num_random_variables=0,
        frac_random_events=0.5, use_purity=True, prune_strength=35.0,
    ),
}

DEFAULT_CUT = {"noise": 0.5, "muon": 0.5}  # pybdt icin referans yok -- baslangic


# ===========================================================================
# DataSet yardimcilari
# ===========================================================================

def make_dataset(X, w, feature_names):
    '''pandas alt-kumesinden pybdt.ml.DataSet olustur.'''
    data = {f: X[f].to_numpy(dtype=float) for f in feature_names}
    data["weight"] = np.asarray(w, dtype=float)
    return ml.DataSet(data)


def score(model, X, feature_names, use_purity=True):
    ds = ml.DataSet({f: X[f].to_numpy(dtype=float) for f in feature_names})
    return np.asarray(model.score_DataSet(ds, use_purity=use_purity, quiet=True))


# ===========================================================================
# Egitim
# ===========================================================================

def train(X, y, w, split, tag, feature_names, params_override=None):
    p = dict(PYBDT_PARAMS[tag])
    if params_override:
        p.update(params_override)

    tr = (split == "train")
    if not tr.any():
        sys.exit("Train seti bos -- parquet'teki 'split' kolonunu kontrol edin.")

    sig_mask = tr & (y == 1)
    bkg_mask = tr & (y == 0)
    if sig_mask.sum() == 0 or bkg_mask.sum() == 0:
        sys.exit("Train setinde sinyal veya arkaplan olayi yok.")

    sig_ds = make_dataset(X[sig_mask], w[sig_mask], feature_names)
    bkg_ds = make_dataset(X[bkg_mask], w[bkg_mask], feature_names)

    learner = ml.BDTLearner(feature_names, "weight", "weight")
    learner.dtlearner.linear_cuts = p["linear_cuts"]
    learner.dtlearner.max_depth = p["max_depth"]
    learner.dtlearner.min_split = p["min_split"]
    learner.dtlearner.num_cuts = p["num_cuts"]
    learner.dtlearner.num_random_variables = p["num_random_variables"]

    learner.num_trees = p["num_trees"]
    learner.beta = p["beta"]
    learner.frac_random_events = p["frac_random_events"]
    learner.use_purity = p["use_purity"]

    learner.add_before_pruner(ml.SameLeafPruner())
    if p.get("prune_strength"):
        learner.add_before_pruner(ml.CostComplexityPruner(p["prune_strength"]))

    print(f"\nEgitim ({tag}, pybdt/AdaBoost) ...")
    print(f"  sinyal={int(sig_mask.sum()):,}  arkaplan={int(bkg_mask.sum()):,}"
          f"  agac={p['num_trees']}  beta={p['beta']}  depth={p['max_depth']}")
    bdt = learner.train(sig_ds, bkg_ds)
    print(f"  bitti -- {len(bdt)} agac")
    return bdt, p


# ===========================================================================
# Degerlendirme (train_L4_classifier.py'nin evaluate()'iyle ayni yapida)
# ===========================================================================

def evaluate(model, X, y, w, wp, split, tag, outdir, feature_names,
             use_purity=True, cut=None):
    cut = cut if cut is not None else DEFAULT_CUT[tag]
    tr, te = split == "train", split == "test"
    prob = score(model, X, feature_names, use_purity=use_purity)

    auc_tr = roc_auc_score(y[tr], prob[tr], sample_weight=w[tr])
    auc_te = roc_auc_score(y[te], prob[te], sample_weight=w[te])

    print(f"\n--- Performans ({tag}, pybdt) ---")
    print(f"  AUC train = {auc_tr:.4f}")
    print(f"  AUC test  = {auc_te:.4f}")
    print(f"  fark      = {auc_tr - auc_te:+.4f}", end="  ")
    print("<-- OVERTRAINING" if auc_tr - auc_te > 0.01 else "(iyi)")

    # --- train/test dagilimi ---
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    lo, hi = float(np.min(prob)), float(np.max(prob))
    bins = np.linspace(lo, hi, 41)
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
    ax.set_yscale("log"); ax.set_xlabel("pybdt skoru")
    ax.set_ylabel("normalize yogunluk"); ax.legend(fontsize=7)
    ax.set_title(f"L4 {tag} (pybdt) — train/test uyumu", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{outdir}/L4_{tag}_pybdt_train_test.png", dpi=130)
    plt.close(fig)

    # --- kesim degerine karsi rate ---
    perf = None
    if np.isfinite(wp).any():
        cuts = np.linspace(lo, hi, 101)
        sig_rate = [wp[(y == 1) & (prob >= c)].sum() for c in cuts]
        bkg_rate = [wp[(y == 0) & (prob >= c)].sum() for c in cuts]
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.plot(cuts, sig_rate, label=POSITIVE_CLASS, color="tab:blue")
        ax.plot(cuts, bkg_rate, label=NEGATIVE_CLASS[tag], color="tab:red")
        ax.axvline(cut, color="k", ls="--", lw=1)
        ax.set_yscale("log"); ax.set_xlabel("kesim degeri (pybdt skoru)")
        ax.set_ylabel("Rate [1/s]"); ax.legend(fontsize=8)
        ax.set_title(f"L4 {tag} (pybdt) — kesim degerine karsi rate", fontsize=10)
        fig.tight_layout(); fig.savefig(f"{outdir}/L4_{tag}_pybdt_rate_vs_cut.png", dpi=130)
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

    # --- ROC ---
    fig, ax = plt.subplots(figsize=(4.2, 4))
    for sset, name in [(tr, "train"), (te, "test")]:
        fpr, tpr, _ = roc_curve(y[sset], prob[sset], sample_weight=w[sset])
        a = auc_tr if name == "train" else auc_te
        ax.plot(fpr, tpr, lw=1.4, label=f"{name} (AUC={a:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("yanlis pozitif oran"); ax.set_ylabel("dogru pozitif oran")
    ax.legend(fontsize=8); ax.set_title(f"L4 {tag} (pybdt) ROC", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{outdir}/L4_{tag}_pybdt_roc.png", dpi=130)
    plt.close(fig)

    # --- degisken onemi (pybdt native) ---
    importance = model.variable_importance(sep_weight=True, tree_weight=True)
    imp_items = sorted(importance.items(), key=lambda kv: -kv[1])
    fig, ax = plt.subplots(figsize=(6, 0.32 * len(imp_items) + 1.6))
    names = [k for k, _ in imp_items][::-1]
    vals = [100 * v for _, v in imp_items][::-1]
    ax.barh(names, vals, color="tab:blue")
    ax.set_xlabel("onem [%]"); ax.set_title(f"L4 {tag} (pybdt) variable importance", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{outdir}/L4_{tag}_pybdt_importance.png", dpi=130)
    plt.close(fig)
    print("\n--- Variable importance (pybdt) ---")
    for k, v in imp_items:
        print(f"  {k:30s} {100*v:6.2f}%")

    return dict(auc_train=float(auc_tr), auc_test=float(auc_te),
                overtraining=float(auc_tr - auc_te),
                num_trees=len(model),
                cut_performance=perf,
                importance=dict(importance))


# ===========================================================================
# Kaydetme
# ===========================================================================

def save_model(bdt, feature_names, tag, outdir, params, metrics):
    '''
    pybdt.util.save ile kaydet (pickle, sadece pybdt kurulu ortamda okunur).

    LightGBM surumundeki .txt/.json fallback'i pybdt'de yok -- bu model
    SADECE icecube.pybdt derlenmis bir ortamda (bkz. modul docstring)
    yuklenebilir. pybdt_classifier_module.py bu dosyayi okur.
    '''
    pkl_path = os.path.join(outdir, f"L4_{tag}_pybdt_model.pkl")
    json_path = os.path.join(outdir, f"L4_{tag}_pybdt_model.json")

    util.save(bdt, pkl_path)

    sidecar = dict(
        tag=tag,
        engine="pybdt",
        features=list(feature_names),
        positive_class=POSITIVE_CLASS,
        negative_class=NEGATIVE_CLASS[tag],
        use_purity=params["use_purity"],
        default_cut=DEFAULT_CUT[tag],
        params=params,
        trained=datetime.datetime.now().isoformat(timespec="seconds"),
        metrics={k: v for k, v in metrics.items() if k != "importance"},
    )
    with open(json_path, "w") as fh:
        json.dump(sidecar, fh, indent=2, default=str)

    print(f"\nModel kaydedildi:")
    print(f"  {pkl_path}   ({os.path.getsize(pkl_path)/1024:.0f} KB)")
    print(f"  {json_path}")

    # geri yukleme testi
    back = util.load(pkl_path)
    assert len(back) == len(bdt), "Geri yuklenen model agac sayisi uyusmuyor"
    print(f"  geri yukleme testi: OK ({len(back)} agac)")


# ===========================================================================

def main():
    p = argparse.ArgumentParser(description="oscNext L4 siniflandirici egitimi (pybdt)")
    p.add_argument("--tag", required=True, choices=["noise", "muon"])
    p.add_argument("--input", required=True, help="Notebook'un urettigi parquet")
    p.add_argument("--outdir", default="./models_pybdt")
    p.add_argument("--features", nargs="+", default=None)
    p.add_argument("--cut", type=float, default=None)
    p.add_argument("--num-trees", type=int, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--no-purity", action="store_true",
                   help="use_purity=False (skor [-1,1] araliginda, SAMME.R degil)")
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"=== L4 {args.tag} siniflandirici (pybdt/AdaBoost) ===")
    print(f"Girdi: {args.input}")
    df, X, y, w, wp, split, features = load_data(args.input, args.features)

    override = {}
    if args.num_trees:
        override["num_trees"] = args.num_trees
    if args.beta:
        override["beta"] = args.beta
    if args.depth:
        override["max_depth"] = args.depth
    if args.no_purity:
        override["use_purity"] = False

    if args.eval_only:
        path = os.path.join(args.outdir, f"L4_{args.tag}_pybdt_model.pkl")
        bdt = util.load(path)
        params = dict(PYBDT_PARAMS[args.tag], **override)
        print(f"Model yuklendi: {path}")
    else:
        bdt, params = train(X, y, w, split, args.tag, features, override)

    metrics = evaluate(bdt, X, y, w, wp, split, args.tag, args.outdir, features,
                       use_purity=params["use_purity"], cut=args.cut)

    if not args.eval_only:
        save_model(bdt, features, args.tag, args.outdir, params, metrics)

    print("\nGrafikler:", args.outdir)


if __name__ == "__main__":
    main()
