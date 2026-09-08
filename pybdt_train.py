#!/usr/bin/env python
'''
oscNext L4 BDT egitimi + dogrulamasi -- saf pybdt.

BAGIMLILIKLAR: pybdt + numpy (ve pybdt.validate'in kendi ihtiyaclari:
scipy, matplotlib).  sklearn / lightgbm / pandas KULLANILMAZ -- IceTray
env-shell'inde bunlarin bulunmadigi varsayilir.

pybdt'nin kendi onerdigi is akisini izler (bkz. pybdt/resources/docs/
man_training.rst, man_validator_setup.rst):

    .ds dosyalari  --BDTLearner-->  .bdt  --Validator-->  grafikler

GIRDI: pybdt native DataSet dosyalari (.ds) -- pybdt.util.save ile
kaydedilmis pybdt.ml.DataSet nesneleri.  Her DataSet:
  * her BDT degiskeni icin bir kolon (numpy dizisi)
  * bir agirlik kolonu (varsayilan adi "weight")

CIKTI (<outdir>/<name>...):
  <name>.bdt          egitilmis BDTModel      (pybdt.util.save)
  <name>.validator    Validator (skorlar onceden hesaplanmis)
  <name>.json         meta: degiskenler, etkin hiperparametreler, KS p
  <name>_overtrain.png, <name>_dist.png, <name>_rate.png

CALISMA ORTAMI: pybdt'nin derlendigi ozel build'in env-shell'i
(bkz. CLAUDE.md).  DIKKAT: "import pybdt" -- "from icecube import pybdt"
DEGIL.

Kullanim:

    python pybdt_train.py --name L4_noise --outdir models_pybdt \
        --sig-train ds/noise_sig_train.ds --bg-train ds/noise_bg_train.ds \
        --sig-test  ds/noise_sig_test.ds  --bg-test  ds/noise_bg_test.ds
'''

import os
import sys
import json
import argparse
import datetime

import numpy as np

try:
    from pybdt import ml, util
except ImportError:
    sys.exit(
        "pybdt import edilemedi.\n"
        "Bu betik SADECE pybdt'nin derlendigi build'in env-shell'i icinde "
        "calisir:\n"
        "  eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/setup.sh)\n"
        "  cd <icetray_build>/build && ./env-shell.sh\n"
        "(Dogru import 'import pybdt', 'from icecube import pybdt' DEGIL.)")


# DataSet'te bulunabilecek ama BDT girdisi OLMAYAN kolonlar.  --features
# verilmediginde bunlar otomatik olarak disarida birakilir.
RESERVED_COLS = {"w_phys", "livetime", "Run", "Event", "SubEvent"}


# ---------------------------------------------------------------------------
# Egitim
# ---------------------------------------------------------------------------

def build_learner(features, weight_col, args):
    '''
    BDTLearner'i kur.  Yalnizca komut satirinda ACIKCA verilen parametreler
    set edilir; geri kalani pybdt'nin kendi varsayilanlarinda birakilir
    (pybdt/resources/scripts/train.py ayni yaklasimi kullanir).
    '''
    learner = ml.BDTLearner(list(features), weight_col, weight_col)

    dt = learner.dtlearner
    if args.depth is not None:
        dt.max_depth = args.depth
    if args.min_split is not None:
        dt.min_split = args.min_split
    if args.num_cuts is not None:
        dt.num_cuts = args.num_cuts
    if args.num_random_variables is not None:
        dt.num_random_variables = args.num_random_variables
    if args.nonlinear_cuts:
        dt.linear_cuts = False

    if args.num_trees is not None:
        learner.num_trees = args.num_trees
    if args.beta is not None:
        learner.beta = args.beta
    if args.frac_random_events is not None:
        learner.frac_random_events = args.frac_random_events
    learner.use_purity = args.use_purity

    learner.add_before_pruner(ml.SameLeafPruner())
    if args.prune_strength is not None:
        learner.add_before_pruner(ml.CostComplexityPruner(args.prune_strength))

    return learner


def effective_params(learner):
    '''Egitimde fiilen kullanilan parametreleri (varsayilanlar dahil) oku.'''
    dt = learner.dtlearner
    return dict(
        num_trees=learner.num_trees,
        beta=learner.beta,
        frac_random_events=learner.frac_random_events,
        use_purity=learner.use_purity,
        max_depth=dt.max_depth,
        min_split=dt.min_split,
        num_cuts=dt.num_cuts,
        num_random_variables=dt.num_random_variables,
        linear_cuts=dt.linear_cuts,
        separation_type=dt.separation_type,
    )


# ---------------------------------------------------------------------------
# Dogrulama
# ---------------------------------------------------------------------------

def build_validator(bdt, datasets, weight_col, use_purity):
    '''
    Validator kur: dort DataSet + agirliklandirmalari.

    datasets: {'train_sig': ds, 'test_sig': ds, 'train_bg': ds, 'test_bg': ds}
    '''
    from pybdt.validate import Validator

    v = Validator(bdt)

    styles = {
        "train_sig": dict(color="tab:blue"),
        "test_sig":  dict(color="tab:cyan"),
        "train_bg":  dict(color="tab:red"),
        "test_bg":   dict(color="tab:orange"),
    }
    labels = {
        "train_sig": "sinyal (train)",
        "test_sig":  "sinyal (test)",
        "train_bg":  "arkaplan (train)",
        "test_bg":   "arkaplan (test)",
    }

    for key, ds in datasets.items():
        # pscores: use_purity ile egitildiyse skorlar da purity ile okunmali
        v.add_data(key, ds, label=labels[key], scores=True, pscores=use_purity)
        v.add_weighting(weight_col, key, **styles[key])

    return v


def score_expr(use_purity):
    return "pscores" if use_purity else "scores"


def score_range(v, keys, expr, pad=0.02):
    vals = np.concatenate([v.get_values_weights(k, expr)[0] for k in keys])
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return (-1.0, 1.0)
    span = hi - lo
    return (lo - pad * span, hi + pad * span)


def make_plots(v, name, outdir, use_purity, bins=60):
    expr = score_expr(use_purity)
    keys = ["train_sig", "test_sig", "train_bg", "test_bg"]
    rng = score_range(v, keys, expr)

    print("\n--- Overtraining kontrolu (pybdt KS testi) ---")
    p_sig = v.get_kolmogorov_smirnov_probability("train_sig", "test_sig", expr=expr)
    p_bg = v.get_kolmogorov_smirnov_probability("train_bg", "test_bg", expr=expr)
    print(f"  p_KS (sinyal train/test)   = {p_sig:.4f}")
    print(f"  p_KS (arkaplan train/test) = {p_bg:.4f}")
    # pybdt dokumantasyonu (man_overtraining.rst): p_KS <~ 0.01 ise
    # overtraining azaltilmali.
    for tag, p in (("sinyal", p_sig), ("arkaplan", p_bg)):
        if p < 0.01:
            print(f"  [!] {tag} p_KS < 0.01 -- OVERTRAINING, agac derinligini "
                  f"dusurun / prune-strength artirin")

    objs = v.create_overtrain_check_plot(
        "train_sig", "test_sig", "train_bg", "test_bg",
        expr=expr, bins=bins, range=rng,
        xlabel="BDT skoru", title=f"{name} — overtraining kontrolu")
    path = os.path.join(outdir, f"{name}_overtrain.png")
    objs["fig"].savefig(path)
    print(f"  -> {path}")

    objs = v.create_plot(
        expr, "dist", keys,
        bins=bins, range=rng, dual=True,
        xlabel="BDT skoru", left_ylabel="agirlikli sayi / bin",
        title=f"{name} — skor dagilimi")
    path = os.path.join(outdir, f"{name}_dist.png")
    objs["fig"].savefig(path)
    print(f"  -> {path}")

    objs = v.create_plot(
        expr, "rate", keys,
        bins=bins, range=rng, dual=True,
        xlabel="BDT skoru", left_ylabel="kesim ustunde kalan agirlik",
        title=f"{name} — kesim degerine karsi oran")
    path = os.path.join(outdir, f"{name}_rate.png")
    objs["fig"].savefig(path)
    print(f"  -> {path}")

    return dict(ks_signal=float(p_sig), ks_background=float(p_bg),
                score_range=[float(rng[0]), float(rng[1])])


def report_efficiency(v, use_purity, cut):
    '''Verilen kesimde sinyal verimi / arkaplan reddi (test setinde).'''
    expr = score_expr(use_purity)
    out = {}
    for key, tag in (("test_sig", "sinyal"), ("test_bg", "arkaplan")):
        a, w = v.get_values_weights(key, expr)
        total = w.sum()
        kept = w[a >= cut].sum()
        frac = kept / total if total > 0 else float("nan")
        out[key] = dict(total=float(total), kept=float(kept), fraction=float(frac))
        print(f"  {tag:9s}: {total:.4e} -> {kept:.4e}  ({100 * frac:.1f}% kaldi)")
    return out


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="oscNext L4 BDT egitimi + dogrulamasi (saf pybdt)")
    p.add_argument("--name", required=True,
                   help="Cikti dosyalarinin on eki, orn. L4_noise")
    p.add_argument("--outdir", default="./models_pybdt")

    p.add_argument("--sig-train", required=True, help="Egitim sinyali (.ds)")
    p.add_argument("--bg-train", required=True, help="Egitim arkaplani (.ds)")
    p.add_argument("--sig-test", required=True, help="Test sinyali (.ds)")
    p.add_argument("--bg-test", required=True, help="Test arkaplani (.ds)")

    p.add_argument("--features", default=None,
                   help="Virgulle ayrilmis degisken listesi. Verilmezse "
                        "egitim sinyali DataSet'indeki tum kolonlar "
                        "(agirlik kolonu haric) kullanilir.")
    p.add_argument("--weight-col", default="weight",
                   help="Agirlik kolonunun adi (varsayilan: weight)")

    # --- decision tree ---
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--min-split", type=int, default=None)
    p.add_argument("--num-cuts", type=int, default=None)
    p.add_argument("--num-random-variables", type=int, default=None)
    p.add_argument("--nonlinear-cuts", action="store_true")
    # --- forest ---
    p.add_argument("--num-trees", type=int, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--frac-random-events", type=float, default=None)
    p.add_argument("--prune-strength", type=float, default=None)
    p.add_argument("--use-purity", action="store_true",
                   help="SAMME.R -- yaprak purity bilgisini kullan")

    p.add_argument("--cut", type=float, default=None,
                   help="Verim/red raporu icin kesim degeri (pybdt skor "
                        "olceginde; LightGBM'in 0-1 olasiligi DEGIL)")
    p.add_argument("--bins", type=int, default=60)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"=== {args.name} — pybdt/AdaBoost ===")
    ds = {
        "train_sig": util.load(args.sig_train),
        "train_bg":  util.load(args.bg_train),
        "test_sig":  util.load(args.sig_test),
        "test_bg":   util.load(args.bg_test),
    }
    for k, d in ds.items():
        print(f"  {k:10s} {len(d):>10,} olay")

    if args.features:
        features = [f.strip() for f in args.features.split(",") if f.strip()]
    else:
        # DataSet'te BDT girdisi OLMAYAN kolonlar da bulunabilir (fiziksel
        # agirlik, livetime...).  Bunlar egitime sizarsa model agirligi bir
        # degisken sanip ogrenir -- sessiz ve ciddi bir hata.
        features = [n for n in ds["train_sig"].names
                    if n not in RESERVED_COLS and n != args.weight_col]
        print(f"  [i] --features verilmedi -> {args.weight_col} ve "
              f"{sorted(RESERVED_COLS)} disindaki tum kolonlar kullaniliyor")
    print(f"  degisken   {len(features)}: {', '.join(features)}")

    for key, d in ds.items():
        missing = [f for f in features + [args.weight_col] if f not in d.names]
        if missing:
            sys.exit(f"{key} DataSet'inde eksik kolon(lar): {missing}")

    learner = build_learner(features, args.weight_col, args)
    params = effective_params(learner)
    print("\nHiperparametreler (etkin):")
    for k, val in params.items():
        print(f"  {k:22s} {val}")

    print("\nEgitim ...")
    bdt = learner.train(ds["train_sig"], ds["train_bg"])
    print(f"  bitti -- {len(bdt)} agac")

    bdt_path = os.path.join(args.outdir, f"{args.name}.bdt")
    util.save(bdt, bdt_path)
    print(f"  -> {bdt_path} ({os.path.getsize(bdt_path) / 1024:.0f} KB)")

    v = build_validator(bdt, ds, args.weight_col, args.use_purity)
    val_path = os.path.join(args.outdir, f"{args.name}.validator")
    util.save(v, val_path)
    print(f"  -> {val_path}")

    metrics = make_plots(v, args.name, args.outdir, args.use_purity, args.bins)

    if args.cut is not None:
        print(f"\n--- Kesim {args.cut} (test seti) ---")
        metrics["cut_performance"] = report_efficiency(v, args.use_purity, args.cut)
        metrics["cut"] = args.cut

    meta = dict(
        name=args.name,
        engine="pybdt",
        features=features,
        weight_col=args.weight_col,
        use_purity=args.use_purity,
        score_expr=score_expr(args.use_purity),
        params=params,
        n_trees=len(bdt),
        inputs=dict(sig_train=os.path.abspath(args.sig_train),
                    bg_train=os.path.abspath(args.bg_train),
                    sig_test=os.path.abspath(args.sig_test),
                    bg_test=os.path.abspath(args.bg_test)),
        trained=datetime.datetime.now().isoformat(timespec="seconds"),
        metrics=metrics,
    )
    meta_path = os.path.join(args.outdir, f"{args.name}.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"\n  -> {meta_path}")

    back = util.load(bdt_path)
    assert len(back) == len(bdt), "Geri yuklenen model agac sayisi uyusmuyor"
    print(f"  geri yukleme testi: OK ({len(back)} agac)")


if __name__ == "__main__":
    main()
