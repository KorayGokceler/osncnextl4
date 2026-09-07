#!/usr/bin/env python
'''
I3Classifier'in model dosyasindan ne bekledigini tespit et.

Egitim scriptini yazmadan once BUNU calistirin.  Yanlis semayla kaydedilen bir
model IceTray'de yukleme aninda patlar ve hata mesaji genelde faydasiz olur.

Uc bilgi kaynagi var, oncelik sirasiyla:

  1) Mevcut bir referans .joblib dosyasi  (en guvenilir)
  2) I3Classifier sinifinin kaynak kodu
  3) Sinifin calisma zamanindaki imzasi

Kullanim:

  # Hepsi birden
  python inspect_classifier_api.py \
      --reference /data/oscNext/models/L4_noise_model.joblib

  # Sadece kaynak koda bak (referans dosya yoksa)
  python inspect_classifier_api.py
'''

import os
import sys
import inspect
import argparse
import textwrap


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1) Referans model dosyasini ac ve icini dok
# ---------------------------------------------------------------------------

def inspect_reference(path, depth=0, max_depth=3, name="<root>"):
    '''Kaydedilmis bir model dosyasinin yapisini rekursif olarak yazdir.'''
    import joblib

    if depth == 0:
        hr(f"REFERANS MODEL: {path}")
        try:
            obj = joblib.load(path)
        except Exception as e:
            print(f"[!] Yuklenemedi: {e}")
            print("    (joblib surumu uyumsuz olabilir, ya da dosya pickle degil)")
            return None
    else:
        obj = path

    pad = "  " * depth
    t = type(obj)
    tname = f"{t.__module__}.{t.__name__}"

    if isinstance(obj, dict):
        print(f"{pad}{name}: dict ({len(obj)} anahtar)")
        if depth < max_depth:
            for k, v in obj.items():
                inspect_reference(v, depth + 1, max_depth, name=repr(k))
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {t.__name__} (uzunluk {len(obj)})")
        if obj and depth < max_depth:
            if all(isinstance(x, str) for x in obj):
                print(f"{pad}  -> {list(obj)}")
            else:
                inspect_reference(obj[0], depth + 1, max_depth, name="[0]")
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        print(f"{pad}{name}: {tname} = {obj!r}")
    else:
        print(f"{pad}{name}: {tname}")
        # Model nesnesi mi?
        interesting = ["feature_name_", "feature_names_in_", "n_features_",
                       "n_features_in_", "classes_", "objective_", "booster_",
                       "best_iteration_", "n_estimators"]
        found = {a: getattr(obj, a) for a in interesting if hasattr(obj, a)}
        for a, v in found.items():
            s = repr(v)
            print(f"{pad}  .{a} = {s[:120]}{'...' if len(s) > 120 else ''}")
        if hasattr(obj, "get_params"):
            try:
                p = obj.get_params()
                print(f"{pad}  .get_params() -> {len(p)} parametre")
                for k in ["objective", "num_leaves", "max_depth", "max_bin",
                          "min_child_samples", "reg_alpha", "reg_lambda",
                          "colsample_bytree", "learning_rate", "n_estimators"]:
                    if k in p:
                        print(f"{pad}      {k} = {p[k]}")
            except Exception:
                pass
        # Hangi predict metotlari var?
        methods = [m for m in ("predict", "predict_proba", "decision_function")
                   if hasattr(obj, m)]
        if methods:
            print(f"{pad}  metotlar: {methods}")

    if depth == 0:
        print("\n--- SONUC ---")
        if isinstance(obj, dict):
            print("Sema: SOZLUK sarmalayici.")
            print("Egitim scriptinde --schema bundle kullanin ve anahtar")
            print("isimlerini yukaridaki listeyle BIREBIR eslestirin.")
        else:
            print("Sema: CIPLAK model nesnesi.")
            print("Egitim scriptinde --schema bare kullanin.")
            print("Degisken sirasi I3Classifier tarafinda sabit kodlanmis olmali;")
            print("egitimde de AYNI sirayi kullandiginizdan emin olun.")
        return obj


# ---------------------------------------------------------------------------
# 2) I3Classifier kaynak kodunu oku
# ---------------------------------------------------------------------------

def inspect_source():
    hr("I3Classifier KAYNAK KODU")
    try:
        from icecube.oscNext.tools import classifier as clsmod
    except ImportError as e:
        print(f"[!] Import edilemedi: {e}")
        print("    Bu scripti IceTray ortaminda calistirin.")
        return None

    print("Modul dosyasi:", getattr(clsmod, "__file__", "?"))
    print("Modul icerigi:", [n for n in dir(clsmod) if not n.startswith("_")])

    for cname in ("I3Classifier", "Classifier"):
        cls = getattr(clsmod, cname, None)
        if cls is None:
            continue
        hr(f"class {cname}")
        try:
            print(textwrap.indent(inspect.getsource(cls), "  "))
        except (OSError, TypeError):
            print("  [kaynak okunamadi]")
            print("  __init__ imzasi:", inspect.signature(cls.__init__))
            if cls.__doc__:
                print("  docstring:", cls.__doc__)
    return clsmod


# ---------------------------------------------------------------------------
# 3) Kaynak okunamiyorsa: yukleme mantigini ara
# ---------------------------------------------------------------------------

def grep_load_logic(clsmod):
    '''Kaynak kodda joblib.load sonrasi ne yapildigini bul.'''
    if clsmod is None:
        return
    path = getattr(clsmod, "__file__", None)
    if not path or not os.path.exists(path.rstrip("c")):
        return
    path = path.rstrip("c")
    hr("YUKLEME MANTIGI (joblib/pickle gecen satirlar)")
    src = open(path).read().splitlines()
    for i, line in enumerate(src):
        low = line.lower()
        if any(k in low for k in ("joblib", "pickle", "load(", "class_key",
                                  "feature", "predict", "model[")):
            print(f"{i+1:5d}: {line}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", default=None,
                   help="Mevcut bir .joblib model dosyasi (en guvenilir kaynak)")
    p.add_argument("--max-depth", type=int, default=3)
    args = p.parse_args()

    if args.reference:
        if os.path.exists(args.reference):
            inspect_reference(args.reference, max_depth=args.max_depth)
        else:
            print(f"[!] Bulunamadi: {args.reference}")

    clsmod = inspect_source()
    grep_load_logic(clsmod)

    hr("SONRAKI ADIM")
    print(textwrap.dedent('''
        Yukaridaki ciktiya gore train_L4_classifier.py'de --schema secin:

          bundle  : {"model": ..., "features": [...], ...} sozlugu
          bare    : ciplak LGBMClassifier nesnesi
          custom  : save_model() icindeki BUNDLE_SCHEMA sozlugunu elle duzenleyin

        Referans dosya varsa anahtar isimlerini BIREBIR kopyalayin --
        "features" vs "feature_names" farki bile yuklemeyi bozar.
    ''').strip())


if __name__ == "__main__":
    main()
