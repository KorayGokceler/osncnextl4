#!/usr/bin/env python
'''
IceTray ortaminda neyin mevcut oldugunu tespit et.

Bu scripti IceTray ortaminda calistirin ve ciktinin tamamini paylasin.
Sonuca gore hangi fallback'lerin gerektigine karar verilir.
'''

import os
import sys
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("PYTHON")
print("=" * 70)
print(sys.version)
print(sys.executable)

# icecube hic import edilemiyorsa detayli teshis basar ve cikar --
# asagidaki tum kontroller anlamsiz olur.
try:
    import icetray_env as _ienv
except ImportError:
    _ienv = None

if _ienv is not None and not _ienv.have_icetray():
    print()
    print(_ienv._env_report())
    print()
    print("Bulunan env-shell.sh adaylari:")
    _c = _ienv.find_env_shells()
    if not _c:
        print("  (yok)  -> export OSCNEXT_I3_BUILD=/tam/yol/build")
    for _kind, _path in _c:
        print("  [%-12s] %s" % (_kind, _path))
    sys.exit(1)

if _ienv is not None:
    print()
    print("=" * 70)
    print("ORTAM")
    print("=" * 70)
    for _v in ("I3_BUILD", "I3_SRC", "SROOT"):
        print("  %-10s %s" % (_v, os.environ.get(_v, "<bos>")))
    try:
        print("  %-10s %s" % ("I3Tray", _ienv.get_I3Tray()))
    except Exception as _e:
        print("  %-10s BULUNAMADI -- %s" % ("I3Tray", _e))


def try_import(name, note=""):
    try:
        m = importlib.import_module(name)
        path = getattr(m, "__file__", "<builtin>")
        print(f"  [OK]      {name:42s} {note}")
        return m
    except Exception as e:
        print(f"  [YOK]     {name:42s} {type(e).__name__}: {e}")
        return None


print("\n" + "=" * 70)
print("TEMEL ICETRAY")
print("=" * 70)
for n in ["icecube", "icecube.icetray", "icecube.dataclasses",
          "icecube.dataio", "icecube.phys_services"]:
    try_import(n)

print("\n" + "=" * 70)
print("BOOKING / TABLO YAZMA  (adim 0'da eksik cikan kisim)")
print("=" * 70)
tableio  = try_import("icecube.tableio",  "genel tablo altyapisi")
hdfw     = try_import("icecube.hdfwriter", "HDF5 yazici")
try_import("icecube.rootwriter", "ROOT yazici (alternatif)")
try_import("tables", "pytables -- fallback booker icin GEREKLI")
try_import("h5py", "h5py -- alternatif fallback")

if tableio and not hdfw:
    print("\n  -> tableio VAR ama hdfwriter YOK.")
    print("     Meta-proje HDF5 gelistirme kutuphaneleri olmadan derlenmis.")
    print("     Cozum: fallback booker (pytables ile dogrudan yazma).")
elif not tableio and not hdfw:
    print("\n  -> Tablo altyapisi hic yok. Fallback booker sart.")

print("\n" + "=" * 70)
print("L4 DEGISKENLERI ICIN GEREKEN PROJELER")
print("=" * 70)
try_import("icecube.DomTools",          "I3OMSelection, I3TimeWindowCleaning")
try_import("icecube.STTools",           "SeededRT temizleme")
try_import("icecube.linefit",           "improved LineFit")
try_import("icecube.tensor_of_inertia", "I3TensorOfInertia")
try_import("icecube.fill_ratio",        "I3FillRatioModule")
try_import("icecube.common_variables",  "HitStatistics / HitMultiplicity")
try_import("icecube.DeepCore_Filter",   "DOMS -- fiducial/veto listeleri")

print("\n  C++ modul kutuphaneleri:")
try:
    from icecube import icetray
    for lib in ["static-twc", "slc-veto", "DomTools", "fill-ratio"]:
        try:
            icetray.load(lib, False)
            print(f"  [OK]      {lib}")
        except Exception as e:
            print(f"  [YOK]     {lib:20s} {e}")
except Exception as e:
    print("  icetray yuklenemedi:", e)

print("\n" + "=" * 70)
print("PYBDT  (BDT egitimi icin -- derlenmis mi diye kontrol)")
print("=" * 70)
pybdt = try_import("icecube.pybdt")
if pybdt:
    try_import("icecube.pybdt.ml", "BDTLearner, DecisionTreeLearner, ...")
    print("\n  -> pybdt zaten derlenmis geliyor, dogrudan kullanilabilir.")
else:
    print("\n  -> pybdt derlenmis degil. pybdt/ klasorundeki kaynak kod")
    print("     (C++/boost_python) IceTray meta-project build sistemi")
    print("     icinde cmake ile derlenmeden calismaz -- sadece dosyalarin")
    print("     var olmasi yetmez.")

print("\n" + "=" * 70)
print("oscNext PROJESI  (adim 1'de eksik cikan kisim)")
print("=" * 70)
osc = try_import("icecube.oscNext")
for sub in ["icecube.oscNext.tools.classifier",
            "icecube.oscNext.selection.oscNext_cuts",
            "icecube.oscNext.frame_objects.geom"]:
    try_import(sub)

if not osc:
    print("\n  -> oscNext projesi YOK.  Sonuclari:")
    print("     * I3Classifier kullanilamaz -> kendi uygulama modulumuzu yazacagiz")
    print("     * oscNext_cut kullanilamaz  -> fallback zaten var (L3 bools okur)")
    print("     * calc_rho_36 kullanilamaz  -> fallback zaten var")

print("\n" + "=" * 70)
print("L3 CIKTISI  (girdi dosyalarinizda ne var?)")
print("=" * 70)
print("Bir L3 dosyasinin ilk Physics frame'ini dokmek icin:")
print()
print("  python -c \"")
print("from icecube import dataio, dataclasses, icetray")
print("f = dataio.I3File('<L3_DOSYANIZ>.i3.zst')")
print("while f.more():")
print("    fr = f.pop_frame()")
print("    if fr.Stop == icetray.I3Frame.Physics:")
print("        for k in sorted(fr.keys()):")
print("            print(f'{k:50s} {type(fr[k]).__name__}')")
print("        break")
print("\"")
print()
print("Ozellikle sunlarin varligini kontrol edin:")
for k in ["IC2018_LE_L3_Vars", "IC2018_LE_L3_bools",
          "SRTTWOfflinePulsesDC", "SplitInIcePulses",
          "SRTTWOfflinePulsesDCHitStatistics", "I3MCWeightDict"]:
    print(f"    {k}")

print("\n" + "=" * 70)
print("SKLEARN / LIGHTGBM  (siniflandirici uygulamasi icin)")
print("=" * 70)
print("Bunlar IceTray ortaminda da gerekli -- model orada yuklenecek:")
for n in ["numpy", "sklearn", "lightgbm", "joblib", "pandas"]:
    m = try_import(n)
    if m and hasattr(m, "__version__"):
        print(f"            surum: {m.__version__}")
