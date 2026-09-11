#!/usr/bin/env python
'''
Report what is actually available in the IceTray environment.

Run this inside the IceTray environment and share the whole output.  Which
fallbacks are needed is decided from the result.
'''

import os
import sys
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))     # repo root, for `oscnext_l4`

print("=" * 70)
print("PYTHON")
print("=" * 70)
print(sys.version)
print(sys.executable)

# When icecube cannot be imported at all, print the detailed diagnosis and
# leave -- every check below would be meaningless.
try:
    from oscnext_l4 import env as _ienv
except ImportError:
    _ienv = None

if _ienv is not None and not _ienv.have_icetray():
    print()
    print(_ienv._env_report())
    print()
    print("env-shell.sh candidates found:")
    _c = _ienv.find_env_shells()
    if not _c:
        print("  (none)  -> export OSCNEXT_I3_BUILD=/full/path/build")
    for _kind, _path in _c:
        print("  [%-12s] %s" % (_kind, _path))
    sys.exit(1)

if _ienv is not None:
    print()
    print("=" * 70)
    print("ENVIRONMENT")
    print("=" * 70)
    for _v in ("I3_BUILD", "I3_SRC", "SROOT"):
        print("  %-10s %s" % (_v, os.environ.get(_v, "<unset>")))
    try:
        print("  %-10s %s" % ("I3Tray", _ienv.get_I3Tray()))
    except Exception as _e:
        print("  %-10s NOT FOUND -- %s" % ("I3Tray", _e))


def try_import(name, note=""):
    try:
        m = importlib.import_module(name)
        path = getattr(m, "__file__", "<builtin>")
        print(f"  [OK]      {name:42s} {note}")
        return m
    except Exception as e:
        print(f"  [MISSING] {name:42s} {type(e).__name__}: {e}")
        return None


print("\n" + "=" * 70)
print("CORE ICETRAY")
print("=" * 70)
for n in ["icecube", "icecube.icetray", "icecube.dataclasses",
          "icecube.dataio", "icecube.phys_services"]:
    try_import(n)

print("\n" + "=" * 70)
print("BOOKING / TABLE WRITING")
print("=" * 70)
tableio  = try_import("icecube.tableio",  "generic table infrastructure")
hdfw     = try_import("icecube.hdfwriter", "HDF5 writer")
try_import("icecube.rootwriter", "ROOT writer (alternative)")
try_import("tables", "pytables -- REQUIRED by the fallback booker")
try_import("h5py", "h5py -- alternative fallback")

if tableio and not hdfw:
    print("\n  -> tableio is present but hdfwriter is NOT.")
    print("     The meta-project was built without the HDF5 development libraries.")
    print("     Remedy: the fallback booker (writes through pytables directly).")
elif not tableio and not hdfw:
    print("\n  -> No table infrastructure at all.  The fallback booker is mandatory.")

print("\n" + "=" * 70)
print("PROJECTS REQUIRED BY THE L4 VARIABLES")
print("=" * 70)
try_import("icecube.DomTools",          "I3OMSelection, I3TimeWindowCleaning")
try_import("icecube.STTools",           "SeededRT cleaning")
try_import("icecube.linefit",           "improved LineFit")
try_import("icecube.tensor_of_inertia", "I3TensorOfInertia")
try_import("icecube.fill_ratio",        "I3FillRatioModule")
try_import("icecube.common_variables",  "HitStatistics / HitMultiplicity")
try_import("icecube.DeepCore_Filter",   "DOMS -- fiducial/veto lists")

print("\n  C++ module libraries:")
try:
    from icecube import icetray
    for lib in ["static-twc", "slc-veto", "DomTools", "fill-ratio"]:
        try:
            icetray.load(lib, False)
            print(f"  [OK]      {lib}")
        except Exception as e:
            print(f"  [MISSING] {lib:20s} {e}")
except Exception as e:
    print("  icetray could not be loaded:", e)

print("\n" + "=" * 70)
print("LIGHTGBM  (classifier training and application)")
print("=" * 70)
lgbm = try_import("lightgbm")
if lgbm:
    print("\n  -> lightgbm is present; training and application will work.")
else:
    print("\n  -> lightgbm is MISSING.  The L4 classifiers cannot be trained without it:")
    print("     pip install --user lightgbm")

print("\n" + "=" * 70)
print("oscNext PROJECT")
print("=" * 70)
osc = try_import("icecube.oscNext")
for sub in ["icecube.oscNext.tools.classifier",
            "icecube.oscNext.selection.oscNext_cuts",
            "icecube.oscNext.frame_objects.geom"]:
    try_import(sub)

if not osc:
    print("\n  -> The oscNext project is MISSING.  Consequences:")
    print("     * I3Classifier unavailable -> we ship our own application module")
    print("     * oscNext_cut unavailable  -> fallback exists (reads the L3 bools)")
    print("     * calc_rho_36 unavailable  -> fallback exists")

print("\n" + "=" * 70)
print("L3 OUTPUT  (what is in your input files?)")
print("=" * 70)
print("To dump the first Physics frame of an L3 file:")
print()
print("  python -c \"")
print("from icecube import dataio, dataclasses, icetray")
print("f = dataio.I3File('<YOUR_L3_FILE>.i3.zst')")
print("while f.more():")
print("    fr = f.pop_frame()")
print("    if fr.Stop == icetray.I3Frame.Physics:")
print("        for k in sorted(fr.keys()):")
print("            print(f'{k:50s} {type(fr[k]).__name__}')")
print("        break")
print("\"")
print()
print("Check in particular that these exist:")
for k in ["IC2018_LE_L3_Vars", "IC2018_LE_L3_bools",
          "SRTTWOfflinePulsesDC", "SplitInIcePulses",
          "SRTTWOfflinePulsesDCHitStatistics", "I3MCWeightDict"]:
    print(f"    {k}")

print("\n" + "=" * 70)
print("PYTHON ML PACKAGES")
print("=" * 70)
print("numpy and lightgbm are needed in the IceTray environment too -- the\nmodel is loaded there.  sklearn / joblib / pandas are listed for information\nonly; nothing in this repo imports them:")
for n in ["numpy", "sklearn", "lightgbm", "joblib", "pandas"]:
    m = try_import(n)
    if m and hasattr(m, "__version__"):
        print(f"            version: {m.__version__}")
