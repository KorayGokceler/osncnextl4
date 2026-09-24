#!/usr/bin/env python
'''
Report what is actually available in the IceTray environment.

Run this inside the IceTray environment and share the whole output.  The
pipeline expects IceTray v1.17.0 from cvmfs (py3-v4.4.2); what it reports
missing there is either expected (oscNext, tau_bdt, ... -- rewritten.py and
frame_objects.py stand in for them) or a broken environment.
'''

import os
import sys
import importlib
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))     # repo root, for `oscnext_l4`

print("=" * 70)
print("PYTHON")
print("=" * 70)
print(sys.version)
print(sys.executable)

print()
print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)
for _v in ("I3_BUILD", "I3_SRC", "SROOT"):
    print("  %-10s %s" % (_v, os.environ.get(_v, "<unset>")))


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
# I3Tray is a CLASS in icecube.icetray (v1.5+), not a submodule, so
# import_module("icecube.icetray.I3Tray") would report it missing wrongly.
try:
    from icecube.icetray import I3Tray  # noqa: F401
    print(f"  [OK]      {'from icecube.icetray import I3Tray':42s}")
except Exception as e:
    print(f"  [MISSING] {'from icecube.icetray import I3Tray':42s} "
          f"{type(e).__name__}: {e}")

if not importlib.util.find_spec("icecube"):
    print("\n  -> this python (%s) is not inside IceTray, so nothing below can"
          % sys.executable)
    print("     be judged.  Load the environment and run this again:")
    print("       eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/setup.sh)")
    print("       /cvmfs/icecube.opensciencegrid.org/py3-v4.4.2/RHEL_9_x86_64_v2/"
          "metaprojects/icetray/v1.17.0/env-shell.sh -- python "
          "scripts/diagnose_env.py")
    sys.exit(1)

print("\n" + "=" * 70)
print("BOOKING / TABLE WRITING")
print("=" * 70)
tableio  = try_import("icecube.tableio",  "generic table infrastructure")
hdfw     = try_import("icecube.hdfwriter", "HDF5 writer")
try_import("icecube.rootwriter", "ROOT writer (alternative)")
try_import("tables", "pytables -- REQUIRED to READ the booked files (data.py)")
try_import("h5py", "h5py -- not used, informational")

if not hdfw:
    print("\n  -> hdfwriter is NOT here, and booking has no fallback.")
    print("     It is also what writes /__I3Index__/<key>, which is the only")
    print("     reliable way to match a table to its events -- so this breaks")
    print("     booking AND the pass2 cross-check.  See oscnext_l4/tray_io.py.")
elif tableio:
    print("\n  -> hdfwriter is present but DEPRECATED (v1.17.0 warns on import).")
    print("     tableio is the generic engine and hdfwriter holds its HDF5 back")
    print("     end, so booking needs hdfwriter either way (oscnext_l4/tray_io.py).")

print("\n" + "=" * 70)
print("PROJECTS REQUIRED BY THE L4 VARIABLES")
print("=" * 70)
try_import("icecube.DomTools",          "I3OMSelection, I3TimeWindowCleaning")
try_import("icecube.linefit",           "improved LineFit")
try_import("icecube.tensor_of_inertia", "I3TensorOfInertia (--run-optional only)")
try_import("icecube.fill_ratio",        "I3FillRatioModule")
try_import("icecube.common_variables",  "HitStatistics / HitMultiplicity")
try_import("icecube.DeepCore_Filter",   "DOMS -- fiducial/veto lists")

print("\n  C++ module libraries:")
try:
    from icecube import icetray
    for lib in ["static-twc", "slc-veto (--run-optional only)"]:
        lib = lib.split()[0]
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
    print("\n  -> lightgbm is MISSING.  It ships inside the cvmfs metaproject, so this")
    print("     python is most likely not the env-shell one (%s)." % sys.executable)

print("\n" + "=" * 70)
print("oscNext PROJECT")
print("=" * 70)
osc = try_import("icecube.oscNext")
for sub in ["icecube.oscNext.tools.classifier",
            "icecube.oscNext.selection.oscNext_cuts",
            "icecube.oscNext.frame_objects.geom"]:
    try_import(sub)

if not osc:
    print("\n  -> The oscNext project is MISSING -- EXPECTED in v1.17.0; this")
    print("     package does not import it:")
    print("     * I3Classifier -> oscnext_l4/classifier.py")
    print("     * oscNext_cut  -> the L3 cut in oscnext_l4/variables.py")
    print("     * calc_rho_36  -> oscnext_l4/frame_objects.py")

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
for k in ["IC2018_LE_L3_Vars", "IC2018_LE_L3_bools", "SplitInIcePulses",
          "SRTTWSplitInIcePulsesDC  (pass3; pass2: SRTTWOfflinePulsesDC, "
          "then give --cleaned-pulses)",
          "I3MCWeightDict  (simulation)"]:
    print(f"    {k}")

print("\n" + "=" * 70)
print("PYTHON ML PACKAGES")
print("=" * 70)
print("numpy and lightgbm are needed in the IceTray environment too -- the\nmodel is loaded there.  sklearn / joblib / pandas are listed for information\nonly; nothing in this repo imports them:")
for n in ["numpy", "sklearn", "lightgbm", "joblib", "pandas"]:
    m = try_import(n)
    if m and hasattr(m, "__version__"):
        print(f"            version: {m.__version__}")
