'''
IceTray environment bridge -- what this meta-project does and does not carry.

`icecube` itself is always there: this pipeline runs inside an icetray
environment, so its modules import `from icecube import ...` directly and a
failure there is a broken environment, not a case to handle.  What this module
is for is everything ABOVE that line:

  1. `I3Tray` lives in two different places depending on the icetray version:
        icetray v1.5+            -> from icecube.icetray import I3Tray
        combo / v1.4 and earlier -> from I3Tray import I3Tray
     In a self-compiled build you cannot tell which; both are tried here.

  2. Individual PROJECTS really can be absent -- `oscNext`, `tau_bdt`,
     `slc-veto`, `tensor_of_inertia`, `fill_ratio`, `DeepCore_Filter`.  That
     is the premise this repository was written on.  Imported hard at module
     level, ONE missing project makes the whole repository unimportable, so
     the import is optional here and the error names what is missing and what
     it was needed for.

Usage:

    from oscnext_l4.env import get_I3Tray, optional_project

    I3Tray = get_I3Tray()
    toi    = optional_project("tensor_of_inertia")
'''

import os
import sys
import glob
import importlib



# ---------------------------------------------------------------------------
# I3Tray  --  independent of the icetray version
# ---------------------------------------------------------------------------

def get_I3Tray():
    '''
    Return the I3Tray class.

    icetray v1.5+   : icecube.icetray.I3Tray
    combo / earlier : the top-level I3Tray module
    '''
    try:
        from icecube.icetray import I3Tray
        return I3Tray
    except ImportError:
        pass
    try:
        from I3Tray import I3Tray          # the old location
        return I3Tray
    except ImportError:
        pass
    try:
        from icecube.icetray.i3tray import I3Tray
        return I3Tray
    except ImportError as exc:
        raise ImportError(
            "the icecube package imported but I3Tray was not found.\n"
            "Tried: icecube.icetray.I3Tray, I3Tray.I3Tray,\n"
            "                icecube.icetray.i3tray.I3Tray\n"
            "The build may be incomplete: python scripts/diagnose_env.py\n"
            "Original error: %s" % exc) from exc


# ---------------------------------------------------------------------------
# Opsiyonel projeler
# ---------------------------------------------------------------------------

# project name -> the L4 variable that needs it (shown in the error message)
_PROJECT_PURPOSE = {
    "DomTools":          "I3OMSelection / I3TimeWindowCleaning (micro_count)",
    "STTools":           "SeededRT cleaning (micro_count)",
    "linefit":           "improved LineFit -> iLineFit_speed (noise BDT)",
    "tensor_of_inertia": "I3TensorOfInertia -> ToI evalratio (muon BDT)",
    "fill_ratio":        "I3FillRatioModule -> fill_ratio (noise BDT)",
    "common_variables":  "HitStatistics/HitMultiplicity -> cog_z, z_sigma, z_travel",
    "DeepCore_Filter":   "the DeepCore fiducial/veto DOM lists (micro_count)",
}

_missing = []


def optional_project(name, required_for=None):
    '''
    Import `icecube.<name>`.  Return None and record it as missing if absent.

    Use this instead of a hard import at module level, so that one missing
    project does not make the whole repository unimportable.
    '''
    try:
        return importlib.import_module("icecube." + name)
    except ImportError as exc:
        purpose = required_for or _PROJECT_PURPOSE.get(name, "")
        _missing.append((name, purpose, str(exc)))
        return None


def report_missing(stream=sys.stderr):
    '''Summarise the missing projects.  process_L4.py calls this at startup.'''
    if not _missing:
        return False
    print("", file=stream)
    print("!" * 72, file=stream)
    print("MISSING ICETRAY PROJECTS -- some L4 variables CANNOT BE PRODUCED",
          file=stream)
    print("!" * 72, file=stream)
    for name, purpose, exc in _missing:
        print("  icecube.%-20s %s" % (name, purpose), file=stream)
        print("      %s" % exc, file=stream)
    print("", file=stream)
    print("  These projects are absent from the meta-project or were not built.",
          file=stream)
    print("  In your own build: check whether src/<name> exists, then rebuild.",
          file=stream)
    print("  Ayrinti: python diagnose_env.py", file=stream)
    print("", file=stream)
    return True


def require_project(name):
    '''Like optional_project, but raises an explicit error when absent.'''
    mod = optional_project(name)
    if mod is None:
        purpose = _PROJECT_PURPOSE.get(name, "")
        raise ImportError(
            "icecube.%s is NOT in this environment.\n"
            "Needed for: %s\n"
            "The project was not built here.  Check whether src/%s exists and\n"
            "rebuild, or use a cvmfs metaproject.\n"
            "Details: python scripts/diagnose_env.py" % (name, purpose, name))


def load_lib(libname, required=False):
    '''
    Load a C++ module library (icetray.load).  True when it succeeded.

    Libraries such as `slc-veto` and `static-twc` are not in every build.
    '''
    from icecube import icetray
    try:
        icetray.load(libname, False)
        return True
    except Exception as exc:
        if required:
            raise ImportError(
                "the C++ library '%s' could not be loaded: %s" % (libname, exc)) from exc
        return False


# ---------------------------------------------------------------------------
# DeepCore DOM lists  --  when DeepCore_Filter is absent
# ---------------------------------------------------------------------------
#
# DOMS.DOMS("IC86") her frame'de yeniden kurulmasin diye burada cache'lenir
# (it used to be called per event inside VICH -- pointless slowdown).

_doms_cache = {}


def deepcore_doms(detector="IC86"):
    '''
    icecube.DeepCore_Filter.DOMS.DOMS(detector) -- cache'li.

    Raises when the DeepCore_Filter project is absent.  These lists are the
    DEFINITION of VICH and micro_count; filling them with a guess would give
    fizik uretir, o yuzden fallback YOK.
    '''
    if detector in _doms_cache:
        return _doms_cache[detector]
    require_project("DeepCore_Filter")
    from icecube.DeepCore_Filter import DOMS
    obj = DOMS.DOMS(detector)
    _doms_cache[detector] = obj
    return obj


def deepcore_veto_domset(detector="IC86"):
    '''The veto OMKey set for VICH -- built once, not per event.'''
    key = ("veto", detector)
    if key not in _doms_cache:
        _doms_cache[key] = set(deepcore_doms(detector).DeepCoreVetoDOMs)
    return _doms_cache[key]


def deepcore_fiducial_domset(detector="IC86"):
    '''
    The fiducial OMKey set -- VICH's COG must be restricted to THESE.

    Technical note sec. 3.4: "the center-of-gravity (COG) of the hits inside
    the FIDUCIAL VOLUME is calculated".  That is the DeepCore Filter's (L2)
    own fiducial/veto split, NOT L3's wider fiducial definition from Table 7.
    '''
    key = ("fiducial", detector)
    if key not in _doms_cache:
        _doms_cache[key] = set(deepcore_doms(detector).DeepCoreFiducialDOMs)
    return _doms_cache[key]


# ---------------------------------------------------------------------------
# lightgbm
# ---------------------------------------------------------------------------
# The only ML dependency of both training and application.  NOT in the
# `icecube` namespace -- an ordinary pip package, present in IceTray.


def have_lightgbm():
    """lightgbm import edilebiliyor mu?"""
    try:
        importlib.import_module("lightgbm")
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Libraries needed for deserialisation
# ---------------------------------------------------------------------------

# They are never used directly in the code, but importing them is MANDATORY --
# without it you get "Deserialization failed for object at frame key 'X'".
_DESERIALIZE_LIBS = ("simclasses", "recclasses", "genie_icetray",
                     "genie_reader", "sim_services", "phys_services")


def load_deserialization_libs():
    '''Import the projects a frame object needs in order to be unpacked.'''
    loaded = []
    for lib in _DESERIALIZE_LIBS:
        try:
            importlib.import_module("icecube." + lib)
            loaded.append(lib)
        except ImportError:
            pass
    return loaded


if __name__ == "__main__":
    import icecube
    print("python  :", sys.executable)
    print("icecube :", os.path.dirname(icecube.__file__))
    try:
        print("I3Tray  :", get_I3Tray())
    except Exception as e:
        print("I3Tray  : NOT FOUND --", e)
    if have_lightgbm():
        import lightgbm
        print("lightgbm:", lightgbm.__version__)
    else:
        print("lightgbm: MISSING  -- training/application will not work")
