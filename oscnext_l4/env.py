'''
IceTray environment bridge -- the single address for "icetray will not import".

Every icetray-dependent module in this repository goes through here instead of
importing the `icecube` package DIRECTLY.  Why:

  1. When `import icecube` fails, Python's
     "ModuleNotFoundError: No module named 'icecube'" says nothing useful.
     This module inspects the environment and reports the REASON instead
     (env-shell not entered / wrong Jupyter kernel / build not sourced).

  2. `I3Tray` lives in two different places depending on the icetray version:
        icetray v1.5+            -> from icecube.icetray import I3Tray
        combo / v1.4 and earlier -> from I3Tray import I3Tray
     In a self-compiled build you cannot tell which; both are tried here.

  3. Projects such as `tensor_of_inertia`, `fill_ratio` and `DeepCore_Filter`
     may be absent from a meta-project.  Imported hard at module level, ONE
     missing project makes the whole repository unimportable.  Here the import
     is optional and the error message is explicit.

Usage:

    from oscnext_l4.env import require_icetray, get_I3Tray, optional_project

    icecube = require_icetray()
    I3Tray  = get_I3Tray()
    toi     = optional_project("tensor_of_inertia")
'''

import os
import sys
import glob
import importlib


# ---------------------------------------------------------------------------
# Teshis
# ---------------------------------------------------------------------------

_CVMFS = "/cvmfs/icecube.opensciencegrid.org"


def _env_report():
    '''icecube import edilemediginde neden edilemedigini anlatan metin.'''
    lines = []
    a = lines.append

    a("=" * 72)
    a("ICETRAY IMPORT EDILEMEDI")
    a("=" * 72)
    a("")
    a("Python : %s" % sys.executable)
    a("Surum  : %s" % sys.version.split()[0])
    a("")

    i3_build = os.environ.get("I3_BUILD")
    i3_src = os.environ.get("I3_SRC")
    i3_shell = os.environ.get("I3_SHELL")
    srv = os.environ.get("SROOT")

    a("I3_BUILD : %s" % (i3_build or "<unset>"))
    a("I3_SRC   : %s" % (i3_src or "<unset>"))
    a("I3_SHELL : %s" % (i3_shell or "<unset>"))
    a("SROOT    : %s" % (srv or "<unset>"))
    a("")

    # --- Diagnosis
    if not srv and not i3_build:
        a("DIAGNOSIS: neither the cvmfs python environment (SROOT) nor an")
        a("      icetray build (I3_BUILD) is set.  No environment is loaded.")
        a("")
        a("REMEDY:")
        a("      ./setup_env.sh                 # says what it found")
        a("      ./setup_env.sh shell           # opens the icetray shell")
    elif srv and not i3_build:
        a("DIAGNOSIS: the cvmfs python environment is loaded (setup.sh was")
        a("      run) but the icetray env-shell.sh was NOT.  setup.sh alone")
        a("      does not put the icecube package on PYTHONPATH.")
        a("")
        a("REMEDY: run env-shell.sh as well:")
        a("      $I3_BUILD/env-shell.sh          # your own build")
        a("      ...or a metaproject env-shell.sh")
        a("")
        a("      CAREFUL: env-shell.sh opens a NEW SHELL.  Writing it and the")
        a("      next command on consecutive lines of a script means the")
        a("      later lines DO NOT run in it.  For a single command:")
        a("          $I3_BUILD/env-shell.sh -- python scripts/process_L4.py ...")
    elif i3_build:
        a("DIAGNOSIS: I3_BUILD is set (%s) but this" % i3_build)
        a("      python cannot find the icecube package.")
        a("")
        libdir = os.path.join(i3_build, "lib")
        if not os.path.isdir(libdir):
            a("      %s is MISSING -> the build never completed." % libdir)
            a("      REMEDY: run `ninja` (or `make`) in the build directory.")
        elif not os.path.isdir(os.path.join(libdir, "icecube")):
            a("      %s/icecube is MISSING -> the build is half done." % libdir)
            a("      REMEDY: run `ninja` (or `make`) in the build directory.")
        elif libdir not in sys.path:
            a("      %s exists but is NOT on sys.path." % libdir)
            a("      TYPICAL CAUSE: the Jupyter kernel was started outside the")
            a("      icetray environment.  The notebook's kernel has to be")
            a("      registered from inside env-shell:")
            a("          ./setup_env.sh kernel")
            a("      then in the notebook: Kernel > Change Kernel > 'IceTray'.")
        else:
            a("      the lib directory is on sys.path -- most likely an")
            a("      ABI/python version mismatch.  Source the SAME cvmfs")
            a("      py3-vX you compiled the build against.")

    a("")
    a("For details:  python scripts/diagnose_env.py")
    a("=" * 72)
    return "\n".join(lines)


class IceTrayNotAvailable(ImportError):
    pass


# ---------------------------------------------------------------------------
# Zorunlu import
# ---------------------------------------------------------------------------

_icecube = None


def require_icetray():
    '''Import the `icecube` package, or raise an EXPLANATORY error.'''
    global _icecube
    if _icecube is not None:
        return _icecube
    try:
        import icecube  # noqa: F401
    except ImportError as exc:
        raise IceTrayNotAvailable(
            "%s\n\nOriginal error: %s" % (_env_report(), exc)) from exc
    _icecube = icecube
    return icecube


def have_icetray():
    '''Quiet check -- for environments without icetray, such as a notebook.'''
    try:
        importlib.import_module("icecube")
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# I3Tray  --  surumden bagimsiz
# ---------------------------------------------------------------------------

def get_I3Tray():
    '''
    I3Tray sinifini dondur.

    icetray v1.5+   : icecube.icetray.I3Tray
    combo / oncesi  : top-level I3Tray modulu
    '''
    require_icetray()
    try:
        from icecube.icetray import I3Tray
        return I3Tray
    except ImportError:
        pass
    try:
        from I3Tray import I3Tray          # eski yerlesim
        return I3Tray
    except ImportError:
        pass
    try:
        from icecube.icetray.i3tray import I3Tray
        return I3Tray
    except ImportError as exc:
        raise IceTrayNotAvailable(
            "icecube paketi import edildi ama I3Tray bulunamadi.\n"
            "Denenen yerler: icecube.icetray.I3Tray, I3Tray.I3Tray,\n"
            "                icecube.icetray.i3tray.I3Tray\n"
            "Build eksik derlenmis olabilir: python diagnose_env.py\n"
            "Original error: %s" % exc) from exc


# ---------------------------------------------------------------------------
# Opsiyonel projeler
# ---------------------------------------------------------------------------

# project name -> the L4 variable that needs it (shown in the error message)
_PROJECT_PURPOSE = {
    "DomTools":          "I3OMSelection / I3TimeWindowCleaning (micro_count)",
    "STTools":           "SeededRT temizleme (micro_count)",
    "linefit":           "improved LineFit -> iLineFit_speed (noise BDT)",
    "tensor_of_inertia": "I3TensorOfInertia -> ToI evalratio (muon BDT)",
    "fill_ratio":        "I3FillRatioModule -> fill_ratio (noise BDT)",
    "common_variables":  "HitStatistics/HitMultiplicity -> cog_z, z_sigma, z_travel",
    "DeepCore_Filter":   "DeepCore fiducial/veto DOM listeleri (VICH, micro_count)",
}

_missing = []


def optional_project(name, required_for=None):
    '''
    Import `icecube.<name>`.  Return None and record it as missing if absent.

    Modul seviyesinde sert import yerine bunu kullanin: tek eksik proje tum
    repoyu import edilemez hale getirmesin.
    '''
    require_icetray()
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
    print("EKSIK ICETRAY PROJELERI -- bazi L4 degiskenleri URETILEMEYECEK",
          file=stream)
    print("!" * 72, file=stream)
    for name, purpose, exc in _missing:
        print("  icecube.%-20s %s" % (name, purpose), file=stream)
        print("      %s" % exc, file=stream)
    print("", file=stream)
    print("  These projects are absent from the meta-project or were not built.",
          file=stream)
    print("  Kendi build'inizde: src/ altinda var mi bakin, sonra yeniden derleyin.",
          file=stream)
    print("  Ayrinti: python diagnose_env.py", file=stream)
    print("", file=stream)
    return True


def require_project(name):
    '''Like optional_project, but raises an explicit error when absent.'''
    mod = optional_project(name)
    if mod is None:
        purpose = _PROJECT_PURPOSE.get(name, "")
        raise IceTrayNotAvailable(
            "icecube.%s bu ortamda YOK.\n"
            "Gerekli oldugu yer: %s\n"
            "Build'inizde bu proje derlenmemis.  src/%s var mi kontrol edip\n"
            "yeniden derleyin, ya da cvmfs metaproject'ini kullanin.\n"
            "Ayrinti: python diagnose_env.py" % (name, purpose, name))


def load_lib(libname, required=False):
    '''
    C++ modul kutuphanesi yukle (icetray.load).  Basarili ise True.

    Libraries such as `slc-veto` and `static-twc` are not in every build.
    '''
    require_icetray()
    from icecube import icetray
    try:
        icetray.load(libname, False)
        return True
    except Exception as exc:
        if required:
            raise IceTrayNotAvailable(
                "C++ kutuphanesi '%s' yuklenemedi: %s" % (libname, exc)) from exc
        return False


# ---------------------------------------------------------------------------
# DeepCore DOM listeleri  --  DeepCore_Filter yoksa
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


def find_env_shells():
    '''
    Sistemde bulunabilen env-shell.sh adaylarini dondur.

    Sirasiyla: I3_BUILD, yerel build dizinleri, cvmfs metaproject'leri.
    '''
    found = []

    i3_build = os.environ.get("I3_BUILD")
    if i3_build:
        p = os.path.join(i3_build, "env-shell.sh")
        if os.path.isfile(p):
            found.append(("I3_BUILD", p))

    home = os.path.expanduser("~")
    user = os.environ.get("USER") or os.path.basename(home)
    # /data/user/$USER is the likeliest place: home is quota-limited on
    # cobalt, so builds go there (see README).
    for pat in ("/data/user/%s/icetray_build/build/env-shell.sh" % user,
                "/data/user/%s/*/build/env-shell.sh" % user,
                "/data/user/%s/build/env-shell.sh" % user,
                "%s/*/build/env-shell.sh" % home,
                "%s/*/*/build/env-shell.sh" % home,
                "%s/icetray/build/env-shell.sh" % home,
                "%s/build/env-shell.sh" % home):
        for p in sorted(glob.glob(pat)):
            if p not in [f[1] for f in found]:
                found.append(("yerel build", p))

    for p in sorted(glob.glob(
            "%s/py3-v*/*/metaprojects/*/*/env-shell.sh" % _CVMFS)):
        found.append(("cvmfs", p))

    return found


if __name__ == "__main__":
    if have_icetray():
        print("icecube import edilebiliyor.")
        print("  python  :", sys.executable)
        import icecube
        print("  icecube :", os.path.dirname(icecube.__file__))
        try:
            print("  I3Tray  :", get_I3Tray())
        except Exception as e:
            print("  I3Tray  : BULUNAMADI --", e)
        if have_lightgbm():
            import lightgbm
            print("  lightgbm:", lightgbm.__version__)
        else:
            print("  lightgbm: YOK  -- egitim/uygulama calismaz")
    else:
        print(_env_report())
        print()
        print("Bulunan env-shell.sh adaylari:")
        cands = find_env_shells()
        if not cands:
            print("  (none)")
        for kind, path in cands:
            print("  [%-12s] %s" % (kind, path))
        sys.exit(1)
