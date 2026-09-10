'''
IceTray ortam koprusu -- "icetray import edilmiyor" hatasinin tek adresi.

Bu repodaki tum icetray'e bagimli moduller `icecube` paketini DOGRUDAN import
etmek yerine buradan gecer.  Neden:

  1. `import icecube` basarisiz oldugunda Python'un verdigi
     "ModuleNotFoundError: No module named 'icecube'" mesaji hicbir sey
     anlatmaz.  Burasi bunun yerine ortami inceleyip SEBEBI soyler
     (env-shell'e girilmemis / yanlis Jupyter kernel'i / build sourcelanmamis).

  2. `I3Tray` icetray surumune gore iki farkli yerde:
        icetray v1.5+           -> from icecube.icetray import I3Tray
        combo / v1.4 ve oncesi  -> from I3Tray import I3Tray
     Kendi derledigin bir build'de hangisi oldugu belli olmaz; burada ikisi de
     denenir.

  3. `tensor_of_inertia`, `fill_ratio`, `DeepCore_Filter` gibi projeler bir
     meta-projede olmayabilir.  Modul seviyesinde sert import edilirlerse TEK
     eksik proje tum repoyu import edilemez hale getirir.  Burada opsiyonel
     import + net hata mesaji var.

Kullanim:

    from icetray_env import require_icetray, get_I3Tray, optional_project

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

    a("I3_BUILD : %s" % (i3_build or "<bos>"))
    a("I3_SRC   : %s" % (i3_src or "<bos>"))
    a("I3_SHELL : %s" % (i3_shell or "<bos>"))
    a("SROOT    : %s" % (srv or "<bos>"))
    a("")

    # --- Tani
    if not srv and not i3_build:
        a("TANI: Ne cvmfs python ortami (SROOT) ne de bir icetray build")
        a("      (I3_BUILD) tanimli.  Hicbir ortam yuklenmemis.")
        a("")
        a("COZUM:")
        a("      ./setup_env.sh                 # ne oldugunu soyler")
        a("      ./setup_env.sh shell           # icetray shell'i acar")
    elif srv and not i3_build:
        a("TANI: cvmfs python ortami yuklu (setup.sh calistirilmis) ama")
        a("      icetray env-shell.sh CALISTIRILMAMIS.  setup.sh tek basina")
        a("      icecube paketini PYTHONPATH'e koymaz.")
        a("")
        a("COZUM: env-shell.sh de calistirin:")
        a("      $I3_BUILD/env-shell.sh          # kendi derlediginiz build")
        a("      ...ya da metaproject env-shell.sh'i")
        a("")
        a("      DIKKAT: env-shell.sh YENI BIR SHELL acar.  Bir script'in")
        a("      icinden ard arda yazarsaniz sonraki satirlar o shell'de")
        a("      CALISMAZ.  Tek komut icin:")
        a("          $I3_BUILD/env-shell.sh -- python process_L4.py ...")
    elif i3_build:
        a("TANI: I3_BUILD tanimli (%s) ama" % i3_build)
        a("      bu python icecube paketini bulamiyor.")
        a("")
        libdir = os.path.join(i3_build, "lib")
        if not os.path.isdir(libdir):
            a("      %s YOK -> build tamamlanmamis." % libdir)
            a("      COZUM: build dizininde `ninja` (veya `make`) calistirin.")
        elif not os.path.isdir(os.path.join(libdir, "icecube")):
            a("      %s/icecube YOK -> derleme yarim." % libdir)
            a("      COZUM: build dizininde `ninja` (veya `make`) calistirin.")
        elif libdir not in sys.path:
            a("      %s var ama sys.path'te DEGIL." % libdir)
            a("      TIPIK SEBEP: Jupyter kernel'i icetray ortami disinda")
            a("      baslatilmis.  Notebook'un kernel'i env-shell icinden")
            a("      kaydedilmis olmali:")
            a("          ./setup_env.sh kernel")
            a("      sonra notebook'ta Kernel > Change Kernel > 'IceTray'.")
        else:
            a("      lib dizini sys.path'te -- muhtemelen ABI/python surum")
            a("      uyusmazligi.  Build'i hangi cvmfs py3-vX ile derlediyseniz")
            a("      AYNISINI source edin.")

    a("")
    a("Detay icin:  python diagnose_env.py")
    a("=" * 72)
    return "\n".join(lines)


class IceTrayNotAvailable(ImportError):
    pass


# ---------------------------------------------------------------------------
# Zorunlu import
# ---------------------------------------------------------------------------

_icecube = None


def require_icetray():
    '''`icecube` paketini import et; olmazsa ACIKLAYICI bir hata firlat.'''
    global _icecube
    if _icecube is not None:
        return _icecube
    try:
        import icecube  # noqa: F401
    except ImportError as exc:
        raise IceTrayNotAvailable(
            "%s\n\nOrijinal hata: %s" % (_env_report(), exc)) from exc
    _icecube = icecube
    return icecube


def have_icetray():
    '''Sessiz kontrol -- notebook gibi icetray'siz ortamlar icin.'''
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
            "Orijinal hata: %s" % exc) from exc


# ---------------------------------------------------------------------------
# Opsiyonel projeler
# ---------------------------------------------------------------------------

# proje adi -> onu kullanan L4 degiskeni (hata mesajinda gosterilir)
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
    `icecube.<name>` import et.  Yoksa None dondur ve eksikler listesine ekle.

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


def missing_projects():
    '''optional_project() ile bulunamayanlarin listesi.'''
    return list(_missing)


def report_missing(stream=sys.stderr):
    '''Eksik projeleri ozetle.  process_L4.py basta bunu cagirir.'''
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
    print("  Bu projeler meta-projede yok ya da build'de derlenmemis.", file=stream)
    print("  Kendi build'inizde: src/ altinda var mi bakin, sonra yeniden derleyin.",
          file=stream)
    print("  Ayrinti: python diagnose_env.py", file=stream)
    print("", file=stream)
    return True


def require_project(name):
    '''optional_project gibi ama yoksa net bir hata firlatir.'''
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

    `slc-veto`, `static-twc` gibi kutuphaneler her build'de yok.
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
# (eskiden VICH icinde olay basina cagriliyordu -- gereksiz yavaslik).

_doms_cache = {}


def deepcore_doms(detector="IC86"):
    '''
    icecube.DeepCore_Filter.DOMS.DOMS(detector) -- cache'li.

    DeepCore_Filter projesi yoksa hata firlatir.  Bu listeler VICH ve
    micro_count'un TANIMI; tahmini bir liste ile doldurmak sessizce yanlis
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
    '''VICH icin veto OMKey seti -- olay basina degil, bir kez kurulur.'''
    key = ("veto", detector)
    if key not in _doms_cache:
        _doms_cache[key] = set(deepcore_doms(detector).DeepCoreVetoDOMs)
    return _doms_cache[key]


def deepcore_fiducial_domset(detector="IC86"):
    '''
    Fiducial OMKey seti -- VICH'in COG'u BUNLARLA sinirlanmali.

    Teknik not §3.4: "the center-of-gravity (COG) of the hits inside the
    FIDUCIAL VOLUME is calculated".  Bu, DeepCore Filter'in (L2) kendi
    fiducial/veto ayrimi; L3'un Tablo 7'deki daha genis fiducial tanimi
    DEGIL.
    '''
    key = ("fiducial", detector)
    if key not in _doms_cache:
        _doms_cache[key] = set(deepcore_doms(detector).DeepCoreFiducialDOMs)
    return _doms_cache[key]


# ---------------------------------------------------------------------------
# lightgbm
# ---------------------------------------------------------------------------
# Egitim ve uygulama tarafinin tek ML bagimliligi.  `icecube` isim alaninda
# DEGIL, siradan bir pip paketi -- IceTray ortaminda kurulu gelir.


def have_lightgbm():
    """lightgbm import edilebiliyor mu?"""
    try:
        importlib.import_module("lightgbm")
        return True
    except ImportError:
        return False


def require_lightgbm():
    """lightgbm'i import et; yoksa ne yapilmasi gerektigini soyle."""
    try:
        import lightgbm
        return lightgbm
    except ImportError as exc:
        raise ImportError(
            "lightgbm import edilemedi (%s).\n"
            "  L4 siniflandiricilari onunla egitiliyor ve uygulaniyor.\n"
            "  IceTray ortamindan kontrol:\n"
            "    ./setup_env.sh run python -c 'import lightgbm'\n"
            "  Yoksa:  pip install --user lightgbm\n" % exc)


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
    # /data/user/$USER en olasi yer: cobalt'ta home kotali oldugu icin
    # build oraya yapiliyor (bkz. README).
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
            print("  (yok)")
        for kind, path in cands:
            print("  [%-12s] %s" % (kind, path))
        sys.exit(1)
