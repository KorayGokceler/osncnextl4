#!/usr/bin/env python
'''
oscNext L4 isleme + HDF5 booking.

Kullanim:

  # MC (GENIE)
  python process_L4.py \
      --gcd  /data/GCD/GeoCalibDetectorStatus_2013.56429_V1.i3.gz \
      --input "/data/L3/genie/14640/*.i3.zst" \
      --output-i3   /data/L4/genie/14640/L4_14640.i3.zst \
      --output-hdf5 /data/L4/hdf5/genie/14640/L4_14640.hdf5 \
      --mc

  # Detektor verisi
  python process_L4.py \
      --gcd  /data/GCD/Run00125150_GCD.i3.zst \
      --input "/data/L3/data/Run00125150/*.i3.zst" \
      --output-hdf5 /data/L4/hdf5/data/Run00125150/L4_Run00125150.hdf5

Onemli: --no-cut varsayilandir.  Siniflandiricilari egitmeden once kesim
uygulamayin; tum olaylari book edin.
'''

import os
import re
import sys
import glob
import json
import time
import argparse

from icetray_env import (require_icetray, get_I3Tray, report_missing,
                         load_deserialization_libs, IceTrayNotAvailable)

try:
    require_icetray()
except IceTrayNotAvailable as _e:
    # Yigin izi degil, ne yapilmasi gerektigini goster.
    sys.exit("\n" + str(_e) + "\n")
from icecube import icetray, dataio, dataclasses
I3Tray = get_I3Tray()

# Frame nesnelerinin deserialize edilebilmesi icin gerekli.  Kodda dogrudan
# kullanilmasalar da import edilmeleri SART -- yoksa
# "Deserialization failed for object at frame key 'X'" hatasi alinir.
#   simclasses     -> I3MCPESeriesMap, I3MCPulseSeriesMap, noise_weight
#   recclasses     -> I3DST, PoleMuonLlhFitFitParams, ...
#   genie_icetray  -> I3GenieInfo, I3GenieResult   <-- n_flux_events icin
#   sim_services   -> I3MCPEShifter vb.
load_deserialization_libs()

from simple_booker import add_booker
from oscNext_L4_variables import (
    oscNext_L4, L4_HDF5_KEYS, HITSTAT_KEY, HITMULT_KEY,
    UNCLEANED_PULSES_DEFAULT, CLEANED_PULSES_DEFAULT,
)


# ---------------------------------------------------------------------------
# Book edilecek frame objeleri
# ---------------------------------------------------------------------------

# L3'ten gelen degiskenler -- muon BDT'nin 4 girdisi ve noise BDT'nin 2 girdisi
# bu map'in icinde.
L3_KEYS = [
    "IC2018_LE_L3_Vars",
    "IC2018_LE_L3_bools",
    "L3_oscNext_bool",      # IC2018_LE_L3_Full AND Data_quality_bool
    "Data_quality_bool",
]

# Hit statistics / multiplicity -- cog_z, z_sigma, z_travel, n_hit_doms
COMMON_VAR_KEYS = [HITSTAT_KEY, HITMULT_KEY]

# Her zaman lazim
BASE_KEYS = ["I3EventHeader"]

# Sadece simulasyon icin.  Not: MCInIcePrimary GURULTU MC'sinde YOKTUR --
# vuvuzela dosyalarini book ederken --noise bayragi ile bunlari cikarin.
# pass3 GENIE L3 ciktisinda MCInIcePrimary YOK -- truth bilgisi
# I3MCWeightDict icinde (PrimaryNeutrinoEnergy/Zenith/Type).
# NEvPerFile agirlik normalizasyonu icin onemli.
MC_KEYS = [
    "I3MCWeightDict",
    "NEvPerFile",
    "I3GenieSystWeightDict",
    # asagidakiler varsa book edilir, yoksa sessizce atlanir
    "MCInIcePrimary",
    "MCDeepCoreStartingEvent",
    "MCExtraTruthInfo",
]

# Vuvuzela: agirlik "noise_weight" icinde.
# BIRIM UYARISI: pass3'te 1/ns (x1e9 gerekir), pass2'de zaten Hz.
NOISE_MC_KEYS = [
    "I3MCWeightDict",
    "noise_weight",
]

# MuonGun agirlik adaylari -- hangisi varsa book edilir
MUONGUN_KEYS = [
    "I3MCWeightDict",
    "MuonWeight",
    "MuonWeight_GaisserH4a",
    "MuonGunWeight",
]

# CORSIKA: agirlik simweights ile hesaplanir.
# simweights CorsikaWeightMap + PolyplopiaPrimary bekler; ikisi de
# book edilmeli yoksa CorsikaWeighter kurulamaz.
CORSIKA_KEYS = [
    "CorsikaWeightMap",
    "PolyplopiaPrimary",
    "PolyplopiaInfo",
    "I3PrimaryInjectorInfo",
    "I3CorsikaInfo",
]


# ---------------------------------------------------------------------------
# Bozuk girdi dosyalari
# ---------------------------------------------------------------------------
#
# pass3 uretiminde ara sira yarim yazilmis / bozuk .i3.zst dosyalari var.
# I3Reader boyle bir dosyaya gelince
#
#     FATAL (I3Reader): Error reading <dosya> at frame N: input stream error!
#
# atip TUM tray'i oldururuyor.  100 dosyalik bir job'da tek bozuk dosya
# yuzunden 99 saglam dosyanin islenmesi bosa gidiyor ve geride yarim,
# acilmayan bir HDF5 kaliyor ("Table ... is still connected" hatasi bunun
# sonucu, ayri bir bug degil).
#
# Cozum iki katmanli:
#   1) On tarama  -- her dosyanin ilk N frame'i okunur, acilmayanlar elenir.
#      Kesik/bos dosyalari saniyeler icinde yakalar (tipik hata frame 4'te).
#   2) Calisma ani -- tray yine de patlarsa hata mesajindan dosya adi
#      cikarilir, kara listeye yazilir ve o dosya haric yeniden denenir.

_BAD_FILE_RE = re.compile(r"Error reading (\S+?) at frame")


# ---------------------------------------------------------------------------
# Ilerleme raporu
# ---------------------------------------------------------------------------
#
# Notebook process_L4.py'yi subprocess olarak calistiriyor.  Ilerleme
# gorunsun diye stdout'a MAKINE OKUNUR satirlar basiyoruz:
#
#   [CHUNK] 3/10 files=30/100 booked=1840 elapsed=312.4
#   [PROGRESS] frames=15000 physics=7100 booked=4300 elapsed=98.2 rate=152.7
#
# Notebook bu satirlari ayristirip bar cizer.  Terminalde de okunabilir.
# stdout satir-tamponlu olmali, yoksa notebook bitene kadar hicbir sey gormez.


def _emit(line):
    """Ilerleme satiri bas ve HEMEN flush et (subprocess tamponuna takilmasin)."""
    print(line, flush=True)


class ProgressReporter:
    """Her N frame'de bir [PROGRESS] satiri basan tray modulu (fonksiyon)."""

    def __init__(self, every, counter, t0):
        self.every = every
        self.counter = counter
        self.t0 = t0
        self.n = 0

    def __call__(self, frame):
        self.n += 1
        if self.every and self.n % self.every == 0:
            dt = time.time() - self.t0
            _emit("[PROGRESS] frames=%d physics=%d booked=%d elapsed=%.1f rate=%.1f"
                  % (self.n, self.counter["physics"], self.counter["n"], dt,
                     self.n / dt if dt > 0 else 0.0))
        return True


def _part_path(path, index):
    """L4_nue.hdf5 -> L4_nue_part003.hdf5  (notebook glob'u L4_nue*.hdf5 ile eslesir)"""
    if not path:
        return None
    base, ext = os.path.splitext(path)
    return "%s_part%03d%s" % (base, index, ext)


def _write_meta(output, meta):
    """
    <cikti>.meta.json yaz.

    EN ONEMLI ALAN: n_l3_files -- bu HDF5'in KAC L3 dosyasindan uretildigi.
    Agirlik normalizasyonu (OneWeight/n_flux/n_files) bu sayiya bolunmeli;
    HDF5 dosya sayisina DEGIL.  Notebook bunu sidecar'dan okuyor.
    """
    if not output:
        return
    path = output + ".meta.json"
    try:
        with open(path, "w") as fh:
            json.dump(meta, fh, indent=2)
    except OSError as e:
        print("  [!] meta yazilamadi: %s" % e)


def _pct(n, total):
    return "%.1f%%" % (100.0 * n / total) if total else "-"


def _bad_list_path(output):
    return (output or "process_L4") + ".badfiles.txt"


def _record_bad(output, paths, reason):
    """Bozuk dosyalari <cikti>.badfiles.txt icine yaz."""
    if not paths:
        return
    path = _bad_list_path(output)
    try:
        with open(path, "a") as fh:
            for p in paths:
                fh.write("%s\t%s\n" % (p, reason))
        print("  Kara liste: %s" % path)
    except OSError as e:
        print("  [!] kara liste yazilamadi: %s" % e)


def validate_files(paths, n_frames=25, verbose=True):
    """
    Her dosyayi acip ilk `n_frames` frame'i oku.  (saglam, bozuk) dondur.

    n_frames=0 -> dosyanin TAMAMI okunur (yavas ama kesin).

    Not: bu, dosyanin tamamen saglam oldugunu garanti etmez -- ortasinda
    bozulma varsa ancak tam tarama yakalar.  Ama pratikte gordugumuz
    hatalar (kesik yazilmis dosya) ilk frame'lerde ortaya cikiyor.
    """
    good, bad = [], []
    for i, path in enumerate(paths):
        try:
            if os.path.getsize(path) == 0:
                bad.append((path, "bos dosya (0 byte)"))
                continue
        except OSError as e:
            bad.append((path, "stat: %s" % e))
            continue
        try:
            f = dataio.I3File(path)
            try:
                n = 0
                while f.more():
                    f.pop_frame()
                    n += 1
                    if n_frames and n >= n_frames:
                        break
            finally:
                f.close()
            good.append(path)
        except Exception as e:
            first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            bad.append((path, first[:160]))
        if verbose and (i + 1) % 100 == 0:
            print("  taranan: %d/%d" % (i + 1, len(paths)))
    return good, bad


def _run_tray(build, infiles, output_hdf5, retries):
    """
    Tray'i calistir.  Bozuk dosya yuzunden patlarsa o dosyayi cikarip
    yeniden dene (en fazla `retries` kez).
    """
    attempt = 0
    while True:
        tray, counter = build(infiles)
        n_frames = getattr(build, "n_frames", 0)
        try:
            if n_frames:
                tray.Execute(n_frames)
            else:
                tray.Execute()
            return counter, infiles
        except RuntimeError as e:
            m = _BAD_FILE_RE.search(str(e))
            if not m or attempt >= retries:
                raise
            bad = m.group(1)
            if bad not in infiles:
                raise
            attempt += 1
            print("\n[!] Bozuk dosya calisma aninda yakalandi:\n    %s" % bad)
            print("    %s" % str(e).strip().splitlines()[0][:200])
            _record_bad(output_hdf5, [bad], "calisma ani: input stream error")
            infiles = [f for f in infiles if f != bad]
            # Yarim kalan HDF5 kullanilamaz -- silinmezse yeniden acilamaz.
            if output_hdf5 and os.path.exists(output_hdf5):
                try:
                    os.remove(output_hdf5)
                    print("    Yarim HDF5 silindi, bastan basliyor.")
                except OSError as rm:
                    print("    [!] yarim HDF5 silinemedi: %s" % rm)
            print("    Kalan dosya: %d  (deneme %d/%d)\n"
                  % (len(infiles), attempt, retries))
            if not infiles:
                raise RuntimeError("Tum girdi dosyalari bozuk cikti.")


def build_key_list(is_mc=False, is_noise=False, is_muongun=False,
                   is_corsika=False, extra=None):
    keys = list(BASE_KEYS) + list(L3_KEYS) + list(COMMON_VAR_KEYS) + list(L4_HDF5_KEYS)
    if is_noise:
        keys += NOISE_MC_KEYS
    elif is_corsika:
        keys += CORSIKA_KEYS
    elif is_muongun:
        keys += MUONGUN_KEYS
    elif is_mc:
        keys += MC_KEYS
    if extra:
        keys += list(extra)
    # sirayi koruyarak tekrarlari at
    return list(dict.fromkeys(keys))


def main():
    p = argparse.ArgumentParser(description="oscNext L4 isleme + HDF5 booking")
    p.add_argument("--gcd", required=True, help="GCD dosyasi")
    p.add_argument("--input", nargs="+", default=[],
                   help="Girdi .i3 dosyalari (glob deseni de olur)")
    p.add_argument("--output-i3", default=None, help="Cikti .i3 (opsiyonel)")
    p.add_argument("--output-hdf5", required=True, help="Cikti .hdf5")

    p.add_argument("--uncleaned-pulses", default=UNCLEANED_PULSES_DEFAULT)
    p.add_argument("--cleaned-pulses", default=CLEANED_PULSES_DEFAULT)
    p.add_argument("--sub-event-stream", default="InIceSplit")

    p.add_argument("--mc", action="store_true", help="Simulasyon (MC truth book et)")
    p.add_argument("--noise", action="store_true",
                   help="Saf gurultu MC (noise_weight book edilir)")
    p.add_argument("--muongun", action="store_true",
                   help="MuonGun MC (muon agirlik anahtarlari book edilir)")
    p.add_argument("--genie", action="store_true",
                   help="GENIE MC -- I3GenieInfo.n_flux_events her frame'e tasinir")
    p.add_argument("--corsika", action="store_true",
                   help="CORSIKA MC -- CorsikaWeightMap + PolyplopiaPrimary book edilir")

    p.add_argument("--no-l3-cut", action="store_true",
                   help="L3 kesimini uygulama (girdi zaten L3 gecmisse)")
    p.add_argument("--apply-cut", action="store_true",
                   help="L4 siniflandirici kesimini uygula (modeller egitilmis olmali)")
    p.add_argument("--model-dir", default=None, help="Egitilmis .joblib modellerin dizini")

    p.add_argument("--n", type=int, default=0, help="Islenecek frame sayisi (0=hepsi)")

    p.add_argument("--input-list", default=None,
                   help="Girdi dosyalarini bu metin dosyasindan oku (satir basina "
                        "bir yol).  scan_files.py --good-list ciktisi ile kullanilir; "
                        "boylece tarama bir kez yapilir, her job tekrar etmez.")
    p.add_argument("--scan", choices=["quick", "full", "off"], default="quick",
                   help="Girdi dosyalarini on tarama: quick=ilk frame'ler "
                        "(varsayilan, kesik dosyalari yakalar), full=tum dosya "
                        "(yavas ama kesin), off=tarama yok")
    p.add_argument("--scan-frames", type=int, default=25,
                   help="--scan quick modunda dosya basina okunacak frame (varsayilan 25)")
    p.add_argument("--progress", type=int, default=5000,
                   help="Her N frame'de bir [PROGRESS] satiri bas (0=kapali). "
                        "Notebook bu satirlardan ilerleme cubugu cizer.")
    p.add_argument("--chunk-files", type=int, default=0,
                   help="Girdiyi N dosyalik parcalar halinde isle; her parca "
                        "ayri bir tray ve ayri bir <cikti>_partNNN.hdf5 uretir. "
                        "Faydasi: GERCEK yuzde/ETA, ve cokme halinde sadece o "
                        "parca kaybolur (tamamlanan parcalar atlanir).")
    p.add_argument("--overwrite", action="store_true",
                   help="--chunk-files ile: var olan parcalari da yeniden uret")
    p.add_argument("--retries", type=int, default=3,
                   help="Calisma aninda bozuk dosya cikarsa onu atip kac kez "
                        "yeniden denensin (varsayilan 3, 0=deneme)")
    p.add_argument("--no-hit-statistics", action="store_true",
                   help="HitStatistics'i hesaplama (L3'te zaten varsa)")
    args = p.parse_args()

    if not args.input and not args.input_list:
        p.error("--input ya da --input-list vermelisiniz.")

    if args.n > 0 and args.chunk_files > 0:
        p.error("--n ile --chunk-files birlikte kullanilmaz: --n her parcada "
                "ayri ayri uygulanir ve anlamsiz sonuc verir.  Smoke test icin "
                "sadece --n, uretim icin sadece --chunk-files.")

    # --- girdi dosyalarini coz ---
    infiles = []
    if args.input_list:
        try:
            with open(args.input_list) as fh:
                infiles = [ln.strip() for ln in fh
                           if ln.strip() and not ln.startswith("#")]
        except OSError as e:
            sys.exit("--input-list okunamadi: %s" % e)
        print("Girdi listesi: %s (%d dosya)" % (args.input_list, len(infiles)))
    for pattern in args.input:
        matched = sorted(glob.glob(pattern))
        infiles.extend(matched if matched else [pattern])
    if not infiles:
        sys.exit("Girdi dosyasi bulunamadi.")
    print("Girdi dosyasi:", len(infiles))

    # --- bozuk dosyalari on taramayla ele ---
    if args.scan != "off":
        nf = 0 if args.scan == "full" else args.scan_frames
        if args.n > 0 and len(infiles) > 5:
            print("On tarama (%s)... [%d dosya]" % (args.scan, len(infiles)))
            print("  NOT: --n verildigi icin tray ilk dosyalarda duracak;")
            print("       tum listeyi taramak bosuna zaman.  Smoke test'te")
            print("       tek dosya verin ya da --scan off kullanin.")
        else:
            print("On tarama (%s)..." % args.scan)
        infiles, bad = validate_files(infiles, n_frames=nf)
        if bad:
            print("  [!] %d bozuk dosya elendi:" % len(bad))
            for path, why in bad[:10]:
                print("      %s\n          %s" % (os.path.basename(path), why))
            if len(bad) > 10:
                print("      ... (+%d tane daha)" % (len(bad) - 10))
            _record_bad(args.output_hdf5, [b[0] for b in bad], "on tarama: " + args.scan)
        print("  Islenecek dosya: %d" % len(infiles))
        if not infiles:
            sys.exit("Saglam girdi dosyasi kalmadi.")

    for out in (args.output_i3, args.output_hdf5):
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    keys = build_key_list(is_mc=args.mc, is_noise=args.noise,
                          is_muongun=args.muongun, is_corsika=args.corsika)
    print("Book edilecek anahtar:", len(keys))

    # --- tray ---
    # Tray'i bir fabrika fonksiyonu icinde kuruyoruz: calisma aninda bozuk
    # dosya cikarsa o dosya haric YENIDEN kurulup calistirilabilsin diye
    # (bir I3Tray ikinci kez Execute edilemez).
    def build_tray(files):
        tray = I3Tray()
        tray.Add("I3Reader", "reader", FilenameList=[args.gcd] + files)

        # --- asamali sayaclar --------------------------------------------
        # Olay kaybinin NEREDE oldugunu gormek icin.  "--n ile 200 frame
        # verdim ama 60 olay cikti" sorusunun cevabi bu dokumde.
        counter = {"physics": 0, "stream": 0, "n": 0}

        def _count_physics(frame):
            counter["physics"] += 1
            return True
        tray.Add(_count_physics, "count_physics",
                 Streams=[icetray.I3Frame.Physics])

        # Ilerleme: TUM frame'lerde sayar (Q/P/G/C/D), her --progress frame'de
        # bir satir basar.  Bu, notebook'un bar'ini besleyen kaynak.
        if args.progress:
            tray.Add(ProgressReporter(args.progress, counter, time.time()),
                     "progress")

        # Sadece fizik sub-event stream'ini isle
        tray.Add(lambda f: f["I3EventHeader"].sub_event_stream == args.sub_event_stream,
                 "stream_filter",
                 Streams=[icetray.I3Frame.Physics])

        def _count_stream(frame):
            counter["stream"] += 1
            return True
        tray.Add(_count_stream, "count_stream",
                 Streams=[icetray.I3Frame.Physics])

        tray.Add(oscNext_L4, "oscNext_L4",
                 uncleaned_pulses=args.uncleaned_pulses,
                 cleaned_pulses=args.cleaned_pulses,
                 apply_l3_cut=not args.no_l3_cut,
                 is_genie=args.genie,
                 compute_hit_statistics=not args.no_hit_statistics,
                 apply_cut=args.apply_cut,
                 classifier_model_dir=args.model_dir)

        # --- L3 kesiminden sonra kalan (book edilecek) ---
        def count(frame):
            counter["n"] += 1
            return True
        tray.Add(count, "counter")

        out_i3 = getattr(build_tray, "output_i3", args.output_i3)
        if out_i3:
            tray.Add("I3Writer", "writer",
                     Filename=out_i3,
                     Streams=[icetray.I3Frame.TrayInfo,
                              icetray.I3Frame.DAQ,
                              icetray.I3Frame.Physics,
                              icetray.I3Frame.Stream("S"),
                              icetray.I3Frame.Stream("M")])

        add_booker(tray, "booker",
                   output=getattr(build_tray, "output_hdf5", args.output_hdf5),
                   keys=keys,
                   sub_event_streams=[args.sub_event_stream])
        return tray, counter

    build_tray.n_frames = args.n if args.n > 0 else 0

    # -----------------------------------------------------------------------
    # Calistir -- tek parca ya da chunk'li
    # -----------------------------------------------------------------------
    t_start = time.time()
    totals = {"physics": 0, "stream": 0, "n": 0}
    used = []

    if args.chunk_files and args.chunk_files > 0 and args.output_hdf5:
        chunks = [infiles[i:i + args.chunk_files]
                  for i in range(0, len(infiles), args.chunk_files)]
        n_chunks = len(chunks)
        print("Parca sayisi: %d  (%d dosya/parca)" % (n_chunks, args.chunk_files))
        _emit("[CHUNK] 0/%d files=0/%d booked=0 elapsed=0.0"
              % (n_chunks, len(infiles)))

        n_done_files = 0
        for ci, chunk in enumerate(chunks):
            out_part = _part_path(args.output_hdf5, ci)
            n_done_files += len(chunk)

            # Tamamlanmis parcayi atla -> cokme sonrasi kaldigi yerden devam
            if os.path.exists(out_part) and not args.overwrite:
                print("[%d/%d] atlandi (zaten var): %s"
                      % (ci + 1, n_chunks, os.path.basename(out_part)))
                _emit("[CHUNK] %d/%d files=%d/%d booked=%d elapsed=%.1f"
                      % (ci + 1, n_chunks, n_done_files, len(infiles),
                         totals["n"], time.time() - t_start))
                continue

            build_tray.output_hdf5 = out_part
            build_tray.output_i3 = _part_path(args.output_i3, ci)
            counts, chunk_used = _run_tray(build_tray, chunk, out_part, args.retries)
            for k in totals:
                totals[k] += counts[k]
            used.extend(chunk_used)

            _write_meta(out_part, dict(
                n_l3_files=len(chunk_used),
                n_l3_files_given=len(chunk),
                physics_frames=counts["physics"],
                sub_event_stream=args.sub_event_stream,
                after_stream_filter=counts["stream"],
                booked=counts["n"],
                chunk_index=ci, n_chunks=n_chunks,
                elapsed_s=round(time.time() - t_start, 1)))

            _emit("[CHUNK] %d/%d files=%d/%d booked=%d elapsed=%.1f"
                  % (ci + 1, n_chunks, n_done_files, len(infiles),
                     totals["n"], time.time() - t_start))
    else:
        build_tray.output_hdf5 = args.output_hdf5
        build_tray.output_i3 = args.output_i3
        totals, used = _run_tray(build_tray, infiles, args.output_hdf5, args.retries)
        _write_meta(args.output_hdf5, dict(
            n_l3_files=len(used),
            n_l3_files_given=len(infiles),
            physics_frames=totals["physics"],
            sub_event_stream=args.sub_event_stream,
            after_stream_filter=totals["stream"],
            booked=totals["n"],
            n_frames_limit=args.n,
            elapsed_s=round(time.time() - t_start, 1)))

    counts = totals
    n_phys, n_stream, n_booked = counts["physics"], counts["stream"], counts["n"]

    print()
    print("Physics frame           : %d" % n_phys)
    print("  %-22s: %d  (%s)" % (args.sub_event_stream, n_stream,
                                 _pct(n_stream, n_phys)))
    print("  L3 kesimi sonrasi     : %d  (%s)" % (n_booked, _pct(n_booked, n_stream)))
    print("Book edilen olay        : %d" % n_booked)

    if args.n > 0:
        # --n frame sayisidir, olay sayisi DEGIL.  Tray erken durdugu icin
        # dosya listesinin tamami okunmamis olabilir.
        print()
        print("NOT: --n %d = %d FRAME (olay degil).  Tray bu sayida frame"
              % (args.n, args.n))
        print("     okuyunca durdu; dosya listesinin tamami okunmamis olabilir.")
        print("     Dosya listesi: %d dosya (kac tanesinin okundugu belli degil)."
              % len(used))
    else:
        print("Islenen dosya           : %d" % len(used))

    if args.chunk_files > 0 and args.output_hdf5:
        print("HDF5 parcalari          : %s"
              % _part_path(args.output_hdf5, 0).replace("_part000", "_partNNN"))
    else:
        print("HDF5:", args.output_hdf5)
    bl = _bad_list_path(args.output_hdf5)
    if os.path.exists(bl):
        print("Bozuk dosya listesi:", bl)

    if n_phys and not n_booked:
        print()
        print("[!] Hic olay book EDILMEDI.  Sirayla kontrol edin:")
        if not n_stream:
            print("    * --sub-event-stream '%s' yanlis olabilir -- hicbir"
                  % args.sub_event_stream)
            print("      Physics frame bu stream'de degil.")
        else:
            print("    * L3 kesimi her seyi eledi -- girdi gercekten L3 ciktisi mi?")
            print("      Test icin: --no-l3-cut")


if __name__ == "__main__":
    main()
