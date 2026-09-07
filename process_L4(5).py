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
import sys
import glob
import argparse

from icecube import icetray, dataio, dataclasses
from icecube.icetray import I3Tray

# Frame nesnelerinin deserialize edilebilmesi icin gerekli.  Kodda dogrudan
# kullanilmasalar da import edilmeleri SART -- yoksa
# "Deserialization failed for object at frame key 'X'" hatasi alinir.
#   simclasses     -> I3MCPESeriesMap, I3MCPulseSeriesMap, noise_weight
#   recclasses     -> I3DST, PoleMuonLlhFitFitParams, ...
#   genie_icetray  -> I3GenieInfo, I3GenieResult   <-- n_flux_events icin
#   sim_services   -> I3MCPEShifter vb.
for _lib in ("simclasses", "recclasses", "genie_icetray", "genie_reader",
             "sim_services", "phys_services"):
    try:
        __import__(f"icecube.{_lib}")
    except ImportError:
        pass

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
    p.add_argument("--input", required=True, nargs="+",
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
    p.add_argument("--no-hit-statistics", action="store_true",
                   help="HitStatistics'i hesaplama (L3'te zaten varsa)")
    args = p.parse_args()

    # --- girdi dosyalarini coz ---
    infiles = []
    for pattern in args.input:
        matched = sorted(glob.glob(pattern))
        infiles.extend(matched if matched else [pattern])
    if not infiles:
        sys.exit("Girdi dosyasi bulunamadi.")
    print("Girdi dosyasi:", len(infiles))

    for out in (args.output_i3, args.output_hdf5):
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    keys = build_key_list(is_mc=args.mc, is_noise=args.noise,
                          is_muongun=args.muongun, is_corsika=args.corsika)
    print("Book edilecek anahtar:", len(keys))

    # --- tray ---
    tray = I3Tray()
    tray.Add("I3Reader", "reader", FilenameList=[args.gcd] + infiles)

    # Sadece fizik sub-event stream'ini isle
    tray.Add(lambda f: f["I3EventHeader"].sub_event_stream == args.sub_event_stream,
             "stream_filter",
             Streams=[icetray.I3Frame.Physics])

    tray.Add(oscNext_L4, "oscNext_L4",
             uncleaned_pulses=args.uncleaned_pulses,
             cleaned_pulses=args.cleaned_pulses,
             apply_l3_cut=not args.no_l3_cut,
             is_genie=args.genie,
             compute_hit_statistics=not args.no_hit_statistics,
             apply_cut=args.apply_cut,
             classifier_model_dir=args.model_dir)

    # --- ne kadar olay kaldi? ---
    counter = {"n": 0}
    def count(frame):
        counter["n"] += 1
        return True
    tray.Add(count, "counter")

    if args.output_i3:
        tray.Add("I3Writer", "writer",
                 Filename=args.output_i3,
                 Streams=[icetray.I3Frame.TrayInfo,
                          icetray.I3Frame.DAQ,
                          icetray.I3Frame.Physics,
                          icetray.I3Frame.Stream("S"),
                          icetray.I3Frame.Stream("M")])

    add_booker(tray, "booker",
               output=args.output_hdf5,
               keys=keys,
               sub_event_streams=[args.sub_event_stream])

    if args.n > 0:
        tray.Execute(args.n)
    else:
        tray.Execute()

    print("Book edilen olay:", counter["n"])
    print("HDF5:", args.output_hdf5)


if __name__ == "__main__":
    main()
