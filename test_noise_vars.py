#!/usr/bin/env python
'''
Noise BDT girdilerinin TEK dosyalik hizli testi.

Amac: uretim kosusu (process_L4.py + HDF5) baslatmadan once, 5 noise BDT
girdisinin gercekten hesaplanip hesaplanmadigini ve makul degerler alip
almadigini bir dakikada gormek.

Hesaplananlar (technical note Tablo 11):

    NchCleaned            L3'ten okunur  (IC2018_LE_L3_Vars)
    micro_count           bizim zincir   (L4_micro_count)
    iLineFit_speed        linefit.simple (L4_iLineFit / L4_iLineFitParams)
    fill_ratio            I3FillRatioModule (L4_fill_ratio)
    FullTimeLengthRatio   bizim bolme    (L4_FullTimeLengthRatio)

Ayrica micro_count'u ORIJINAL pass2 zinciriyle de (temizlenmemis seriden
baslayarak) hesaplar ve iki degeri yan yana gosterir -- bkz. CLAUDE.md
acik risk 5b.  HDF5 YAZMAZ, hicbir seyi diske dokmez.

Kullanim (repo dizininden, IceTray ortami acikken):

    python test_noise_vars.py --sample nue    --n 400
    python test_noise_vars.py --sample noise  --n 2000
    python test_noise_vars.py --file /tam/yol/dosya.i3.zst --n 400

Ortam acik degilse:  ./setup_env.sh run python test_noise_vars.py --sample nue
'''

import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import oscNext_L4_variables as V
from icetray_env import get_I3Tray, deepcore_doms, load_lib, require_project
from icecube import icetray, dataclasses, dataio        # noqa: F401


# Notebook bolum 0'daki SAMPLES/GCD ile ayni yollar
GCD_DEFAULT = ("/cvmfs/icecube.opensciencegrid.org/data/GCD/"
               "GeoCalibDetectorStatus_IC86.All_Pass3.i3.gz")

SAMPLE_GLOBS = {
    "nue":     "/data/ana/LE/oscNext/pass3/genie/level3/23800/*.i3.zst",
    "numu":    "/data/ana/LE/oscNext/pass3/genie/level3/23799/*.i3.zst",
    "corsika": "/data/ana/LE/oscNext/pass3/corsika/level3/23694/*.i3.zst",
    "noise":   "/data/ana/LE/oscNext/pass3/noise/level3/23813/*.i3.zst",
}

MC_PASS2_KEY = "L4_micro_count_pass2"     # sadece bu testte, karsilastirma icin

COLS = ["NchCleaned", "micro_count", "micro_cnt(pass2)", "iLineFit_speed",
        "fill_ratio", "FullTimeLenRatio", "CleanedFTL", "UncleanedFTL"]


def add_micro_count_chain(tray, name, pulses, out_key):
    '''
    micro_count zincirini elle kur.

    Segment iki kez cagrilamaz (ikisi de ayni L4_micro_count anahtarina
    yazar), o yuzden pass2 surumu icin zinciri burada tekrar kuruyoruz --
    adimlar oscNext_L4_noise_cut_variables ile birebir ayni, sadece
    baslangic serisi ve cikti anahtari farkli.

    DIKKAT: I3OMSelection yan urun olarak frame'e "BadOMSelection" yazar ve
    bu anahtar parametreyle degistirilmiyor.  Tray'de iki I3OMSelection
    oldugu icin (segmentinki + buradaki) ikincisi ayni anahtari yazmaya
    calisip "frame already contains" ile patliyordu.  Araya Delete koyup
    onceki adimin biraktigi anahtari siliyoruz -- sadece bu testte gerekli,
    uretimde tek OMSelection var.
    '''
    tray.AddModule("Delete", name + "_DelBadOM",
                   Keys=["BadOMSelection"])

    tw = name + "_TW"
    tray.AddModule("I3StaticTWC<I3RecoPulseSeries>", name + "_STWC",
                   InputResponse=pulses,
                   OutputResponse=tw,
                   TriggerConfigIDs=[1010, 1011],
                   TriggerName="I3TriggerHierarchy",
                   WindowMinus=V.STW_MINUS,
                   WindowPlus=V.STW_PLUS)

    fid = tw + "_DCFid"
    tray.AddModule("I3OMSelection<I3RecoPulseSeries>", name + "_Fid",
                   selectInverse=True,
                   InputResponse=tw,
                   OutputResponse=fid,
                   OmittedKeys=deepcore_doms("IC86").DeepCoreFiducialDOMs)

    dtw = fid + ("_DTW%i" % V.DTW)
    tray.AddModule("I3TimeWindowCleaning<I3RecoPulse>", name + "_DTW",
                   InputResponse=fid,
                   OutputResponse=dtw,
                   TimeWindow=V.DTW)

    tray.Add(V._micro_count, name + "_count",
             pulses_key=dtw, output_key=out_key, subkey=V.MICROCOUNT_SUBKEY)


def speed_of(frame):
    '''iLineFit hizi -- surume gore I3Particle'da ya da Params'ta olabiliyor.'''
    if V.L4_LINEFIT_KEY in frame:
        p = frame[V.L4_LINEFIT_KEY]
        if p.fit_status == dataclasses.I3Particle.FitStatus.OK:
            return float(p.speed)
    par = V.L4_LINEFIT_KEY + "Params"
    if par in frame:
        for attr in ("LFVel", "lf_vel", "speed"):
            if hasattr(frame[par], attr):
                return float(getattr(frame[par], attr))
    return np.nan


def pick_input(args):
    if args.file:
        return args.file
    if not args.sample:
        sys.exit("--sample ya da --file verin.  Ornekler: %s"
                 % ", ".join(sorted(SAMPLE_GLOBS)))
    files = sorted(glob.glob(SAMPLE_GLOBS[args.sample]))
    if not files:
        sys.exit("Dosya bulunamadi: %s" % SAMPLE_GLOBS[args.sample])
    return files[args.index]


def build_tray(args, inp, rows, counts):
    I3Tray = get_I3Tray()
    tray = I3Tray()
    tray.Add("I3Reader", FilenameList=[args.gcd, inp])

    def stream_cut(frame):
        counts["physics"] += 1
        ok = frame["I3EventHeader"].sub_event_stream == args.sub_event_stream
        counts["stream"] += ok
        return ok
    tray.Add(stream_cut, "stream_cut", Streams=[icetray.I3Frame.Physics])

    def l3_cut(frame):
        if args.no_l3_cut:
            counts["l3"] += 1
            return True
        ok = False
        if "L3_oscNext_bool" in frame:
            ok = bool(frame["L3_oscNext_bool"].value)
        elif "IC2018_LE_L3_bools" in frame:
            ok = bool(frame["IC2018_LE_L3_bools"]["IC2018_LE_L3_Full"])
        counts["l3"] += ok
        return ok
    tray.Add(l3_cut, "l3_cut", Streams=[icetray.I3Frame.Physics])

    # ortak: L4_first_hlc (fill_ratio'nun vertex'i) + FullTimeLengthRatio
    tray.Add(V.oscNext_L4_common_variables, "common",
             cleaned_pulses=args.cleaned,
             uncleaned_pulses=args.uncleaned)

    # micro_count (NOT surumu: temizlenmis seriden) + fill_ratio
    tray.Add(V.oscNext_L4_noise_cut_variables, "noise",
             fill_ratio_vertex=V.L4_FIRST_HLC_KEY,
             cleaned_pulses=args.cleaned)

    # micro_count (PASS2 surumu: temizlenmemis seriden) -- karsilastirma icin
    add_micro_count_chain(tray, "pass2", args.uncleaned, MC_PASS2_KEY)

    tray.AddSegment(V.linefit.simple, "ilf",
                    inputResponse=args.cleaned,
                    fitName=V.L4_LINEFIT_KEY)

    def collect(frame):
        def l3v(col):
            if "IC2018_LE_L3_Vars" in frame:
                m = frame["IC2018_LE_L3_Vars"]
                if col in m:
                    return float(m[col])
            return np.nan

        def micro(key):
            if key in frame and V.MICROCOUNT_SUBKEY in frame[key]:
                return float(frame[key][V.MICROCOUNT_SUBKEY])
            return np.nan

        rows.append((
            l3v("NchCleaned"),
            micro(V.L4_MICROCOUNT_KEY),
            micro(MC_PASS2_KEY),
            speed_of(frame),
            (float(frame[V.L4_FILL_RATIO_KEY].fill_ratio_from_mean)
             if V.L4_FILL_RATIO_KEY in frame else np.nan),
            (float(frame[V.L4_FTLR_KEY].value)
             if V.L4_FTLR_KEY in frame else np.nan),
            l3v("CleanedFullTimeLength"),
            l3v("UncleanedFullTimeLength"),
        ))
        return True
    tray.Add(collect, "collect", Streams=[icetray.I3Frame.Physics])
    return tray


def report(a, counts, args):
    print("")
    print("=" * 78)
    print("Physics frame        : %d" % counts["physics"])
    print("  %-18s : %d" % (args.sub_event_stream, counts["stream"]))
    print("  L3 kesimi sonrasi  : %d%s"
          % (counts["l3"], "  (--no-l3-cut)" if args.no_l3_cut else ""))
    print("Toplanan olay        : %d" % len(a))
    print("=" * 78)
    if len(a) == 0:
        print("Hic olay yok -- --n degerini buyutun ya da --no-l3-cut deneyin.")
        return

    print("")
    print("ILK %d OLAY" % min(args.show, len(a)))
    hdr = "".join("%18s" % n for n in COLS[:6])
    print(hdr)
    print("-" * len(hdr))
    for r in a[:args.show]:
        print("".join("%18.4g" % v for v in r[:6]))

    print("")
    print("OZET  (%d olay)" % len(a))
    print("%-18s %7s %10s %10s %10s %10s"
          % ("degisken", "NaN", "min", "medyan", "ort", "max"))
    print("-" * 70)
    for i, n in enumerate(COLS):
        c = a[:, i]
        nan = int(np.isnan(c).sum())
        if nan == len(c):
            print("%-18s %7d %10s %10s %10s %10s" % (n, nan, "-", "-", "-", "-"))
            continue
        print("%-18s %7d %10.4g %10.4g %10.4g %10.4g"
              % (n, nan, np.nanmin(c), np.nanmedian(c),
                 np.nanmean(c), np.nanmax(c)))

    print("")
    print("SAGLIK KONTROLU")
    print("-" * 70)

    def rng_check(label, col, lo, hi):
        c = a[:, col]
        c = c[~np.isnan(c)]
        if len(c) == 0:
            print("  [ ? ] %-22s hepsi NaN" % label)
            return
        bad = int(((c < lo) | (c > hi)).sum())
        print("  [%s] %-22s [%g, %g] disinda: %d / %d"
              % ("OK " if bad == 0 else "!! ", label, lo, hi, bad, len(c)))

    rng_check("fill_ratio", 4, 0.0, 1.0)           # Sekil 13 ekseni
    rng_check("FullTimeLengthRatio", 5, 0.0, 1.0)  # Sekil 13 ekseni

    # FTLR gercekten cleaned/uncleaned mi?
    ftlr, cl, ucl = a[:, 5], a[:, 6], a[:, 7]
    m = ~(np.isnan(ftlr) | np.isnan(cl) | np.isnan(ucl)) & (ucl > 0)
    if m.sum():
        d = np.abs(ftlr[m] - cl[m] / ucl[m])
        print("  [%s] %-22s maks sapma %.3g (%d olay)"
              % ("OK " if d.max() < 1e-9 else "!! ",
                 "FTLR == cl/uncl", d.max(), int(m.sum())))

    # micro_count: not surumu vs pass2 surumu
    mc, mc2 = a[:, 1], a[:, 2]
    m = ~(np.isnan(mc) | np.isnan(mc2))
    if m.sum():
        print("  [ i ] micro_count  not=%.3g  pass2=%.3g  (medyan)"
              % (np.median(mc[m]), np.median(mc2[m])))
        print("        pass2 > not : %d / %d olay   (esit: %d)"
              % (int((mc2[m] > mc[m]).sum()), int(m.sum()),
                 int((mc2[m] == mc[m]).sum())))
        print("        --> pass2 surumu temizlenmemis seriden sayiyor;")
        print("            gurultu hitleri de sayildigi icin daha buyuk olmali.")

    sp = a[:, 3]
    sp = sp[~np.isnan(sp)]
    if len(sp):
        near_c = int(((sp > 0.2) & (sp < 0.4)).sum())
        print("  [ i ] iLineFit_speed  0.2-0.4 m/ns arasi: %d / %d  (%.0f%%)"
              % (near_c, len(sp), 100.0 * near_c / len(sp)))
    print("")


def main():
    ap = argparse.ArgumentParser(
        description="Noise BDT girdilerinin tek dosyalik testi (HDF5 yazmaz).")
    ap.add_argument("--sample", choices=sorted(SAMPLE_GLOBS),
                    help="hazir set adi (yollar notebook bolum 0 ile ayni)")
    ap.add_argument("--index", type=int, default=0,
                    help="--sample icinde kacinci dosya (varsayilan 0)")
    ap.add_argument("--file", help="--sample yerine tam dosya yolu")
    ap.add_argument("--gcd", default=GCD_DEFAULT)
    ap.add_argument("--n", type=int, default=400,
                    help="islenecek FRAME sayisi (olay degil!), 0=hepsi")
    ap.add_argument("--show", type=int, default=15,
                    help="ekrana basilacak olay sayisi")
    ap.add_argument("--cleaned", default=V.CLEANED_PULSES_DEFAULT)
    ap.add_argument("--uncleaned", default=V.UNCLEANED_PULSES_DEFAULT)
    ap.add_argument("--sub-event-stream", default="InIceSplit")
    ap.add_argument("--no-l3-cut", action="store_true")
    args = ap.parse_args()

    require_project("DomTools")
    require_project("fill_ratio")
    require_project("linefit")
    if not load_lib("static-twc"):
        sys.exit("static-twc kutuphanesi yok -- python diagnose_env.py")

    inp = pick_input(args)
    print("GCD   : %s" % args.gcd)
    print("Girdi : %s" % inp)
    print("Seri  : cleaned=%s  uncleaned=%s" % (args.cleaned, args.uncleaned))

    rows = []
    counts = {"physics": 0, "stream": 0, "l3": 0}
    tray = build_tray(args, inp, rows, counts)

    if args.n > 0:
        tray.Execute(args.n)        # DIKKAT: --n FRAME sayar, olay degil
    else:
        tray.Execute()

    a = np.array(rows, dtype=float) if rows else np.zeros((0, len(COLS)))
    report(a, counts, args)


if __name__ == "__main__":
    main()
