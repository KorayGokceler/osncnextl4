#!/usr/bin/env python
'''
oscNext L4 processing + HDF5 booking.

Usage:

  # MC (GENIE)
  python process_L4.py \
      --gcd  /data/GCD/GeoCalibDetectorStatus_2013.56429_V1.i3.gz \
      --input "/data/L3/genie/14640/*.i3.zst" \
      --output-i3   /data/L4/genie/14640/L4_14640.i3.zst \
      --output-hdf5 /data/L4/hdf5/genie/14640/L4_14640.hdf5 \
      --mc

  # Detector data
  python process_L4.py \
      --gcd  /data/GCD/Run00125150_GCD.i3.zst \
      --input "/data/L3/data/Run00125150/*.i3.zst" \
      --output-hdf5 /data/L4/hdf5/data/Run00125150/L4_Run00125150.hdf5

Important: --no-cut is the default.  Do not apply a cut before the
classifiers are trained; book every event.
'''

import os
import re
import sys
import glob
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))     # repo root, for `oscnext_l4`

from oscnext_l4.env import (require_icetray, get_I3Tray, report_missing,
                            load_deserialization_libs, IceTrayNotAvailable)

try:
    require_icetray()
except IceTrayNotAvailable as _e:
    # Show what to do, not a stack trace.
    sys.exit("\n" + str(_e) + "\n")
from icecube import icetray, dataio, dataclasses
I3Tray = get_I3Tray()

# Required so that frame objects can be deserialised.  Importing them is
# MANDATORY even though they are not used directly -- otherwise you get
# "Deserialization failed for object at frame key 'X'".
#   simclasses     -> I3MCPESeriesMap, I3MCPulseSeriesMap, noise_weight
#   recclasses     -> I3DST, PoleMuonLlhFitFitParams, ...
#   genie_icetray  -> I3GenieInfo, I3GenieResult   <-- for n_flux_events
#   sim_services   -> I3MCPEShifter vb.
load_deserialization_libs()

from oscnext_l4.booker import add_booker
# validate_files was USED at the --scan step but never imported, so the default
# --scan quick died with NameError on every CLI run.  It went unnoticed because
# the notebook path (run_process_parallel) passes --scan off.
from oscnext_l4.filescan import validate_files
from oscnext_l4.variables import (
    oscNext_L4, L4_HDF5_KEYS, HITSTAT_KEY, HITMULT_KEY,
    UNCLEANED_PULSES_DEFAULT, CLEANED_PULSES_DEFAULT,
)


# ---------------------------------------------------------------------------
# Frame objects to book
# ---------------------------------------------------------------------------

# Variables coming from L3 -- 4 of the muon BDT's inputs and 2 of the noise
# BDT's live inside this map.
L3_KEYS = [
    "IC2018_LE_L3_Vars",
    "IC2018_LE_L3_bools",
    "L3_oscNext_bool",      # IC2018_LE_L3_Full AND Data_quality_bool
    "Data_quality_bool",
]

# Hit statistics / multiplicity -- cog_z, z_sigma, z_travel, n_hit_doms
COMMON_VAR_KEYS = [HITSTAT_KEY, HITMULT_KEY]

# Always needed
BASE_KEYS = ["I3EventHeader"]

# Simulation only.  Note: MCInIcePrimary does NOT exist in the noise MC --
# use the --noise flag to leave these out when booking vuvuzela files.
# It is absent from the pass3 GENIE L3 output too; the truth information is in
# I3MCWeightDict (PrimaryNeutrinoEnergy/Zenith/Type).
# NEvPerFile matters for the weight normalisation.
MC_KEYS = [
    "I3MCWeightDict",
    "NEvPerFile",
    "I3GenieSystWeightDict",
    # booked when present, silently skipped otherwise
    "MCInIcePrimary",
    "MCDeepCoreStartingEvent",
    "MCExtraTruthInfo",
]

# Vuvuzela: the weight is in "noise_weight".
# UNIT WARNING: 1/ns in pass3 (needs x1e9), already Hz in pass2.
NOISE_MC_KEYS = [
    "I3MCWeightDict",
    "noise_weight",
]

# MuonGun weight candidates -- whichever exists is booked
MUONGUN_KEYS = [
    "I3MCWeightDict",
    "MuonWeight",
    "MuonWeight_GaisserH4a",
    "MuonGunWeight",
]

# CORSIKA: the weight is computed with simweights, which expects
# CorsikaWeightMap + PolyplopiaPrimary.  Both must be booked or CorsikaWeighter
# cannot be constructed.
CORSIKA_KEYS = [
    "CorsikaWeightMap",
    "PolyplopiaPrimary",
    "PolyplopiaInfo",
    "I3PrimaryInjectorInfo",
    "I3CorsikaInfo",
]


# ---------------------------------------------------------------------------
# Corrupt input files
# ---------------------------------------------------------------------------
#
# The pass3 production contains the occasional half-written / corrupt
# .i3.zst.  When I3Reader reaches one it throws
#
#     FATAL (I3Reader): Error reading <file> at frame N: input stream error!
#
# and kills the WHOLE tray.  In a 100-file job, one bad file wastes the
# processing of the 99 healthy ones and leaves behind a half-written HDF5 that
# cannot be opened (the "Table ... is still connected" message is a consequence
# of that, not a separate bug).
#
# The protection has two layers:
#   1) Pre-scan   -- the first N frames of every file are read and the ones
#      that do not open are dropped.  Catches truncated/empty files in seconds
#      (the typical error is at frame 4).
#   2) Run time   -- if the tray still dies, the file name is extracted from
#      the error message, written to a blacklist, and the run is retried
#      without that file.

_BAD_FILE_RE = re.compile(r"Error reading (\S+?) at frame")


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------
#
# Notebook process_L4.py'yi subprocess olarak calistiriyor.  Ilerleme
# gorunsun diye stdout'a MAKINE OKUNUR satirlar basiyoruz:
#
#   [CHUNK] 3/10 files=30/100 booked=1840 elapsed=312.4
#   [PROGRESS] frames=15000 physics=7100 booked=4300 elapsed=98.2 rate=152.7
#
# The notebook parses these lines and draws the bar; they are readable in a
# terminal too.  stdout must be line buffered, otherwise the notebook sees
# nothing until the job is over.


def _emit(line):
    """Print a progress line and flush IMMEDIATELY (so it is not held in the
    subprocess buffer)."""
    print(line, flush=True)


class ProgressReporter:
    """Tray module (function) that prints a [PROGRESS] line every N frames."""

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
    """L4_nue.hdf5 -> L4_nue_part003.hdf5  (matches the notebook glob L4_nue*.hdf5)"""
    if not path:
        return None
    base, ext = os.path.splitext(path)
    return "%s_part%03d%s" % (base, index, ext)


def _write_meta(output, meta):
    """
    Write <output>.meta.json.

    THE FIELD THAT MATTERS: n_l3_files -- how many L3 files this HDF5 was made
    from.  The weight normalisation (OneWeight/n_flux/n_files) must be divided
    by that number, NOT by the number of HDF5 files.  The notebook reads it
    from this sidecar.
    """
    if not output:
        return
    path = output + ".meta.json"
    try:
        with open(path, "w") as fh:
            json.dump(meta, fh, indent=2)
    except OSError as e:
        print("  [!] could not write meta: %s" % e)


_WANT_USAGE = {"on": False}


def _print_usage(tray):
    """
    Modul bazli CPU zamanini bas (--usage).

    IceTray her modulun ne kadar surdugunu zaten tutuyor; tahmin etmek
    yerine buna bakin.  Ciktida "usermodule" satirlari en yavas modulleri
    gosterir -- optimizasyona oradan baslayin.
    """
    if not _WANT_USAGE["on"]:
        return
    try:
        usage = tray.Usage()
    except Exception as e:
        print("  [!] modul zamanlamasi alinamadi: %s" % e)
        return
    rows = []
    for key, u in usage.items():
        rows.append((getattr(u, "usertime", 0.0) + getattr(u, "systime", 0.0),
                     getattr(u, "ncall", 0), key))
    rows.sort(reverse=True)
    total = sum(r[0] for r in rows) or 1.0
    print()
    print("=" * 66)
    print("MODUL BAZLI CPU ZAMANI  (toplam %.1f s)" % total)
    print("=" * 66)
    print("%-38s %9s %7s %8s" % ("modul", "cpu [s]", "%", "cagri"))
    for t, n, key in rows[:25]:
        print("%-38s %9.1f %6.1f%% %8d" % (key[:38], t, 100 * t / total, n))
    print()


def _pct(n, total):
    return "%.1f%%" % (100.0 * n / total) if total else "-"


def _bad_list_path(output):
    return (output or "process_L4") + ".badfiles.txt"


def _record_bad(output, paths, reason):
    """Append the corrupt files to <output>.badfiles.txt."""
    if not paths:
        return
    path = _bad_list_path(output)
    try:
        with open(path, "a") as fh:
            for p in paths:
                fh.write("%s\t%s\n" % (p, reason))
        print("  Blacklist: %s" % path)
    except OSError as e:
        print("  [!] could not write the blacklist: %s" % e)


def _run_tray(build, infiles, output_hdf5, retries):
    """
    Run the tray.  If it dies on a corrupt file, drop that file and retry
    (at most `retries` times).
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
            _print_usage(tray)
            return counter, infiles
        except RuntimeError as e:
            m = _BAD_FILE_RE.search(str(e))
            if not m or attempt >= retries:
                raise
            bad = m.group(1)
            if bad not in infiles:
                raise
            attempt += 1
            print("\n[!] Corrupt file caught at run time:\n    %s" % bad)
            print("    %s" % str(e).strip().splitlines()[0][:200])
            _record_bad(output_hdf5, [bad], "run time: input stream error")
            infiles = [f for f in infiles if f != bad]
            # A half-written HDF5 is unusable and cannot be reopened.
            if output_hdf5 and os.path.exists(output_hdf5):
                try:
                    os.remove(output_hdf5)
                    print("    Removed the partial HDF5, starting over.")
                except OSError as rm:
                    print("    [!] could not remove the partial HDF5: %s" % rm)
            print("    Files left: %d  (attempt %d/%d)\n"
                  % (len(infiles), attempt, retries))
            if not infiles:
                raise RuntimeError("Every input file turned out to be corrupt.")


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
    p = argparse.ArgumentParser(description="oscNext L4 processing + HDF5 booking")
    p.add_argument("--gcd", required=True, help="GCD file")
    p.add_argument("--input", nargs="+", default=[],
                   help="input .i3 files (glob patterns accepted)")
    p.add_argument("--output-i3", default=None, help="output .i3 (optional)")
    p.add_argument("--output-hdf5", required=True, help="output .hdf5")

    p.add_argument("--uncleaned-pulses", default=UNCLEANED_PULSES_DEFAULT)
    p.add_argument("--cleaned-pulses", default=CLEANED_PULSES_DEFAULT)
    p.add_argument("--sub-event-stream", default="InIceSplit")

    p.add_argument("--mc", action="store_true", help="simulation (book the MC truth)")
    p.add_argument("--noise", action="store_true",
                   help="pure noise MC (books noise_weight)")
    p.add_argument("--muongun", action="store_true",
                   help="MuonGun MC (books the muon weight keys)")
    p.add_argument("--genie", action="store_true",
                   help="GENIE MC -- carries I3GenieInfo.n_flux_events into every frame")
    p.add_argument("--corsika", action="store_true",
                   help="CORSIKA MC -- books CorsikaWeightMap + PolyplopiaPrimary")

    p.add_argument("--no-l3-cut", action="store_true",
                   help="do not apply the L3 cut (when the input already passed L3)")
    p.add_argument("--apply-cut", action="store_true",
                   help="apply the L4 classifier cut (the models must be trained)")
    p.add_argument("--model-dir", default=None,
                   help="directory holding the trained L4_<tag>_model.txt files")

    p.add_argument("--n", type=int, default=0, help="number of frames to process (0=all)")

    p.add_argument("--input-list", default=None,
                   help="read the input files from this text file (one path per "
                        "line).  Use with the scan_files.py --good-list output so "
                        "the scan happens once instead of in every job.")
    p.add_argument("--scan", choices=["quick", "full", "off"], default="quick",
                   help="pre-scan the input files: quick=first frames (default, "
                        "catches truncated files), full=the whole file (slow but "
                        "certain), off=no scan")
    p.add_argument("--scan-frames", type=int, default=25,
                   help="frames read per file in --scan quick mode (default 25)")
    p.add_argument("--run-optional", action="store_true",
                   help="also compute the variables that are NOT BDT inputs: "
                        "I3TensorOfInertia (L4_ToI) and separation_in_cogs.  "
                        "Neither appears in Table 11/12, so this is off by "
                        "default; it only costs processing time.")
    p.add_argument("--micro-count-uncleaned", action="store_true",
                   help="start the micro_count chain from the UNCLEANED series -- "
                        "the behaviour of the original pass2 code.  The default "
                        "(no flag) follows the technical note and starts from the "
                        "cleaned series.  Only for reproducing the pass2 numbers.")
    p.add_argument("--usage", action="store_true",
                   help="print PER-MODULE CPU time at the end, to see which module "
                        "is slow.  Run this before optimising anything.")
    p.add_argument("--progress", type=int, default=5000,
                   help="print a [PROGRESS] line every N frames (0=off).  The "
                        "notebook draws its progress bar from these lines.")
    p.add_argument("--chunk-files", type=int, default=0,
                   help="process the input in parts of N files; each part is its "
                        "own tray and its own <output>_partNNN.hdf5.  The benefit: "
                        "a REAL percentage/ETA, and a crash loses only that part "
                        "(finished parts are skipped).")
    p.add_argument("--overwrite", action="store_true",
                   help="with --chunk-files: regenerate parts that already exist")
    p.add_argument("--retries", type=int, default=3,
                   help="how many times to drop a corrupt file found at run time "
                        "and retry (default 3, 0=no retry)")
    p.add_argument("--no-hit-statistics", action="store_true",
                   help="do not compute HitStatistics (when L3 already has them)")
    args = p.parse_args()

    if not args.input and not args.input_list:
        p.error("--input ya da --input-list vermelisiniz.")

    _WANT_USAGE["on"] = args.usage

    if args.n > 0 and args.chunk_files > 0:
        p.error("--n and --chunk-files cannot be combined: --n would be "
                "applied separately to every part and give a meaningless "
                "result.  Use --n alone for a smoke test and --chunk-files "
                "alone for production.")

    # --- resolve the input files ---
    infiles = []
    if args.input_list:
        try:
            with open(args.input_list) as fh:
                infiles = [ln.strip() for ln in fh
                           if ln.strip() and not ln.startswith("#")]
        except OSError as e:
            sys.exit("could not read --input-list: %s" % e)
        print("Input list: %s (%d files)" % (args.input_list, len(infiles)))
    for pattern in args.input:
        matched = sorted(glob.glob(pattern))
        infiles.extend(matched if matched else [pattern])
    if not infiles:
        sys.exit("No input files found.")
    print("Input files:", len(infiles))

    # --- remove corrupt files with a pre-scan ---
    if args.scan != "off":
        nf = 0 if args.scan == "full" else args.scan_frames
        if args.n > 0 and len(infiles) > 5:
            print("Pre-scan (%s)... [%d files]" % (args.scan, len(infiles)))
            print("  NOTE: --n was given, so the tray stops in the first files;")
            print("        scanning the whole list is wasted time.  For a smoke")
            print("        test pass a single file or use --scan off.")
        else:
            print("Pre-scan (%s)..." % args.scan)
        infiles, bad = validate_files(infiles, n_frames=nf)
        if bad:
            print("  [!] %d corrupt files dropped:" % len(bad))
            for path, why in bad[:10]:
                print("      %s\n          %s" % (os.path.basename(path), why))
            if len(bad) > 10:
                print("      ... (+%d more)" % (len(bad) - 10))
            _record_bad(args.output_hdf5, [b[0] for b in bad], "pre-scan: " + args.scan)
        print("  Files to process: %d" % len(infiles))
        if not infiles:
            sys.exit("No healthy input file left.")

    for out in (args.output_i3, args.output_hdf5):
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    keys = build_key_list(is_mc=args.mc, is_noise=args.noise,
                          is_muongun=args.muongun, is_corsika=args.corsika)
    print("Keys to book:", len(keys))

    # --- tray ---
    # The tray is built inside a factory function so that, when a corrupt
    # file shows up at run time, it can be REBUILT and rerun without that file
    # (an I3Tray cannot be Executed twice).
    def build_tray(files):
        tray = I3Tray()
        tray.Add("I3Reader", "reader", FilenameList=[args.gcd] + files)

        # --- per-stage counters ------------------------------------------
        # To see WHERE events are lost.  "I gave --n 200 frames and got 60
        # events" is answered by this breakdown.
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
                 run_optional=args.run_optional,
                 apply_cut=args.apply_cut,
                 classifier_model_dir=args.model_dir,
                 micro_count_uncleaned=args.micro_count_uncleaned)

        # --- what survives the L3 cut (and gets booked) ---
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
        print("Parts: %d  (%d files/part)" % (n_chunks, args.chunk_files))
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
        # With --n the tray stops early, so the file list may not have been
        # read in full and n_l3_files is NOT RELIABLE.  None is written so it
        # cannot be used as a weight divisor; load_sample treats the sidecar as
        # missing and warns.
        _write_meta(args.output_hdf5, dict(
            n_l3_files=(None if args.n > 0 else len(used)),
            n_l3_files_unreliable=bool(args.n > 0),
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
    print("  after the L3 cut      : %d  (%s)" % (n_booked, _pct(n_booked, n_stream)))
    print("Events booked           : %d" % n_booked)

    if args.n > 0:
        # --n is a frame count, NOT an event count.  Because the tray stops
        # early, the file list may not have been read in full.
        print()
        print("NOTE: --n %d = %d FRAMES (not events).  The tray stopped after"
              % (args.n, args.n))
        print("      reading that many; the file list may not have been read in")
        print("      full.  File list: %d files (how many were read is unknown)."
              % len(used))
    else:
        print("Files processed         : %d" % len(used))

    if args.chunk_files > 0 and args.output_hdf5:
        print("HDF5 parts              : %s"
              % _part_path(args.output_hdf5, 0).replace("_part000", "_partNNN"))
    else:
        print("HDF5:", args.output_hdf5)
    bl = _bad_list_path(args.output_hdf5)
    if os.path.exists(bl):
        print("Corrupt file list:", bl)

    if n_phys and not n_booked:
        print()
        print("[!] NO events were booked.  Check in order:")
        if not n_stream:
            print("    * --sub-event-stream '%s' may be wrong -- no Physics"
                  % args.sub_event_stream)
            print("      frame is in that stream.")
        else:
            print("    * the L3 cut removed everything -- is the input really")
            print("      L3 output?  To test: --no-l3-cut")


if __name__ == "__main__":
    main()
