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

  # L3 .i3 -> L4 .i3 only: the L4 variables, both classifier scores and the
  # L4_oscNext_bool, with no HDF5 at all (needs BOTH trained models)
  python process_L4.py \
      --gcd  /data/GCD/GeoCalibDetectorStatus_2013.56429_V1.i3.gz \
      --input /data/L3/genie/14640/L3_14640.000000.i3.zst \
      --output-i3 /data/L4/genie/14640/L4_14640.000000.i3.zst \
      --mc --genie --apply-cut --model-dir models

At least one of --output-i3 / --output-hdf5 is required, and either may be
given alone.  The side files (<output>.meta.json, <output>.badfiles.txt) go
next to the HDF5 when there is one, and next to the .i3 otherwise.

Important: --apply-cut is OFF by default.  Do not apply it before the
classifiers are trained; book every event.  With it, the L4_oscNext_bool is
WRITTEN for every event and NO event is dropped -- the same as the
production's pass2 L4 files, which hold every event with its bool.
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

from oscnext_l4.env import (get_I3Tray, report_missing,
                            load_deserialization_libs)
from icecube import icetray, dataio, dataclasses
I3Tray = get_I3Tray()

# Required so that frame objects can be deserialised.  Importing them is
# MANDATORY even though they are not used directly -- otherwise you get
# "Deserialization failed for object at frame key 'X'".
#   simclasses     -> I3MCPESeriesMap, I3MCPulseSeriesMap, noise_weight
#   recclasses     -> I3DST, PoleMuonLlhFitFitParams, ...
#   genie_icetray  -> I3GenieInfo, I3GenieResult   <-- for n_flux_events
#   sim_services   -> I3MCPEShifter and the like
load_deserialization_libs()

# validate_files was USED at the --scan step but never imported, so the default
# --scan quick died with NameError on every CLI run.  It went unnoticed because
# the notebook path (run_process_parallel) passes --scan off.
from oscnext_l4.tray_io import add_booker, validate_files
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
# The notebook runs process_L4.py as a subprocess, so progress has to travel
# back over stdout.  These are MACHINE READABLE lines written for that:
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


# The compression suffixes I3Writer and I3File recognise.  They sit AFTER
# ".i3", so a plain splitext would split "L4_nue.i3.zst" into ("L4_nue.i3",
# ".zst") and name the part "L4_nue.i3_part003.zst" -- a file no *.i3.zst glob
# finds.  HDF5 names carry a single extension and are unaffected.
_I3_COMPRESSION = (".zst", ".gz", ".bz2", ".xz")


def _part_path(path, index):
    """
    L4_nue.hdf5   -> L4_nue_part003.hdf5    (matches the notebook glob L4_nue*.hdf5)
    L4_nue.i3.zst -> L4_nue_part003.i3.zst  (the compression suffix stays last)
    """
    if not path:
        return None
    base, ext = os.path.splitext(path)
    if ext in _I3_COMPRESSION:
        base, inner = os.path.splitext(base)
        ext = inner + ext
    return "%s_part%03d%s" % (base, index, ext)


def _output_kind(path):
    """What a run's anchor output is, for the messages that name it."""
    return "HDF5" if path.endswith((".hdf5", ".h5")) else "I3 file"


def _write_meta(output, meta):
    """
    Write <output>.meta.json.

    THE FIELD THAT MATTERS: n_l3_files -- how many L3 files this HDF5 was made
    from.  The weight normalisation (OneWeight/n_flux/n_files) must be divided
    by that number, NOT by the number of HDF5 files.  The notebook reads it
    from this sidecar.  An i3-only run writes it next to the .i3 instead, where
    it records the same thing for whoever weights that file later.
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

    IceTray already records how long every module took, so read that rather
    than guessing.  The "usermodule" lines name the slowest modules -- that is
    where optimisation starts.
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


def _run_tray(build, infiles, output, retries):
    """
    Run the tray.  If it dies on a corrupt file, drop that file and retry
    (at most `retries` times).

    `output` is the run's ANCHOR: the HDF5 when there is one, the .i3
    otherwise.  The blacklist goes next to it and a half-written one is
    removed before the retry.  (With both outputs the .i3 is not removed: the
    rebuilt tray's I3Writer truncates it anyway.)
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
            _record_bad(output, [bad], "run time: input stream error")
            infiles = [f for f in infiles if f != bad]
            # A half-written output is unusable; an HDF5 cannot even be
            # reopened.
            if output and os.path.exists(output):
                try:
                    os.remove(output)
                    print("    Removed the partial %s, starting over."
                          % _output_kind(output))
                except OSError as rm:
                    print("    [!] could not remove the partial %s: %s"
                          % (_output_kind(output), rm))
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
    p.add_argument("--output-i3", default=None,
                   help="output .i3 (.i3.zst etc.): the L3 frames plus every L4 "
                        "key.  Physics frames that fail the L3 cut or are not "
                        "in --sub-event-stream are not written.")
    p.add_argument("--output-hdf5", default=None,
                   help="output .hdf5: the booked keys, for training.  At least "
                        "one of --output-i3 / --output-hdf5 is required.")

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
                   help="run both trained classifiers and write their scores "
                        "and the L4_oscNext_bool into every event.  No event "
                        "is dropped.  Needs --model-dir.")
    p.add_argument("--model-dir", default=None,
                   help="directory holding the trained L4_noise_model.txt and "
                        "L4_muon_model.txt (each with its .json beside it)")

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
    p.add_argument("--accumulated-time-pass2", dest="accumulated_time_pass2",
                   action="store_true", default=True,
                   help="reproduce the pass2 production's accumulated_time: take "
                        "the pulse BEFORE the cumulative charge crosses 75%% "
                        "(99.40%% over 8144 pass2 events).  THIS IS THE DEFAULT.")
    p.add_argument("--accumulated-time-note", dest="accumulated_time_pass2",
                   action="store_false",
                   help="follow the technical note instead: take the pulse AT the "
                        "crossing.  Agrees with pass2 in 0.98%% of events.")
    p.add_argument("--micro-count-uncleaned", dest="micro_count_uncleaned",
                   action="store_true", default=True,
                   help="start the micro_count chain from the UNCLEANED series, as "
                        "the original pass2 code does.  THIS IS THE DEFAULT.")
    p.add_argument("--micro-count-cleaned", dest="micro_count_uncleaned",
                   action="store_false",
                   help="follow Table 11 instead and start from the cleaned "
                        "series.  The two differ in about 9%% of events.")
    p.add_argument("--usage", action="store_true",
                   help="print PER-MODULE CPU time at the end, to see which module "
                        "is slow.  Run this before optimising anything.")
    p.add_argument("--progress", type=int, default=5000,
                   help="print a [PROGRESS] line every N frames (0=off).  The "
                        "notebook draws its progress bar from these lines.")
    p.add_argument("--chunk-files", type=int, default=0,
                   help="process the input in parts of N files; each part is its "
                        "own tray and its own <output>_partNNN.hdf5 (and/or "
                        "_partNNN.i3.zst).  The benefit: a REAL percentage/ETA, "
                        "and a crash loses only that part (finished parts are "
                        "skipped).")
    p.add_argument("--overwrite", action="store_true",
                   help="with --chunk-files: regenerate parts that already exist")
    p.add_argument("--retries", type=int, default=3,
                   help="how many times to drop a corrupt file found at run time "
                        "and retry (default 3, 0=no retry)")
    p.add_argument("--no-hit-statistics", action="store_true",
                   help="do not compute HitStatistics (when L3 already has them)")
    args = p.parse_args()

    if not args.input and not args.input_list:
        p.error("--input or --input-list is required.")

    if not args.output_i3 and not args.output_hdf5:
        p.error("no output: give --output-i3, --output-hdf5, or both.")

    # Checked here, not where the tray finally needs them: the L4Classifier
    # modules only open their model files when the tray is configured, which
    # is AFTER the pre-scan -- on a long input list, minutes into the job.
    if args.apply_cut:
        if not args.model_dir:
            p.error("--apply-cut needs --model-dir (the trained models).")
        absent = [f for f in ("L4_noise_model.txt", "L4_muon_model.txt")
                  if not os.path.exists(os.path.join(args.model_dir, f))]
        if absent:
            p.error("--apply-cut runs BOTH classifiers, and %s has no %s."
                    % (args.model_dir, " / ".join(absent)))

    # THE ANCHOR: the output the side files are named after.  The HDF5 when
    # there is one -- so a run that books one names everything exactly as it
    # always has -- and the .i3 otherwise:
    #     <anchor>.meta.json      n_l3_files, the weight divisor
    #     <anchor>.badfiles.txt   the corrupt inputs that were dropped
    # and, with --chunk-files, the part whose existence marks it finished.
    anchor = args.output_hdf5 or args.output_i3

    # Which icetray projects are absent.  `variables.py` records them at import
    # time through `optional_project`; without this call the record is kept and
    # never shown, so a missing project only surfaced later as a require_project
    # error deep inside a segment.
    report_missing()

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

    # BEFORE the pre-scan, which writes its blacklist next to the anchor.  This
    # used to come after it, so on a first run into a new directory the list of
    # dropped files was lost to "could not write the blacklist".  The notebook
    # never saw it: runner.py creates the directory before calling this.
    for out in (args.output_i3, args.output_hdf5):
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

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
            _record_bad(anchor, [b[0] for b in bad], "pre-scan: " + args.scan)
        print("  Files to process: %d" % len(infiles))
        if not infiles:
            sys.exit("No healthy input file left.")

    keys = build_key_list(is_mc=args.mc, is_noise=args.noise,
                          is_muongun=args.muongun, is_corsika=args.corsika)
    if args.output_hdf5:
        print("Keys to book:", len(keys))
    else:
        print("No HDF5 -- nothing is booked; the .i3 carries every frame key.")

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

        # Progress: counts EVERY frame (Q/P/G/C/D) and prints one line every
        # --progress frames.  This is what feeds the notebook's bar.
        if args.progress:
            tray.Add(ProgressReporter(args.progress, counter, time.time()),
                     "progress")

        # Only the physics sub-event stream
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
                 micro_count_uncleaned=args.micro_count_uncleaned,
                 accumulated_time_pass2=args.accumulated_time_pass2)

        # --- what survives the L3 cut (and gets booked) ---
        def count(frame):
            counter["n"] += 1
            return True
        tray.Add(count, "counter")

        out_i3 = getattr(build_tray, "output_i3", args.output_i3)
        if out_i3:
            # DropOrphanStreams=[DAQ]: a Q frame none of whose P frames
            # survived the stream filter and the L3 cut is not written.  The
            # production's own writer does exactly this
            # (icetray-oscNext/.../tools/processor.py), and without it an L4
            # file is mostly Q frames with nothing after them -- the L3 cut
            # keeps only ~8% of noise events.  The HDF5 is not affected:
            # hdfwriter books P frames only.
            tray.Add("I3Writer", "writer",
                     Filename=out_i3,
                     Streams=[icetray.I3Frame.TrayInfo,
                              icetray.I3Frame.DAQ,
                              icetray.I3Frame.Physics,
                              icetray.I3Frame.Stream("S"),
                              icetray.I3Frame.Stream("M")],
                     DropOrphanStreams=[icetray.I3Frame.DAQ])

        out_hdf5 = getattr(build_tray, "output_hdf5", args.output_hdf5)
        if out_hdf5:
            add_booker(tray, "booker",
                       output=out_hdf5,
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

    if args.chunk_files and args.chunk_files > 0:
        chunks = [infiles[i:i + args.chunk_files]
                  for i in range(0, len(infiles), args.chunk_files)]
        n_chunks = len(chunks)
        print("Parts: %d  (%d files/part)" % (n_chunks, args.chunk_files))
        _emit("[CHUNK] 0/%d files=0/%d booked=0 elapsed=0.0"
              % (n_chunks, len(infiles)))

        n_done_files = 0
        for ci, chunk in enumerate(chunks):
            # The anchor's part marks the part finished; see `anchor` above.
            hdf5_part = _part_path(args.output_hdf5, ci)
            i3_part = _part_path(args.output_i3, ci)
            out_part = hdf5_part or i3_part
            n_done_files += len(chunk)

            # Skip a finished part -> a crashed run resumes where it stopped
            if os.path.exists(out_part) and not args.overwrite:
                print("[%d/%d] skipped (already there): %s"
                      % (ci + 1, n_chunks, os.path.basename(out_part)))
                _emit("[CHUNK] %d/%d files=%d/%d booked=%d elapsed=%.1f"
                      % (ci + 1, n_chunks, n_done_files, len(infiles),
                         totals["n"], time.time() - t_start))
                continue

            build_tray.output_hdf5 = hdf5_part
            build_tray.output_i3 = i3_part
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
        totals, used = _run_tray(build_tray, infiles, anchor, args.retries)
        # With --n the tray stops early, so the file list may not have been
        # read in full and n_l3_files is NOT RELIABLE.  None is written so it
        # cannot be used as a weight divisor; load_sample treats the sidecar as
        # missing and warns.
        _write_meta(anchor, dict(
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

    # Each output that was asked for, in the chunked or the single form.
    for label, out in (("HDF5", args.output_hdf5), ("I3", args.output_i3)):
        if not out:
            continue
        if args.chunk_files > 0:
            print("%-24s: %s" % (label + " parts",
                                 _part_path(out, 0).replace("_part000", "_partNNN")))
        else:
            print("%s:" % label, out)
    bl = _bad_list_path(anchor)
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
