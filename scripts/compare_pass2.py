#!/usr/bin/env python
"""
Cross-check our L4 production against the real pass2 L4 files.

    pass2 L3  --(process_L4.py, our segment)-->  ours.hdf5
    pass2 L4  --(this script, `book`)---------->  pass2.hdf5
    `report`  --> per-variable agreement

See oscnext_l4/pass2.py for what is compared and why, and for the two things
that have to be right before any of it means anything (the pass2 cleaned pulse
series name, and one L3 file per HDF5 so events can be matched).

`inspect` / `plan` / `book` / `report` are the cross-check.  The `fit` /
`fit-report` subcommands that used to sit beside them were scaffolding for
finding the definitions our rewrite did not reproduce; those are settled and
recorded in variables.py and CLAUDE.md, so the scaffolding is gone.  Recover
it from the tag pass2-verified-v1 if a definition ever has to be fitted again.

Subcommands
-----------
  inspect  what is actually in a pass2 frame -- run this FIRST
  plan     print the process_L4.py command lines for a sample
  book     book the stored L4 variables out of pass2 L4 files
  report   match the two HDF5 files and compare
"""

import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oscnext_l4 import pass2 as P


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def expand(patterns):
    """Expand globs, keeping the order and dropping duplicates."""
    out = []
    for pat in patterns:
        hits = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
        if not hits:
            print("  [!] no match: %s" % pat)
        out.extend(hits)
    return list(dict.fromkeys(out))


def _icetray():
    """Import IceTray only in the subcommands that need it."""
    from oscnext_l4.env import get_I3Tray, load_deserialization_libs
    from icecube import icetray, dataio, dataclasses
    load_deserialization_libs()
    return icetray, dataio, dataclasses, get_I3Tray()


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

def _typename(frame, key):
    """
    The frame object's type, or why it cannot be read.

    A pass2 L4 frame carries keys belonging to projects this meta-project does
    not have -- L4_Dunkman_<pulses>_Variables needs analysis.event_selection,
    one of the projects whose absence is why the variables were rewritten in
    the first place.  Touching such a key raises, so the dump must not touch
    it blindly: the whole point of inspecting is to LIST what is there, and
    dying on the first unreadable key hides everything after it.
    """
    try:
        return type(frame[key]).__name__
    except Exception as e:
        msg = str(e).split(".")[0]
        return "<unreadable: %s>" % msg[:70]


def cmd_inspect(args):
    """
    Dump the keys of the first matching P frame.

    Run this before anything else.  It answers the two questions the note
    cannot: what the pulse series are actually called in THESE files, and
    which L4 keys the L4 files really carry.  Everything downstream is built
    on those two answers.
    """
    icetray, dataio, dataclasses, _ = _icetray()

    for path in expand(args.input):
        print("=" * 78)
        print(path)
        print("=" * 78)
        if not os.path.exists(path):
            print("  [!] does not exist")
            continue
        try:
            f = dataio.I3File(path)
        except RuntimeError as e:
            # One unreadable file must not kill the whole inspection -- the
            # usual reason is that the L4 path was guessed wrong, and the
            # remaining files are exactly what would tell us the right one.
            print("  [!] cannot open: %s" % e)
            continue
        shown = 0
        while f.more() and shown < args.n:
            frame = f.pop_frame()
            if frame.Stop != icetray.I3Frame.Physics:
                continue
            if args.sub_event_stream and "I3EventHeader" in frame:
                if frame["I3EventHeader"].sub_event_stream != args.sub_event_stream:
                    continue
            shown += 1
            print("\n--- P frame %d ---" % shown)
            keys = sorted(frame.keys())
            pulses = [k for k in keys if "Pulse" in k]
            l4 = [k for k in keys if k.startswith("L4_")]
            stats = [k for k in keys if "HitStatistics" in k or "HitMultiplicity" in k]
            print("  pulse series (%d):" % len(pulses))
            for k in pulses:
                print("      %-50s %s" % (k, _typename(frame, k)))
            print("  hit statistics (%d):" % len(stats))
            for k in stats:
                print("      %s" % k)
            print("  L4_* keys (%d):" % len(l4))
            for k in l4:
                print("      %-50s %s" % (k, _typename(frame, k)))
            if args.all_keys:
                print("  all keys (%d):" % len(keys))
                for k in keys:
                    print("      %-50s %s" % (k, _typename(frame, k)))
        f.close()
        if shown == 0:
            print("  no P frame matched (sub_event_stream=%r)"
                  % args.sub_event_stream)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def cmd_plan(args):
    """
    Print the process_L4.py command lines for the "ours" side.

    One L3 file per HDF5 on purpose -- see pass2.match(): (Run, Event,
    SubEvent) is only unique within a single L3 file, so a chunked production
    cannot be matched against the answer key.
    """
    sample = args.sample
    if sample not in P.PASS2_L3:
        sys.exit("unknown sample %r -- known: %s"
                 % (sample, ", ".join(sorted(P.PASS2_L3))))

    l3_files = expand(P.PASS2_L3[sample])[:args.n_files]
    pairs, orphans = P.pair_files(l3_files)

    print("# sample: %s   L3 files: %d   with an L4 partner: %d"
          % (sample, len(l3_files), len(pairs)))
    if orphans:
        print("# no L4 partner for %d file(s), e.g. %s"
              % (len(orphans), orphans[0][1]))
    flags = " ".join(P.PASS2_FLAGS.get(sample, []))
    for l3, l4 in pairs:
        tag = os.path.basename(l3).replace(".i3.zst", "").replace(".i3", "")
        print()
        print("python scripts/process_L4.py --gcd %s \\\n"
              "    --input %s \\\n"
              "    --cleaned-pulses %s \\\n"
              "    --output-hdf5 %s/ours_%s.hdf5 %s"
              % (args.gcd, l3, args.cleaned_pulses, args.outdir, tag, flags))
        print("python scripts/compare_pass2.py book --gcd %s \\\n"
              "    --input %s \\\n"
              "    --cleaned-pulses %s \\\n"
              "    --output-hdf5 %s/pass2_%s.hdf5"
              % (args.gcd, l4, args.cleaned_pulses, args.outdir, tag))
        print("python scripts/compare_pass2.py report \\\n"
              "    --ours %s/ours_%s.hdf5 --pass2 %s/pass2_%s.hdf5 \\\n"
              "    --cleaned-pulses %s"
              % (args.outdir, tag, args.outdir, tag, args.cleaned_pulses))


# ---------------------------------------------------------------------------
# book
# ---------------------------------------------------------------------------

def cmd_book(args):
    """
    Book the L4 variables that are ALREADY in the pass2 L4 files.

    Nothing is computed here.  That is the whole point: this side is the
    answer key, so it must be read out untouched.
    """
    icetray, dataio, dataclasses, I3Tray = _icetray()
    from oscnext_l4.tray_io import add_booker

    # hdfwriter's converters for I3HitStatisticsValues / I3HitMultiplicityValues
    # come from common_variables and are registered when that project is
    # imported.  process_L4.py gets this for free -- its hit-statistics segment
    # imports the module to COMPUTE them.  This script only books what is
    # already in the frame, so nothing would import it and I3TableWriter dies
    # mid-run with "No converter found for ... I3HitMultiplicityValues",
    # leaving a half-written HDF5 behind.
    from oscnext_l4.env import require_project
    require_project("common_variables")
    from icecube import common_variables      # noqa: F401  (registers converters)

    files = expand(args.input)
    if not files:
        sys.exit("no input file")

    keys = P.pass2_book_keys(args.cleaned_pulses)
    print("Booking %d keys from %d file(s) -> %s"
          % (len(keys), len(files), args.output_hdf5))

    if os.path.exists(args.output_hdf5) and not args.overwrite:
        sys.exit("%s exists (use --overwrite)" % args.output_hdf5)

    outdir = os.path.dirname(os.path.abspath(args.output_hdf5))
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir)

    tray = I3Tray()
    tray.Add("I3Reader", "reader", FilenameList=[args.gcd] + files)
    tray.Add(lambda f: (f["I3EventHeader"].sub_event_stream
                        == args.sub_event_stream),
             "stream_filter", Streams=[icetray.I3Frame.Physics])

    counter = {"n": 0}

    def _count(frame):
        counter["n"] += 1
        return True
    tray.Add(_count, "count", Streams=[icetray.I3Frame.Physics])

    add_booker(tray, "booker", output=args.output_hdf5, keys=keys,
               sub_event_streams=(args.sub_event_stream,))
    if args.n:
        tray.Execute(args.n)
    else:
        tray.Execute()

    print("Events booked: %d" % counter["n"])


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def cmd_report(args):
    rows = P.report(args.ours, args.pass2,
                    cleaned_pulses=args.cleaned_pulses,
                    n_worst=args.n_worst,
                    strict=not args.allow_duplicates)
    if args.fail_on_diff:
        bad = [r for r in rows if r["rewritten"] and r["status"] == "DIFFERS"]
        if bad:
            sys.exit(1)



# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("--cleaned-pulses", default=P.PASS2_CLEANED_PULSES,
                        help="pass2 cleaned pulse series (default: %(default)s"
                             " -- technical note sec. 3.2)")

    sp = sub.add_parser("inspect", help="what is in a pass2 frame")
    sp.add_argument("--input", nargs="+", required=True)
    sp.add_argument("--n", type=int, default=1, help="P frames to show")
    sp.add_argument("--sub-event-stream", default="InIceSplit")
    sp.add_argument("--all-keys", action="store_true")
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("plan", help="print the command lines for a sample")
    sp.add_argument("--sample", required=True,
                    help="one of: %s" % ", ".join(sorted(P.PASS2_L3)))
    sp.add_argument("--gcd", required=True)
    sp.add_argument("--outdir", default="L4_output/pass2_check")
    sp.add_argument("--n-files", type=int, default=5)
    common(sp)
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("book", help="book the stored L4 variables")
    sp.add_argument("--gcd", required=True)
    sp.add_argument("--input", nargs="+", required=True)
    sp.add_argument("--output-hdf5", required=True)
    sp.add_argument("--sub-event-stream", default="InIceSplit")
    sp.add_argument("--n", type=int, default=0)
    sp.add_argument("--overwrite", action="store_true")
    common(sp)
    sp.set_defaults(func=cmd_book)

    sp = sub.add_parser("report", help="compare the two HDF5 files")
    sp.add_argument("--ours", required=True)
    sp.add_argument("--pass2", required=True)
    sp.add_argument("--n-worst", type=int, default=5)
    sp.add_argument("--allow-duplicates", action="store_true",
                    help="match even when (Run, Event, SubEvent) repeats -- "
                         "the match is then NOT reliable, see pass2.match()")
    sp.add_argument("--fail-on-diff", action="store_true",
                    help="exit 1 when a rewritten variable disagrees")
    common(sp)
    sp.set_defaults(func=cmd_report)

    args = p.parse_args()
    if not getattr(args, "func", None):
        p.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
