#!/usr/bin/env python
'''
Scan input .i3 files and find the corrupt ones.

process_L4.py already does this itself via --scan; this script runs the same
scan standalone, before any processing starts.  Running it ONCE per set before
production and keeping the blacklist is the efficient route: individual jobs
then do not repeat the scan.

Usage:

  # Quick scan (first 25 frames per file) -- catches truncated files
  python scan_files.py '/data/ana/LE/oscNext/pass3/genie/level3/23799/*.i3.zst'

  # Full scan (every frame read; slow but certain)
  python scan_files.py --full '/data/.../*.i3.zst'

  # Write the list of healthy files -> feed it to process_L4.py --input-list
  python scan_files.py --good-list good_23799.txt '/data/.../*.i3.zst'

Output:
  <goodlist>          healthy files, one path per line
  scan_bad.txt        corrupt files + reason  (override with --bad-list)
'''

import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))     # repo root, for `oscnext_l4`
from oscnext_l4.tray_io import validate_files


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="+", help="file paths or glob patterns")
    p.add_argument("--full", action="store_true",
                   help="read every frame (slow but certain).  Default: first N frames.")
    p.add_argument("--frames", type=int, default=25,
                   help="frames read per file in quick mode (default 25)")
    p.add_argument("--good-list", default=None, help="write the healthy files here")
    p.add_argument("--bad-list", default="scan_bad.txt", help="write the corrupt files here")
    args = p.parse_args()

    files, unmatched = [], []
    for pattern in args.input:
        m = sorted(glob.glob(pattern))
        if m:
            files.extend(m)
        elif os.path.exists(pattern):
            files.append(pattern)
        else:
            unmatched.append(pattern)
    for pattern in unmatched:
        # Not a corrupt file: nothing is there.  Listing it among the corrupt
        # ones sent people looking for a damaged file that does not exist.
        print("[!] matches nothing: %s" % pattern)
    if not files:
        sys.exit("No files found.")
    for out in (args.good_list, args.bad_list):
        d = os.path.dirname(os.path.abspath(out)) if out else None
        if d and not os.path.isdir(d):
            sys.exit("no such directory for %s" % out)

    print("Files to scan: %d  (%s mode)"
          % (len(files), "full" if args.full else "quick/%d frames" % args.frames))

    good, bad = validate_files(files, n_frames=0 if args.full else args.frames)

    print()
    print("=" * 62)
    print("Healthy : %d" % len(good))
    print("Corrupt : %d" % len(bad))
    print("=" * 62)

    if bad:
        print()
        for path, why in bad:
            print("  %s" % path)
            print("      %s" % why)
        with open(args.bad_list, "w") as fh:
            for path, why in bad:
                fh.write("%s\t%s\n" % (path, why))
        print("\n-> %s" % args.bad_list)
    elif os.path.exists(args.bad_list):
        # A clean scan must not leave an earlier run's list looking current.
        os.remove(args.bad_list)

    if args.good_list:
        with open(args.good_list, "w") as fh:
            fh.write("\n".join(good) + "\n")
        print("-> %s  (%d files)" % (args.good_list, len(good)))
        print("\nUsage:")
        print("  python scripts/process_L4.py --input-list %s --scan off ..."
              % args.good_list)

    # Exit code 1 when anything was corrupt or missing, so a wrapper script
    # can check it.
    return 1 if bad or unmatched else 0


if __name__ == "__main__":
    sys.exit(main())
