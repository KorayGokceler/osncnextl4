#!/usr/bin/env python
"""
Run the pass2 cross-check over EVERY paired L3/L4 file of a sample.

One L3 file per HDF5 (pass2.match() needs that), and the two HDF5 files are
deleted as soon as their report is printed, so the disk stays bounded no
matter how many files there are.

TEMPORARY -- goes with the rest of the fitting scaffolding.
"""
import os
import sys
import glob
import argparse
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from oscnext_l4 import pass2 as P

ap = argparse.ArgumentParser()
ap.add_argument("--sample", required=True)
ap.add_argument("--gcd", required=True)
ap.add_argument("--outdir", default="L4_output/pass2_check")
ap.add_argument("--n-files", type=int, default=0, help="0 = every file")
ap.add_argument("--keep", action="store_true", help="do not delete the HDF5")
# Anything this parser does not recognise is passed straight through to
# process_L4.py.  nargs="*" cannot carry flags: argparse reads the next
# "--flag" as one of its own options and fails.
a, passthrough = ap.parse_known_args()
a.extra = [x for x in passthrough if x != "--"]
if a.extra:
    print("passing through to process_L4.py: %s" % " ".join(a.extra))

pats = P.PASS2_L3[a.sample]
if isinstance(pats, str):
    pats = [pats]
l3 = sorted(f for p in pats for f in glob.glob(p))
if a.n_files:
    l3 = l3[:a.n_files]
pairs, orphans = P.pair_files(l3)

print("sample %s: %d L3 files, %d with an L4 partner, %d orphan(s)"
      % (a.sample, len(l3), len(pairs), len(orphans)), flush=True)
if not os.path.isdir(a.outdir):
    os.makedirs(a.outdir)

flags = list(P.PASS2_FLAGS.get(a.sample, [])) + list(a.extra)
cp = P.PASS2_CLEANED_PULSES
ok = fail = 0

for i, (f3, f4) in enumerate(pairs, 1):
    tag = os.path.basename(f3).replace(".i3.zst", "").replace(".i3", "")
    ours = os.path.join(a.outdir, "ours_%s.hdf5" % tag)
    ref = os.path.join(a.outdir, "pass2_%s.hdf5" % tag)
    print("\n===== [%d/%d] %s =====" % (i, len(pairs), tag), flush=True)

    steps = [
        ["python", "scripts/process_L4.py", "--gcd", a.gcd, "--input", f3,
         "--cleaned-pulses", cp, "--output-hdf5", ours, "--scan", "off"] + flags,
        ["python", "scripts/compare_pass2.py", "book", "--gcd", a.gcd,
         "--input", f4, "--cleaned-pulses", cp, "--output-hdf5", ref,
         "--overwrite"],
        ["python", "scripts/compare_pass2.py", "report", "--ours", ours,
         "--pass2", ref, "--cleaned-pulses", cp],
    ]
    try:
        for cmd in steps:
            r = subprocess.run(cmd)
            if r.returncode != 0:
                raise RuntimeError("returncode %d: %s" % (r.returncode, cmd[1]))
        ok += 1
    except Exception as exc:
        fail += 1
        print("FAILED %s -- %s" % (tag, exc), flush=True)
    finally:
        if not a.keep:
            for f in (ours, ref):
                for extra in (f, f + ".badfiles.txt", f.replace(".hdf5", ".meta.json")):
                    if os.path.exists(extra):
                        os.remove(extra)

print("\n%s: %d file(s) done, %d failed" % (a.sample, ok, fail), flush=True)
