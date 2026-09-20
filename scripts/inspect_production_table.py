#!/usr/bin/env python
'''
Dump the production's own classifier training table.

    /data/ana/LE/oscNext/pass2/resources/classifier_models/level4_muon/
        L4_muon_model_data.hdf5        135 MB   the model the sample used
        L4_muon_model_muongun.hdf5     106 MB   the MC-trained one, unused
    .../level4_noise/
        L4_noise_model.hdf5

WHY THIS IS WORTH READING
  Its README says these are not metadata -- they are the ACTUAL train/test
  events the released models were fitted on.  That makes them a stronger
  source than anything we have used so far:

  - the input list is whatever columns are in the table, not our reading of
    Tables 11/12;
  - the run numbers in it settle which detector-data runs were really used,
    where the note (18 runs, 2012-2017) and the fridge ("one run per month
    2012-2018") disagree;
  - if it carries a train/test flag, our model can be evaluated on THEIR
    held-out set, and theirs on ours;
  - whatever weight column it has is the weight the production trained with,
    which is the open question our own `w_phys` choice sits on.

  Read-only.  It prints; it changes nothing.

USAGE
    python scripts/inspect_production_table.py <file.hdf5> [--rows 5]

pytables + numpy only.
'''
from __future__ import print_function

import argparse
import numpy as np


def walk(h5):
    """Every leaf, with its kind -- the layout is not assumed."""
    out = []
    for node in h5.walk_nodes("/"):
        cls = node.__class__.__name__
        if cls in ("Table", "Array", "CArray", "EArray"):
            out.append((node._v_pathname, cls, node))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--rows", type=int, default=5,
                    help="how many example rows to print per table")
    ap.add_argument("--max-cols", type=int, default=200)
    args = ap.parse_args()

    import tables
    h5 = tables.open_file(args.path, "r")
    try:
        nodes = walk(h5)
        print("file   : %s" % args.path)
        print("leaves : %d" % len(nodes))
        print()

        for path, cls, node in nodes:
            n = int(node.nrows) if hasattr(node, "nrows") else int(node.shape[0])
            print("=" * 74)
            print("%s   [%s]   %d rows" % (path, cls, n))
            print("=" * 74)

            if cls != "Table":
                print("  shape %s dtype %s" % (node.shape, node.dtype))
                if n:
                    print("  first: %s" % np.asarray(node[:min(args.rows, n)]))
                continue

            cols = list(node.colnames)[:args.max_cols]
            print("  %d columns: %s" % (len(node.colnames), ", ".join(cols)))

            # A class/label column is what tells signal from background, and
            # the production names it by CLASS rather than 0/1 -- print the
            # distinct values so the class names are read off, not guessed.
            for c in node.colnames:
                lc = c.lower()
                if lc in ("class", "label", "y", "subsample", "istrain",
                          "train", "type", "dataset", "category"):
                    v = node.col(c)
                    u, cnt = np.unique(v, return_counts=True)
                    print("  %-14s ->" % c, ", ".join(
                        "%s: %d" % (a, b) for a, b in list(zip(u, cnt))[:20]))

            # Run numbers: which detector-data runs really went in.
            for c in node.colnames:
                if c.lower() in ("run", "run_id", "runid"):
                    v = np.asarray(node.col(c))
                    u = np.unique(v)
                    print("  %-14s -> %d distinct" % (c, len(u)))
                    print("       %s" % (sorted(u.tolist())[:40],))

            if n:
                k = min(args.rows, n)
                print("  first %d rows:" % k)
                for r in node[:k]:
                    print("    %s" % (r,))
            print()
    finally:
        h5.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
