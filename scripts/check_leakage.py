#!/usr/bin/env python
'''
Train/test leakage check on a .npz training set.

WHY THIS EXISTS
  The split is drawn per EVENT (`rng.random(n) < TRAIN_FRAC`).  That is only
  correct if every row is an independent event.  Three things can break it,
  and none of them raises -- the model simply scores better than it should:

  1. DUPLICATED ROWS.  A stale HDF5 part from an earlier run with a different
     `jobs`/`chunk_files` is still matched by the `L4_<sample>*.hdf5` glob, so
     the same L3 file is loaded twice.  A retry after a corrupt file that
     failed to delete its partial output does the same.  Both put an exact
     copy of an event on each side of the split.
  2. SUB-EVENTS.  One DAQ event can split into several P frames, which share
     the neutrino interaction and part of the pulse series.  They are separate
     rows with the same (Run, Event) and different SubEvent, so an event-level
     split can put correlated rows on both sides.
  3. OVERSAMPLING.  One CORSIKA air shower is reused `OverSampling` times.
     `split_by_shower` handles that for the muon background; nothing else in
     the pipeline splits by anything but the row.

  Case 1 is detectable from the .npz alone: an exact duplicate of all feature
  values is, to a very good approximation, the same event twice.  Cases 2 and
  3 need Run/Event/SubEvent, which build_dataset does not store -- for those
  this script says what to check upstream instead of guessing.

USAGE
    python scripts/check_leakage.py /path/to/L4_noise_dataset.npz

numpy only, so it runs in the IceTray environment.
'''
from __future__ import print_function

import sys
import numpy as np


def bytes_key(cols):
    """One bytewise key per row, so identical rows collapse and NaN matches NaN."""
    m = np.ascontiguousarray(np.column_stack([np.asarray(c, dtype=np.float64)
                                              for c in cols]))
    return m.view([("", m.dtype)] * m.shape[1]).ravel()


def straddle_report(key, istrain, label, title):
    """Duplicate groups, and how many of them cross the train/test boundary."""
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    n_groups = len(uniq)
    n_tr = np.bincount(inv, weights=istrain.astype(np.float64),
                       minlength=n_groups)
    dup = cnt > 1
    cross = dup & (n_tr > 0) & (n_tr < cnt)

    n_rows = len(key)
    rows_in_dup = int(cnt[dup].sum())
    rows_in_cross = int(cnt[cross].sum())

    print("\n--- %s ---" % title)
    print("  rows                         %12d" % n_rows)
    print("  distinct rows                %12d" % n_groups)
    print("  rows in a duplicate group    %12d  (%.3f%%)"
          % (rows_in_dup, 100.0 * rows_in_dup / max(n_rows, 1)))
    print("  rows in a group that CROSSES %12d  (%.3f%%)   <-- leakage"
          % (rows_in_cross, 100.0 * rows_in_cross / max(n_rows, 1)))
    if rows_in_cross:
        sizes = cnt[cross]
        print("      %d crossing groups, sizes %d..%d (median %d)"
              % (cross.sum(), sizes.min(), sizes.max(), int(np.median(sizes))))
        # which class
        first = np.flatnonzero(cross)
        of_sig = 0
        for g in first[:2000]:
            of_sig += int(label[np.flatnonzero(inv == g)[0]] == 1)
        print("      of the first %d crossing groups, %d are signal"
              % (min(len(first), 2000), of_sig))
    return rows_in_cross


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    d = np.load(path, allow_pickle=False)
    features = [str(f) for f in d["features"]]
    istrain = d["istrain"].astype(bool)
    label = d["label"]

    print("dataset : %s" % path)
    print("features: %s" % ", ".join(features))
    print("rows    : %d   (train %d / test %d)"
          % (len(istrain), int(istrain.sum()), int((~istrain).sum())))
    print("          signal %d / background %d"
          % (int((label == 1).sum()), int((label == 0).sum())))

    # Include the label in the key: a signal and a background row that happen
    # to share all five values are not the same event, and counting them as a
    # duplicate would overstate the problem.
    cols = [d[f] for f in features] + [label]
    n_cross = straddle_report(bytes_key(cols), istrain, label,
                              "All features (exact, bitwise)")

    # An integer-valued feature collides by chance; the continuous ones do not.
    # Re-run on the continuous subset as a cross-check: if the two numbers
    # agree the duplicates are real, if only the full-key number is large the
    # collisions are coincidental.
    cont = [f for f in features
            if not np.array_equal(d[f], np.round(d[f]))]
    if cont and len(cont) < len(features):
        print("\n  (continuous features: %s)" % ", ".join(cont))
        straddle_report(bytes_key([d[f] for f in cont] + [label]),
                        istrain, label, "Continuous features only")

    print("\n--- VERDICT ---")
    if n_cross == 0:
        print("  No duplicated event crosses the split.  Mechanism 1 is clear.")
    else:
        print("  DUPLICATED EVENTS CROSS THE SPLIT.  Most likely a stale HDF5")
        print("  part: check that each L3 file is loaded exactly once --")
        print("    ls -la <HDF_BASE>/<sample>/    and compare the job/part")
        print("    numbering with the `jobs` used in the LAST run.")
    print("\n  Mechanisms this file CANNOT see (Run/Event/SubEvent are not")
    print("  stored in the .npz):")
    print("    - sub-events of one DAQ event on both sides of the split")
    print("    - oversampling (CORSIKA); the muon set uses split_by_shower")
    print("  To check those, read SubEvent from the HDF5 and count events with")
    print("  more than one sub-event that passes the L3 cut.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
