#!/usr/bin/env python
"""Uretilen HDF5'teki tam sutun isimlerini dok -- feature registry'yi hizalamak icin."""
import sys, tables

path = sys.argv[1] if len(sys.argv) > 1 else "L4_output/hdf5/genie/nue/L4_nue_00000.hdf5"

WANT = ["L4_iLineFitParams", "L4_fill_ratio", "L4_ToIParams",
        "SRTTWSplitInIcePulsesDCHitStatistics",
        "SRTTWSplitInIcePulsesDCHitMultiplicity",
        "IC2018_LE_L3_Vars", "L4_micro_count", "L4_first_hlc"]

with tables.open_file(path, "r") as h5:
    have = {n.name: n for n in h5.walk_nodes("/", "Table")}
    for t in WANT:
        node = have.get(t)
        print("=" * 70)
        print(t, "  (satir: %d)" % node.nrows if node else "  <TABLO YOK>")
        if node is None:
            continue
        cols = [c for c in node.colnames
                if c not in ("Run", "Event", "SubEvent", "SubEventStream", "exists")]
        for c in cols:
            print("   ", c)
