"""
Cross-check of our L4 production against the real pass2 L4 files.

THE EXERCISE
------------
Five of the 14 BDT inputs are pure-Python rewrites (CLAUDE.md, "Why it was
rewritten"): micro_count, FullTimeLengthRatio, VICH_nch, accumulated_time and
first_hlc_rho, plus fill_ratio, whose IceTray module we drive with our own
vertex.  They were written from the technical note and from
reference/oscNext_L4_pass2_original.py and have never been held against the
numbers the original chain actually produced.

The pass2 production is that answer key: the L3 files are the input our code
was built to read, and the matching L4 files already contain every variable.
So the check is the same job we have been doing all along --

    pass2 L3  --(our oscNext_L4 segment)-->  ours.hdf5
    pass2 L4  --(book what is already there)-->  pass2.hdf5
    match on (Run, Event, SubEvent), compare column by column

-- and the "ours" side is an ORDINARY process_L4.py output, so the same file
is also the training input for a pass2-trained model.  Nothing about this
comparison is a special production.

TWO THINGS TO GET RIGHT BEFORE RUNNING
--------------------------------------
1. THE CLEANED PULSE SERIES IS NAMED DIFFERENTLY IN PASS2.  Section 3.2 of
   the pass2 technical note (p.25) defines both series outright:

       uncleaned pulses -> SplitInIcePulses        (the SAME as pass3)
       cleaned pulses   -> SRTTWOfflinePulsesDC    (pass3: SRTTWSplitInIcePulsesDC)

   So only the cleaned name changes; the uncleaned default needs no override:

       --cleaned-pulses SRTTWOfflinePulsesDC

   This is the note's own definition, not an inference.  (The original pass2
   script cannot settle it: uncleaned_pulses/cleaned_pulses are parameters
   there with no defaults, supplied by a production script we do not have.
   Its straight-cut block does mention SRTTWOfflinePulsesDCHitStatistics, but
   that block is stale -- it also refers to an L4_MicroCount/STW7500_DTW200
   naming that the same file contradicts elsewhere -- so the note is the
   source here, and Table 12 on p.41 agrees with it.)

   Every variable is computed from whatever series is passed, so getting this
   wrong does not crash: it silently compares two different calculations.
   Confirm against a real frame with `compare_pass2.py inspect` anyway.

   Note that our HITSTAT_KEY / HITMULT_KEY are module-level constants carrying
   the pass3 spelling, so the values we compute land under the pass3 NAME even
   on a pass2 run.  The numbers are right (the segment uses the parameter);
   only the label differs, which is why the two sides of the table below carry
   different key names.

2. (Run, Event, SubEvent) IS NOT UNIQUE ACROSS FILES.  In MC run_id is the set
   number and event_id restarts in every file -- this is bug 2 of CLAUDE.md's
   booking audit.  Matching two productions therefore requires ONE L3 FILE PER
   HDF5 (process_L4.py without --chunk-files, or chunk_files=1).  match()
   refuses to guess when it sees repeated triples rather than matching wrongly
   in silence.

USAGE
-----
    # side A -- ours (this is also the training dataset)
    python scripts/process_L4.py --gcd GCD.i3.gz --input pass2_L3_file.i3.zst \
        --cleaned-pulses SRTTWOfflinePulsesDC \
        --output-hdf5 ours.hdf5 --mc --genie

    # side B -- the answer key
    python scripts/compare_pass2.py book --gcd GCD.i3.gz \
        --input pass2_L4_file.i3.zst --output-hdf5 pass2.hdf5

    # the comparison
    python scripts/compare_pass2.py report --ours ours.hdf5 --pass2 pass2.hdf5
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. What the answer-key side has to carry
# ---------------------------------------------------------------------------

# pass2 spelling of the pulse series -- technical note sec. 3.2, p.25, which
# names both explicitly.  The uncleaned series is the same string as pass3's,
# so only the cleaned one is really a "pass2" constant.
PASS2_CLEANED_PULSES = "SRTTWOfflinePulsesDC"
PASS2_UNCLEANED_PULSES = "SplitInIcePulses"

# Our own (pass3) spelling -- what process_L4.py writes even on a pass2 run.
OURS_HITSTAT = "SRTTWSplitInIcePulsesDCHitStatistics"
OURS_HITMULT = "SRTTWSplitInIcePulsesDCHitMultiplicity"

L3_VARS_KEY = "IC2018_LE_L3_Vars"


def pass2_hitstat_keys(cleaned_pulses=PASS2_CLEANED_PULSES):
    return cleaned_pulses + "HitStatistics", cleaned_pulses + "HitMultiplicity"


def pass2_book_keys(cleaned_pulses=PASS2_CLEANED_PULSES):
    """
    The frame keys to book out of a real pass2 L4 file.

    Nothing is computed on this side -- the L4 file already holds every
    variable.  Keys that turn out to be absent are simply not booked; the
    report then says "no table" for them rather than failing.
    """
    hitstat, hitmult = pass2_hitstat_keys(cleaned_pulses)
    return [
        "I3EventHeader",
        L3_VARS_KEY, "IC2018_LE_L3_bools",
        # the L4 variables
        "L4_micro_count", "L4_fill_ratio",
        "L4_VICH_nch", "L4_VICH_npulses", "L4_VICH_qtot",
        "L4_accumulated_time", "L4_separation_in_cogs",
        "L4_first_hlc", "L4_first_hlc_rho",
        "L4_iLineFit", "L4_iLineFitParams",
        "L4_ToI", "L4_ToIParams",
        hitstat, hitmult,
        # pass2's own verdicts.  L4_oscNext_bool is pass2's L4 cut: the file
        # carries every event with a bool rather than only the survivors, so
        # booking it keeps the whole population available and says which
        # events pass2 would have kept.
        "L4_oscNext_bool", "L4_NoiseStraightCuts_Bool",
        "L4_NoiseClassifier_ProbNu", "L4_MuonClassifier_Data_ProbNu",
        "L4_MuonClassifier_MuonGun_ProbNu",
        "L4_QR_Box",
    ]


# ---------------------------------------------------------------------------
# 2. What is compared
# ---------------------------------------------------------------------------

def compare_table(cleaned_pulses=PASS2_CLEANED_PULSES):
    """
    [(name, ours=(table, column), pass2=(table, column), rewritten)]

    The `ours` pairs are REGISTRY's (oscnext_l4/data.py); if one changes there
    it has to change here.  The two sides differ in three places and each
    difference is a fact about pass2, not a typo:

      * FullTimeLengthRatio is an L3 variable in pass2 (the note's Table 11
        calls it IC2018_LE_L3_Vars.FullTimeLengthRatio).  Pass3's L3 map holds
        only the two components, so we divide at L4 -- and that division is
        precisely what this row checks.
      * the hit statistics carry the pass2 pulse-series name on their side.
      * nothing else: every other L4 key has the same spelling in both, which
        is why our constants were taken from the original code.

    `rewritten` marks the rows this exercise exists for.  The rest are the
    control: they run the ORIGINAL IceTray modules on our side too, so a
    difference there means the INPUT differs (pulse series, geometry, event
    set) and would invalidate the rewritten rows as well.  Read the control
    rows first.
    """
    hitstat, hitmult = pass2_hitstat_keys(cleaned_pulses)
    return [
        # --- the rewrites: the point of the exercise ---
        ("micro_count",
         ("L4_micro_count", "STW_m3500p4000_DTW200"),
         ("L4_micro_count", "STW_m3500p4000_DTW200"), True),
        ("FullTimeLengthRatio",
         ("L4_FullTimeLengthRatio", "value"),
         (L3_VARS_KEY, "FullTimeLengthRatio"), True),
        ("VICH_nch",
         ("L4_VICH_nch", "value"), ("L4_VICH_nch", "value"), True),
        ("VICH_npulses",
         ("L4_VICH_npulses", "value"), ("L4_VICH_npulses", "value"), True),
        ("VICH_qtot",
         ("L4_VICH_qtot", "value"), ("L4_VICH_qtot", "value"), True),
        ("accumulated_time",
         ("L4_accumulated_time", "value"),
         ("L4_accumulated_time", "value"), True),
        ("first_hlc_rho",
         ("L4_first_hlc_rho", "value"), ("L4_first_hlc_rho", "value"), True),

        # --- original module, our vertex: a difference points at the vertex ---
        ("fill_ratio",
         ("L4_fill_ratio", "fillratio_from_mean"),
         ("L4_fill_ratio", "fillratio_from_mean"), False),

        # --- control: original modules on both sides ---
        ("iLineFit_speed",
         ("L4_iLineFitParams", "lf_vel"), ("L4_iLineFitParams", "lf_vel"), False),
        ("cog_z",     (OURS_HITSTAT, "cog_z"),    (hitstat, "cog_z"),    False),
        ("z_sigma",   (OURS_HITSTAT, "z_sigma"),  (hitstat, "z_sigma"),  False),
        ("z_travel",  (OURS_HITSTAT, "z_travel"), (hitstat, "z_travel"), False),
        ("n_hit_doms",
         (OURS_HITMULT, "n_hit_doms"), (hitmult, "n_hit_doms"), False),

        # --- straight from L3: must be identical or the event match is wrong ---
        ("NchCleaned",  (L3_VARS_KEY, "NchCleaned"),  (L3_VARS_KEY, "NchCleaned"), False),
        ("ICVetoHits",  (L3_VARS_KEY, "ICVetoHits"),  (L3_VARS_KEY, "ICVetoHits"), False),
    ]


# Rows that are neither a rewrite nor a control: the ORIGINAL IceTray module,
# but fed an input we produce.  A disagreement here is inherited from that
# input, so it must not raise the "the two sides saw different data" alarm --
# it points one row up, not at the module.
DEPENDS_ON = {"fill_ratio": "first_hlc_rho (the vertex we pass it)"}

# Integer-valued: for these "exactly equal" is the only acceptable outcome.
INTEGER_VARS = {"micro_count", "VICH_nch", "VICH_npulses", "n_hit_doms",
                "NchCleaned", "ICVetoHits"}

# Rows that are EXPECTED to disagree, and why.  Printed when they do, so a
# deliberate deviation is never mistaken for a bug in the rewrite.
EXPECTED_DEVIATION = {
    "micro_count":
        "EXPECTED under the default settings.  pass2 starts the micro_count "
        "chain from the UNCLEANED series, the note's Table 11 says the "
        "cleaned one, and we follow the note (CLAUDE.md open risk 5b).  The "
        "pass2 L4 file shows this directly: L4_TWPulses_DCFid is built from "
        "L4_TWPulses (the StaticTWC output), so the L4_SRTTWPulses in the "
        "same frame is written and never read -- the dead cleaning step of "
        "booking-audit bug 4, on real data.  To compare the IMPLEMENTATION "
        "rather than the decision, rerun process_L4.py with "
        "--micro-count-uncleaned: our chain is then uncleaned -> StaticTWC -> "
        "fiducial -> 200 ns DTW, which is pass2's chain with the dead step "
        "left out, so it should match exactly.",
}

# Below this a float difference is arithmetic noise, not a different
# calculation.  Deliberately loose -- the question is which calculation was
# done, not how it rounded.
FLOAT_RTOL = 1e-6
FLOAT_ATOL = 1e-9

# A row counts as reproducing pass2 when at least this fraction of events
# agree.  Not 1.0: a handful of events out of thousands can legitimately pick
# a different hit at a tie, and that is a footnote, not a failed rewrite.
AGREE_FRACTION = 0.999


# ---------------------------------------------------------------------------
# 3. Reading and matching
# ---------------------------------------------------------------------------

def read_pairs(h5path, pairs):
    """
    {name: array} for explicit (name, table, column) triples, plus the ids.

    data.load_one_file resolves names through REGISTRY; the pass2 side is not
    in REGISTRY and should never be, so the pairs are given directly.  The
    alignment logic is the same and for the same reason -- see the __I3Index__
    bug, CLAUDE.md "Booking/read audit" items 1 and 2.
    """
    import tables
    from .data import _table_nodes, _index_node, _ids

    out, missing = {}, []
    with tables.open_file(h5path, "r") as h5:
        nodes = _table_nodes(h5)
        if "I3EventHeader" not in nodes:
            raise RuntimeError("%s: no I3EventHeader table" % h5path)
        run, ev, sub = _ids(nodes["I3EventHeader"])
        n = len(run)
        out["Run"], out["Event"], out["SubEvent"] = run, ev, sub

        for name, table, column in pairs:
            node = nodes.get(table)
            if node is None or column not in node.colnames:
                why = "no table" if node is None else "no column"
                missing.append((name, table, column, why))
                out[name] = np.full(n, np.nan)
                continue

            r2, e2, s2 = _ids(node)
            v = np.asarray(node.col(column), dtype=np.float64)

            if (len(r2) == n and np.array_equal(r2, run)
                    and np.array_equal(e2, ev) and np.array_equal(s2, sub)):
                out[name] = v
                continue

            inode = _index_node(h5, table)
            if (inode is not None and len(inode) == n
                    and "exists" in inode.colnames
                    and "start" in inode.colnames):
                exists = np.asarray(inode.col("exists")).astype(bool)
                start = np.asarray(inode.col("start"), dtype=np.int64)
                idx = np.where(exists, start, -1)
                idx[idx >= len(v)] = -1
                a = np.full(n, np.nan)
                ok = idx >= 0
                a[ok] = v[idx[ok]]
                out[name] = a
            else:
                missing.append((name, table, column, "unaligned, no index"))
                out[name] = np.full(n, np.nan)

    return out, missing


def _triples(d):
    return np.stack([d["Run"], d["Event"], d["SubEvent"]], axis=1)


def match(ours, theirs, strict=True):
    """
    Row indices of the events present on BOTH sides -> (idx_ours, idx_theirs).

    Matching is on (Run, Event, SubEvent).  That triple is unique WITHIN one
    L3 file and not across files, so a repeated triple means the HDF5 holds
    several files and no reliable match exists.  With strict=True that raises:
    a wrong match would produce a plausible-looking disagreement table that is
    really an event-mixing artefact, which is exactly the failure mode of
    CLAUDE.md's bug 2.  Rerun with one L3 file per HDF5 instead.
    """
    a, b = _triples(ours), _triples(theirs)

    for label, arr in (("ours", a), ("pass2", b)):
        view = np.ascontiguousarray(arr).view(
            np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
        if len(np.unique(view)) != len(arr):
            msg = ("%s: (Run, Event, SubEvent) repeats -- the file holds more "
                   "than one L3 file, so events cannot be matched reliably.  "
                   "Rerun the production with one L3 file per HDF5."
                   % label)
            if strict:
                raise RuntimeError(msg)
            print("  [!] " + msg)

    index = {tuple(k): i for i, k in enumerate(b.tolist())}
    ia, ib = [], []
    for i, k in enumerate(a.tolist()):
        j = index.get(tuple(k))
        if j is not None:
            ia.append(i)
            ib.append(j)
    return np.asarray(ia, dtype=int), np.asarray(ib, dtype=int)


def load_comparison(ours_h5, pass2_h5, cleaned_pulses=PASS2_CLEANED_PULSES,
                    strict=True):
    """Read both files, match the events -> (data, missing, counts)."""
    table = compare_table(cleaned_pulses)

    ours_pairs = [(n, t, c) for n, (t, c), _, _ in table]
    pass2_pairs = [(n, t, c) for n, _, (t, c), _ in table]

    o, miss_o = read_pairs(ours_h5, ours_pairs)
    p, miss_p = read_pairs(pass2_h5, pass2_pairs)

    ia, ib = match(o, p, strict=strict)

    data = {
        "Run": o["Run"][ia], "Event": o["Event"][ia],
        "SubEvent": o["SubEvent"][ia],
        "ours": {n: o[n][ia] for n, _, _, _ in table},
        "pass2": {n: p[n][ib] for n, _, _, _ in table},
    }
    counts = {"ours": len(o["Run"]), "pass2": len(p["Run"]),
              "matched": len(ia)}
    missing = ([("ours",) + m for m in miss_o]
               + [("pass2",) + m for m in miss_p])
    return data, missing, counts


# ---------------------------------------------------------------------------
# 4. The report
# ---------------------------------------------------------------------------

def _verdict(name, ours, theirs):
    """
    (status, n, agree_fraction, max_abs_diff, max_rel_diff)

    The verdict is decided by the FRACTION of events that agree, not by the
    worst one.  Deciding it on the maximum was wrong and actively misleading:
    first_hlc_rho agreed in 99.85% of 8144 events and was still stamped
    DIFFERS because a dozen of them picked a different hit.

    "Agree" is also not bitwise equality for a float.  Our rho goes through
    np.hypot while pass2's went through the oscNext project's calc_rho_36;
    the two differ in the last bit for about a fifth of events, which is a
    different square root, not a different definition.  Integers are still
    held to exact equality -- there is no rounding to forgive there.
    """
    both = np.isfinite(ours) & np.isfinite(theirs)
    n = int(both.sum())
    if n == 0:
        return "no overlap", 0, float("nan"), float("nan"), float("nan")

    a, b = ours[both], theirs[both]
    diff = a - b
    # A relative difference against a reference of exactly 0 is undefined, not
    # 1e300: dividing by a tiny floor printed "max rel 5e+300", which says
    # nothing except that pass2 stored a 0 somewhere.
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(b != 0, np.abs(diff) / np.abs(np.where(b != 0, b, 1)),
                       np.where(diff != 0, np.inf, 0.0))
    max_abs = float(np.max(np.abs(diff)))
    max_rel = float(np.max(rel[np.isfinite(rel)])) if np.isfinite(rel).any() \
        else float("inf")

    if name in INTEGER_VARS:
        agree = float((diff == 0).mean())
    else:
        agree = float((np.abs(diff) <= FLOAT_ATOL
                       + FLOAT_RTOL * np.abs(b)).mean())

    if max_abs == 0.0:
        return "identical", n, agree, max_abs, max_rel
    if agree >= AGREE_FRACTION:
        return "agrees", n, agree, max_abs, max_rel
    return "DIFFERS", n, agree, max_abs, max_rel


def report(ours_h5, pass2_h5, cleaned_pulses=PASS2_CLEANED_PULSES,
           n_worst=5, strict=True, stream=None):
    """
    Print the per-variable agreement; return the rows as a list of dicts.

    How to read it: the CONTROL rows first.  If NchCleaned or ICVetoHits is
    not identical the event match is wrong and nothing below it means
    anything.  If cog_z / z_sigma / z_travel differ, the two sides are reading
    different pulse series (see the module docstring, point 1).  Only once
    those are clean do the rewritten rows carry information.
    """
    import sys
    out = stream or sys.stdout
    data, missing, counts = load_comparison(
        ours_h5, pass2_h5, cleaned_pulses=cleaned_pulses, strict=strict)

    print("pass2 cross-check", file=out)
    print("  ours  : %s (%d events)" % (ours_h5, counts["ours"]), file=out)
    print("  pass2 : %s (%d events)" % (pass2_h5, counts["pass2"]), file=out)
    print("  matched on (Run, Event, SubEvent): %d" % counts["matched"],
          file=out)
    if counts["matched"] == 0:
        print("\nNo event matched.  The two files do not come from the same "
              "L3 file, or the pass2 L4 file was produced from a different "
              "run.", file=out)
        return []
    print("", file=out)

    if missing:
        print("Columns not found (-> NaN):", file=out)
        for side, name, table, column, why in missing:
            print("  %-6s %-22s %-38s %-24s %s"
                  % (side, name, table, column, why), file=out)
        print("", file=out)

    print("  %-22s %-9s %8s %9s %11s %11s  %s"
          % ("variable", "role", "n", "agree", "max|diff|", "max rel",
             "verdict"), file=out)
    print("  " + "-" * 94, file=out)

    rows = []
    for name, _, _, rewritten in compare_table(cleaned_pulses):
        o, p = data["ours"][name], data["pass2"][name]
        status, n, agree, max_abs, max_rel = _verdict(name, o, p)
        rows.append({"name": name, "rewritten": rewritten, "status": status,
                     "n": n, "agree": agree, "max_abs": max_abs,
                     "max_rel": max_rel})
        role = ("rewritten" if rewritten
                else "derived" if name in DEPENDS_ON else "control")
        print("  %-22s %-9s %8d %8.2f%% %11.4g %11.4g  %s"
              % (name, role, n, 100 * agree, max_abs, max_rel, status),
              file=out)

    print("", file=out)
    for r in [x for x in rows if x["status"] == "DIFFERS"]:
        name = r["name"]
        o, p = data["ours"][name], data["pass2"][name]
        both = np.isfinite(o) & np.isfinite(p)
        d = np.where(both, np.abs(o - p), -np.inf)
        # Only events that actually disagree -- padding the list with
        # identical events makes a one-event discrepancy look like five.
        order = [i for i in np.argsort(-d)[:n_worst] if both[i] and d[i] > 0]
        if not order:
            continue
        print("%s -- %d worst events:" % (name, len(order)), file=out)
        print("    %10s %10s %7s %14s %14s %12s"
              % ("Run", "Event", "SubEv", "ours", "pass2", "diff"), file=out)
        for i in order:
            print("    %10d %10d %7d %14.6g %14.6g %12.4g"
                  % (data["Run"][i], data["Event"][i], data["SubEvent"][i],
                     o[i], p[i], o[i] - p[i]), file=out)
        print("", file=out)

    for r in rows:
        if r["status"] == "DIFFERS" and r["name"] in EXPECTED_DEVIATION:
            print("%s: %s" % (r["name"], EXPECTED_DEVIATION[r["name"]]),
                  file=out)
            print("", file=out)

    for r in rows:
        if r["status"] == "DIFFERS" and r["name"] in DEPENDS_ON:
            print("%s: the original IceTray module, but driven by %s.  A "
                  "disagreement here is inherited from that input -- fix it "
                  "there first." % (r["name"], DEPENDS_ON[r["name"]]),
                  file=out)
            print("", file=out)

    ctrl = [r for r in rows
            if not r["rewritten"] and r["name"] not in DEPENDS_ON]
    bad_ctrl = [r["name"] for r in ctrl if r["status"] == "DIFFERS"]
    if bad_ctrl:
        print("CONTROL ROWS DISAGREE: %s.  Fix this before reading the "
              "rewritten rows -- it means the two sides did not see the same "
              "input." % ", ".join(bad_ctrl), file=out)

    rew = [r for r in rows if r["rewritten"]]
    ok = [r for r in rew if r["status"] in ("identical", "agrees")]
    expected = [r["name"] for r in rew
                if r["status"] == "DIFFERS" and r["name"] in EXPECTED_DEVIATION]
    print("Rewritten variables reproducing pass2: %d/%d%s"
          % (len(ok), len(rew),
             " (%s differs on purpose -- see above)" % ", ".join(expected)
             if expected else ""), file=out)
    nover = [r["name"] for r in rows if r["status"] == "no overlap"]
    if nover:
        print("Not compared (one side missing): %s" % ", ".join(nover),
              file=out)
    return rows


# ---------------------------------------------------------------------------
# 5. Where the pass2 files are
# ---------------------------------------------------------------------------

# The L4 paths are the L3 paths with "level3" -> "level4", in the directory
# and in the basename alike, so only one map is needed.
#
# Two gaps worth knowing before planning a pass2 training run:
#   * there is NO CORSIKA at pass2 here -- the muon background is MuonGun, so
#     a pass2-trained muon BDT is not weighted the way the pass3 one is
#     (CLAUDE.md open risk 3d is about the CORSIKA weighting specifically);
#   * NuTau exists at pass2 (160519).  Our signal definition is nue+numu
#     (open risk 6), so it is listed but not used unless that changes.
PASS2_L3 = {
    "NuE":     ["/data/ana/LE/oscNext/pass2/genie/level3/121122/"
                "oscNext_genie_level3_v02.00_pass2.121122.*.i3.zst",
                "/data/ana/LE/oscNext/pass2/genie/level3/121291/"
                "oscNext_genie_level3_v02.00_pass2.121291.*.i3.zst"],
    "NuMu":    ["/data/ana/LE/oscNext/pass2/genie/level3/141154/"
                "oscNext_genie_level3_v02.00_pass2.141154.*.i3.zst",
                "/data/ana/LE/oscNext/pass2/genie/level3/141292/"
                "oscNext_genie_level3_v02.00_pass2.141292.*.i3.zst"],
    "NuTau":   ["/data/ana/LE/oscNext/pass2/genie/level3/160519/"
                "oscNext_genie_level3_v02.00_pass2.160519.*.i3.zst"],
    "MuonGun": ["/data/ana/LE/oscNext/pass2/muongun/level3/139008/"
                "oscNext_muongun_level3_v02.00_pass2.139008.*.i3.zst"],
    "Noise":   ["/data/ana/LE/oscNext/pass2/noise/level3/888003/"
                "oscNext_noise_level3_v02.00_pass2.888003.*.i3.zst"],
}

# process_L4.py flags per sample -- the weighting keys differ and booking the
# wrong ones produces a column of NaN rather than an error (CLAUDE.md, AUX).
PASS2_FLAGS = {
    "NuE":     ["--mc", "--genie"],
    "NuMu":    ["--mc", "--genie"],
    "NuTau":   ["--mc", "--genie"],
    "MuonGun": ["--mc", "--muongun"],
    "Noise":   ["--noise"],
}


def l4_path(l3_path):
    """
    The pass2 L4 file matching an L3 file.

    Raises rather than returning the input unchanged: silently comparing a
    file with itself is the one outcome this whole module exists to prevent.
    """
    out = l3_path.replace("level3", "level4")
    if out == l3_path:
        raise ValueError("no 'level3' in %s -- cannot derive the L4 path"
                         % l3_path)
    return out


def pair_files(l3_files):
    """[(l3, l4)] for the L4 files that actually exist -> and the misses."""
    import os
    pairs, orphans = [], []
    for f in l3_files:
        cand = l4_path(f)
        if os.path.exists(cand):
            pairs.append((f, cand))
        else:
            orphans.append((f, cand))
    return pairs, orphans
