"""
L4 HDF5 -> numpy: the feature registry, loading, and weights.

WHY A SEPARATE FILE: notebook output cells churn on every run; with the logic
here it stays versioned and updates arrive through `git pull`.

USAGE (notebook):

    from oscnext_l4.data import (REGISTRY, AUX, NOISE_FEATURES, MUON_FEATURES,
                                 WANTED, dump_tables, check_registry,
                                 check_feature_map, load_sample, add_weights)

    TABLES = dump_tables(smoke_hdf5)
    check_registry(TABLES, NOISE_FEATURES)
    check_feature_map()

    data = {n: load_sample(n, SAMPLES, WANTED) for n in SAMPLES}
    data = {n: d for n, d in data.items() if d is not None}
    add_weights(data, SAMPLES)

CRITICAL SYNCHRONISATION POINT: the REGISTRY here and FEATURE_MAP in
classifier.py must agree line for line.  If one reads a different column than
the other, the model produces wrong predictions SILENTLY -- it does not raise.
check_feature_map() checks this in code.
"""

import os
import glob
import json

import numpy as np
import tables


IDX = ("Run", "Event", "SubEvent")


# BDT variable name -> (HDF5 table, column)
L3V     = "IC2018_LE_L3_Vars"
HITSTAT = "SRTTWSplitInIcePulsesDCHitStatistics"
HITMULT = "SRTTWSplitInIcePulsesDCHitMultiplicity"

REGISTRY = {
    # --- noise BDT (Table 11) ---
    "NchCleaned":          (L3V, "NchCleaned"),
    "micro_count":         ("L4_micro_count", "STW_m3500p4000_DTW200"),
    # The pass3 hdfwriter column is "lf_vel" (classifier.FEATURE_MAP agrees);
    # "LFVel" was the old assumption.  Table 11 of the note names the variable
    # "L4_iLineFit.speed" (an I3Particle field).  All three are in ALTS.
    "iLineFit_speed":      ("L4_iLineFitParams", "lf_vel"),
    # VERIFIED against real pass3 HDF5 output: the column is
    # "fillratio_from_mean" -- no underscore.  Table 11 writes
    # "fill_ratio_from_mean"; the hdfwriter converter names it differently.
    # Both are in ALTS.
    "fill_ratio":          ("L4_fill_ratio", "fillratio_from_mean"),
    "FullTimeLengthRatio": ("L4_FullTimeLengthRatio", "value"),

    # --- muon BDT (Table 12) ---
    "ICVetoHits":       (L3V, "ICVetoHits"),
    "RTVeto250Hits":    (L3V, "RTVeto250Hits"),
    "NAbove200Hits":    (L3V, "NAbove200Hits"),
    "VICH_nch":         ("L4_VICH_nch", "value"),
    "accumulated_time": ("L4_accumulated_time", "value"),
    "first_hlc_rho":    ("L4_first_hlc_rho", "value"),
    "cog_z":            (HITSTAT, "cog_z"),
    "z_sigma":          (HITSTAT, "z_sigma"),
    "z_travel":         (HITSTAT, "z_travel"),

    # --- candidates / for scans ---
    "n_hit_doms":       (HITMULT, "n_hit_doms"),
    "C2HR6":            (L3V, "C2HR6"),
    "CausalVetoHits":   (L3V, "CausalVetoHits"),
    "VertexGuessZ":     (L3V, "VertexGuessZ"),
    "DCFiducialHits":   (L3V, "DCFiducialHits"),
}

NOISE_FEATURES = ["NchCleaned", "micro_count", "iLineFit_speed",
                  "fill_ratio", "FullTimeLengthRatio"]

# Table 12 lists 10 variables.  NchCleaned is an input to BOTH the noise and
# the muon BDT -- it was missing from the muon list in the first version.
MUON_FEATURES = ["ICVetoHits", "RTVeto250Hits", "NchCleaned", "NAbove200Hits",
                 "VICH_nch",
                 "accumulated_time", "first_hlc_rho", "cog_z", "z_sigma",
                 "z_travel"]

# Extra columns needed for the weight calculation (NOT BDT inputs)
#
# CAREFUL: not all of these exist in every sample.  noise_weight is only in
# the vuvuzela noise MC, OneWeight/PrimaryNeutrino* only in GENIE.  Checking
# all of them against a single sample produces FALSE ALARMS -- select the
# relevant ones with aux_for(kind).
AUX = {
    "true_energy":   ("I3MCWeightDict", "PrimaryNeutrinoEnergy"),
    "OneWeight":     ("I3MCWeightDict", "OneWeight"),
    "NEvents":       ("I3MCWeightDict", "NEvents"),
    "pdg":           ("I3MCWeightDict", "PrimaryNeutrinoType"),
    "n_flux_events": ("L4_n_flux_events", "value"),
    # Verified in the earlier LightGBM notebook: the column is "weight".
    # "value" was a wrong assumption and left the noise weight entirely NaN.
    # Both are in ALTS.
    "noise_weight":  ("noise_weight", "weight"),
    # CORSIKA -- for the manual formula when simweights is absent
    "cwm_Weight":       ("CorsikaWeightMap", "Weight"),
    "cwm_NEvents":      ("CorsikaWeightMap", "NEvents"),
    "cwm_OverSampling": ("CorsikaWeightMap", "OverSampling"),
    # MuonGun (the pass2 muon background).  process_L4.py books all three
    # spellings because which one a production wrote is not fixed; the first
    # that is actually present is used.
    "MuonWeight":           ("MuonWeight", "value"),
    "MuonWeight_GaisserH4a": ("MuonWeight_GaisserH4a", "value"),
    "MuonGunWeight":        ("MuonGunWeight", "value"),
}

# AUX column -> which WEIGHT SCHEMES need it (SAMPLES[...]["weight"]).
#
# Keyed by scheme rather than by `kind`, because one kind can have several
# schemes: pass3's muon_bg is CORSIKA and pass2's is MuonGun, and asking a
# MuonGun file for CorsikaWeightMap prints three false alarms per file.
AUX_SCHEMES = {
    "true_energy":   ("genie",),
    "OneWeight":     ("genie",),
    "NEvents":       ("genie",),
    "pdg":           ("genie",),
    "n_flux_events": ("genie",),
    "noise_weight":  ("noise",),
    "cwm_Weight":       ("corsika",),
    "cwm_NEvents":      ("corsika",),
    "cwm_OverSampling": ("corsika",),
    "MuonWeight":           ("muongun",),
    "MuonWeight_GaisserH4a": ("muongun",),
    "MuonGunWeight":        ("muongun",),
}

# Fallback for a sample spec that predates the `weight` field.
_KIND_TO_SCHEME = {"signal": "genie", "noise_bg": "noise", "muon_bg": "corsika"}


def scheme_of(cfg):
    """The weight scheme of one sample spec, falling back to its `kind`."""
    return cfg.get("weight") or _KIND_TO_SCHEME.get(cfg.get("kind"))


def aux_for(scheme):
    """The AUX columns EXPECTED under this weight scheme."""
    return [k for k, schemes in AUX_SCHEMES.items() if scheme in schemes]


def _table_nodes(h5):
    """
    The REAL data tables in the file -> {name: node}.

    WHY A SEPARATE FUNCTION: h5.walk_nodes("/", "Table") DESCENDS INTO
    SUBGROUPS TOO.  hdfwriter writes two tables per key:

        /L4_VICH_nch                 <- the data   (Run, Event, ..., value)
        /__I3Index__/L4_VICH_nch     <- the index  (exists, start, stop)

    Keyed by leaf name the two COLLIDE and the index table overwrote the data:
    every table then appeared to have "start/stop" columns, the wanted columns
    were not found, and EVERY VARIABLE CAME OUT NaN.

    Fix: skip everything under __I3Index__, and when a name appears twice pick
    the one CLOSEST TO THE ROOT.
    """
    best = {}
    for node in h5.walk_nodes("/", "Table"):
        path = node._v_pathname
        if "__I3Index__" in path:
            continue
        depth = path.count("/")
        if node.name not in best or depth < best[node.name][0]:
            best[node.name] = (depth, node)
    return {k: v[1] for k, v in best.items()}


def _index_node(h5, name):
    """
hdfwriter's /__I3Index__/<key> table, or None when absent.

    It holds one row per FRAME: exists (is the key present in this frame) and
    start/stop (the row range in the data table).  This is the CORRECT way to
    match.  Matching on Run/Event/SubEvent is unreliable in MC because the
    triples can repeat within one part (run_id is the set number and event_id
    restarts from zero in every L3 file).
    """
    try:
        return h5.get_node("/__I3Index__/" + name)
    except Exception:
        return None


def dump_tables(h5path, only=None, max_cols=40):
    """List the tables in an HDF5 file and their columns."""
    if not os.path.exists(h5path):
        print("MISSING:", h5path); return {}
    found = {}
    with tables.open_file(h5path, "r") as h5:
        for node in _table_nodes(h5).values():
            cols = [c for c in node.colnames
                    if c not in ("Run", "Event", "SubEvent", "SubEventStream", "exists")]
            found[node.name] = (node.nrows, cols)
    for name in sorted(found):
        nrows, cols = found[name]
        if only and not any(o in name for o in only):
            continue
        print("%-45s %8d rows" % (name, nrows))
        for c in cols[:max_cols]:
            print("      %s" % c)
        if len(cols) > max_cols:
            print("      ... (+%d columns)" % (len(cols) - max_cols))
    return found

def check_registry(found, names, verbose=True):
    """
    Do the registry's columns actually exist in the HDF5 file?

    ALTS is tried too, so whichever variant loading will use is the one
    reported here.  (It used to consult REGISTRY only, and said "column
    missing" even when ALTS resolved a variant fine.)

    found: the dump_tables output, {table: (nrows, [columns])}
    Returns the list of the ones not found.
    """
    available = {t: set(cols) for t, (_, cols) in found.items()}
    ok, bad, alt_used = [], [], []
    for n in names:
        hit = resolve_one(n, available)
        if hit is None:
            bad.append((n, REGISTRY.get(n, AUX.get(n))))
            continue
        ok.append(n)
        first = (ALTS.get(n) or [REGISTRY.get(n, AUX.get(n))])[0]
        if tuple(hit) != tuple(first):
            alt_used.append((n, hit, first))

    if verbose:
        print("found: %d/%d" % (len(ok), len(names)))
        for n, hit, first in alt_used:
            print("  [i] %-22s %s[%s]  (first candidate %s[%s] was absent)"
                  % (n, hit[0], hit[1], first[0], first[1]))
        for n, pair in bad:
            tbl = pair[0] if pair else "?"
            why = "no such table" if tbl not in available else "no such column"
            print("  [!] %-22s %s  -- %s" % (n, pair, why))
            if tbl in available:
                cols = sorted(available[tbl])
                print("      columns in %s: %s"
                      % (tbl, ", ".join(cols[:12]) + (" ..." if len(cols) > 12 else "")))
    return bad


def _ids(tbl):
    return (np.asarray(tbl.col("Run"), dtype=np.int64),
            np.asarray(tbl.col("Event"), dtype=np.int64),
            np.asarray(tbl.col("SubEvent"), dtype=np.int64))


def _has_duplicate_ids(r, e, s):
    """Are the triples unique?  (quick check)"""
    if len(r) == 0:
        return False
    arr = np.empty(len(r), dtype=[("r", np.int64), ("e", np.int64), ("s", np.int64)])
    arr["r"], arr["e"], arr["s"] = r, e, s
    return len(np.unique(arr)) < len(arr)


# Name variations: column names differ between meta-project and pass versions.
# Whichever one is ACTUALLY in the file is used; if none is, NaN plus a warning.
ALTS = {
    "iLineFit_speed": [("L4_iLineFitParams", "lf_vel"),
                       ("L4_iLineFitParams", "LFVel"),
                       ("L4_iLineFit", "speed")],       # the name used in Table 11
    "micro_count":    [("L4_micro_count", "STW_m3500p4000_DTW200"),
                       ("L4_micro_count", "STW7500_DTW200")],
    # Table 11 says "L4_fill_ratio.fill_ratio_from_mean".  hdfwriter's
    # I3FillRatioInfo converter names it differently across versions.
    # EVERY ENTRY BELOW IS THE SAME QUANTITY (fill ratio about the mean) --
    # DIFFERENT quantities such as from_rms / from_nch are deliberately left
    # out, otherwise a different piece of physics would be read silently.
    "noise_weight":   [("noise_weight", "weight"),
                       ("noise_weight", "value")],
    "fill_ratio":     [("L4_fill_ratio", "fillratio_from_mean"),
                       ("L4_fill_ratio", "fill_ratio_from_mean"),
                       ("L4_fill_ratio", "fillRatioFromMean"),
                       ("L4_fill_ratio", "FillRatioFromMean")],
}

_warned_unresolved = set()


def _sample_tag(path):
    """L4_nue_job3_part002.hdf5 -> 'L4_nue'  (to warn once per sample)"""
    b = os.path.basename(path)
    for sep in ("_job", "_part"):
        if sep in b:
            b = b.split(sep)[0]
    return b


def resolve_one(name, available):
    """
    Pick the (table, column) pair that ACTUALLY exists in the file.

    available: {table_name: set(column_names)}
    Returns None when nothing matches.
    """
    cands = ALTS.get(name)
    if cands is None:
        pair = REGISTRY.get(name, AUX.get(name))
        cands = [pair] if pair else []
    for tbl, col in cands:
        if tbl in available and col in available[tbl]:
            return tbl, col
    return None


def load_one_file(path, wanted):
    """Read the wanted variables out of one HDF5 file -> {name: array}."""
    with tables.open_file(path, "r") as h5:
        available = {k: set(n.colnames) for k, n in _table_nodes(h5).items()}

    need, unresolved = {}, []
    for name in wanted:
        hit = resolve_one(name, available)
        if hit:
            need.setdefault(hit[0], []).append((name, hit[1]))
        else:
            unresolved.append(name)
            # Unresolved variable -> NaN.  Warn once per SAMPLE, not per
            # file (with 16 parts it used to print 16 times).
            key = (_sample_tag(path), name)
            if key not in _warned_unresolved:
                _warned_unresolved.add(key)
                pair = REGISTRY.get(name, AUX.get(name))
                tbl = pair[0] if pair else "?"
                if tbl not in available:
                    print("  [!] %-20s NO SUCH TABLE: %s -> NaN" % (name, tbl))
                else:
                    cols = sorted(available[tbl])
                    print("  [!] %-20s no column '%s' in %s -> NaN"
                          % (name, pair[1], tbl))
                    print("      columns present: %s"
                          % (", ".join(cols[:10]) + (" ..." if len(cols) > 10 else "")))

    with tables.open_file(path, "r") as h5:
        nodes = _table_nodes(h5)
        if "I3EventHeader" not in nodes:
            raise RuntimeError("%s: no I3EventHeader table" % path)

        ref = nodes["I3EventHeader"]
        run, ev, sub = _ids(ref)
        n = len(run)
        out = {"Run": run, "Event": ev, "SubEvent": sub}

        # Unresolved names are FILLED with NaN -- if the key were absent
        # entirely the caller would get a KeyError.
        for name in unresolved:
            out[name] = np.full(n, np.nan)

        # Are there repeated Run/Event/SubEvent triples?
        dup = _has_duplicate_ids(run, ev, sub)
        if dup:
            key = (_sample_tag(path), "__dup__")
            if key not in _warned_unresolved:
                _warned_unresolved.add(key)
                print("  (i) %s: Run/Event/SubEvent triples repeat within the "
                      "part (expected: one part holds several L3 files) -- "
                      "matching goes through __I3Index__"
                      % _sample_tag(path))
        ref_key = None          # built on demand (expensive)

        for tbl, cols in need.items():
            node = nodes.get(tbl)
            if node is None:
                for name, _ in cols:
                    out[name] = np.full(n, np.nan)
                continue

            r2, e2, s2 = _ids(node)
            same = (len(r2) == n and np.array_equal(r2, run)
                    and np.array_equal(e2, ev) and np.array_equal(s2, sub))

            if same:
                idx = None                       # aligned: use directly
            else:
                idx = None
                # 1) THE RIGHT WAY: hdfwriter's frame index
                inode = _index_node(h5, tbl)
                if inode is not None and len(inode) == n \
                        and "exists" in inode.colnames and "start" in inode.colnames:
                    ex = np.asarray(inode.col("exists")).astype(bool)
                    st = np.asarray(inode.col("start"), dtype=np.int64)
                    idx = np.where(ex, st, -1)
                    idx[idx >= len(r2)] = -1     # defensive
                elif dup:
                    # 2) No index AND repeating triples -> reliable matching
                    #    is IMPOSSIBLE.  Leave NaN and SAY SO, rather than
                    #    matching wrongly in silence.
                    key = (_sample_tag(path), "__ambig__" + tbl)
                    if key not in _warned_unresolved:
                        _warned_unresolved.add(key)
                        print("  [!] %s: no __I3Index__ and Run/Event/SubEvent "
                              "repeats -> no reliable matching, NaN"
                              % tbl)
                    idx = np.full(n, -1, dtype=np.int64)
                else:
                    # 3) No index but the triples are unique -> dict match
                    if ref_key is None:
                        ref_key = {k: i for i, k in
                                   enumerate(zip(run.tolist(), ev.tolist(),
                                                 sub.tolist()))}
                    idx = np.full(n, -1, dtype=np.int64)
                    for j, k in enumerate(zip(r2.tolist(), e2.tolist(),
                                              s2.tolist())):
                        i = ref_key.get(k)
                        if i is not None:
                            idx[i] = j

            for name, col in cols:
                if col not in node.colnames:
                    out[name] = np.full(n, np.nan)
                    continue
                v = np.asarray(node.col(col), dtype=np.float64)
                if idx is None:
                    out[name] = v
                else:
                    a = np.full(n, np.nan)
                    ok = idx >= 0
                    a[ok] = v[idx[ok]]
                    out[name] = a
    return out


def n_l3_files(h5path):
    """
    How many L3 files this HDF5 was produced from.

    CRITICAL: the weight normalisation (OneWeight/n_flux/n_files) must be
    divided by the number of L3 FILES.  Earlier code used the number of HDF5
    files -- with 100 L3 files booked into one HDF5 the divisor came out as 1
    and the weights were 100 TIMES too large.  process_L4.py now writes
    <output>.meta.json next to every output; the right number is in there.
    """
    meta = h5path + ".meta.json"
    if os.path.exists(meta):
        try:
            with open(meta) as fh:
                m = json.load(fh)
            v = m.get("n_l3_files")
            if v is None or m.get("n_l3_files_unreliable"):
                # Smoke output produced with --n: the tray stops early, so
                # the file count is unreliable and unusable as a divisor.
                return None
            return int(v)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return None


def sample_files(name, SAMPLES, include_smoke=False):
    """
    Find the HDF5 files that EXIST for a sample.

    L4_nue.hdf5, L4_nue_part000.hdf5, ... all match.  _smoke files are
    EXCLUDED by default (small test output, not production); if no production
    file exists it falls back to them.
    """
    pattern = SAMPLES[name]["hdf5"].replace(".hdf5", "*.hdf5")
    files = sorted(glob.glob(pattern))
    if not include_smoke:
        real = [f for f in files if "_smoke" not in f]
        if real:
            return real
    return files


def find_hdf5(name, SAMPLES, verbose=True):
    """
    Return the FIRST existing HDF5 of a sample -- for dump_tables.

    Use this instead of hardcoding a "_smoke.hdf5" path: if the smoke test was
    never run that file does NOT exist while the production output does.
    """
    files = sample_files(name, SAMPLES)
    if files:
        if verbose:
            print("%s: %d HDF5 files found, inspecting the first -> %s"
                  % (name, len(files), os.path.basename(files[0])))
        return files[0]
    if verbose:
        print("[!] no HDF5 for %s: %s"
              % (name, SAMPLES[name]["hdf5"].replace(".hdf5", "*.hdf5")))
        print("    Run section 1 first (run_all).")
    return None


def load_sample(name, SAMPLES, wanted, max_files=None):
    """Read and concatenate every HDF5 file of a sample."""
    # Do not ask for AUX columns this sample is not EXPECTED to have.
    # Otherwise every CORSIKA file prints a misleading "OneWeight missing".
    scheme = scheme_of(SAMPLES[name])
    if scheme:
        drop = set(AUX) - set(aux_for(scheme))
        skipped = [w for w in wanted if w in drop]
        wanted = [w for w in wanted if w not in drop]
        if skipped:
            print("  (%s: skipped %d AUX columns not expected in this sample: %s)"
                  % (name, len(skipped), ", ".join(sorted(skipped))))

    files = sample_files(name, SAMPLES)
    if max_files:
        files = files[:max_files]
    if not files:
        print("[!] %s: no HDF5 found (%s)"
              % (name, SAMPLES[name]["hdf5"].replace(".hdf5", "*.hdf5")))
        return None

    parts = [load_one_file(f, wanted) for f in files]
    keys = parts[0].keys()
    data = {k: np.concatenate([p[k] for p in parts]) for k in keys}

    # --- number of L3 files: the divisor of the weights ---
    per_file = [n_l3_files(f) for f in files]
    if all(v is not None for v in per_file):
        n_l3 = sum(per_file)
        src = "meta.json"
    else:
        n_l3 = SAMPLES[name].get("n_l3_files") or len(files)
        src = "SAMPLES['%s']['n_l3_files'] (no meta.json!)" % name
        print("  [!] %s: some HDF5 files have no meta.json -> n_l3_files=%d (%s)"
              % (name, n_l3, src))
        print("      The weights may be WRONG.  Re-run process_L4.py with this")
        print("      version, or verify n_l3_files by hand.")
    data["_n_files"] = float(n_l3)
    data["_n_hdf5"] = len(files)
    # The CORSIKA weighting has to REOPEN the files (simweights reads the
    # HDF5 directly), so keep the list and the per-part L3 counts.
    data["_files"] = list(files)
    data["_n_l3_per_file"] = [n_l3_files(f) for f in files]

    print("%-8s %8d events, %d HDF5, %d L3 files (%s)"
          % (name, len(data["Run"]), len(files), n_l3, src))
    return data


WANTED = sorted(set(NOISE_FEATURES) | set(MUON_FEATURES) | set(AUX))


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
#
# The divisor d["_n_files"] is the number of L3 FILES, NOT the number of HDF5
# files -- load_sample sums it from n_l3_files in <output>.hdf5.meta.json.
# (Earlier code used the HDF5 count: with 100 L3 files booked into one HDF5
# the divisor came out as 1 and the weights were 100 TIMES too large.)

# VERIFIED against the official project: oscNext/frame_objects/weighting.py
# passes norm=2.e-2, spectral_index=-3. to add_single_powerlaw_flux_weight for
# machine-learning training samples, and neutrinos.py defaults gen_ratio to 0.7
# for GENIE (1 - 0.7 for antineutrinos).  These are not our invention.
NORM, GAMMA = 2e-2, -3.0
NU_FRAC, NUBAR_FRAC = 0.7, 0.3

# Unit of the vuvuzela noise_weight.  THIS DIFFERS BETWEEN PRODUCTIONS:
#   pass3 -> 1/ns, so the rate in Hz is weight * 1e9
#   pass2 -> already Hz, so the factor is 1
#
# Getting it wrong does not raise; it scales every noise rate by 1e9 and the
# only symptom is an absurd number far downstream.  So it is not a bare
# constant any more: set it with set_noise_weight_unit(), and every weight
# calculation prints which unit it used.
#
# The pass3 value is VERIFIED by measurement (CLAUDE.md open risk 5d): it put
# the sample's total noise rate at 41.4 mHz against the note's 36.6 mHz.  The
# pass2 value has NOT been verified by us -- Table 13 of the note is pass2's
# own number, so the same comparison settles it on the first pass2 run.
NOISE_UNITS = {"per_ns": 1e9, "hz": 1.0}
NOISE_WEIGHT_UNIT = "per_ns"          # pass3 default
NOISE_NS_SCALE = NOISE_UNITS[NOISE_WEIGHT_UNIT]


def set_noise_weight_unit(unit):
    """
    Declare the unit of the vuvuzela noise_weight column: "per_ns" (pass3) or
    "hz" (pass2).  Call it before add_weights() when working on pass2.
    """
    global NOISE_WEIGHT_UNIT, NOISE_NS_SCALE
    if unit not in NOISE_UNITS:
        raise ValueError("unit must be one of %s, not %r"
                         % (sorted(NOISE_UNITS), unit))
    NOISE_WEIGHT_UNIT = unit
    NOISE_NS_SCALE = NOISE_UNITS[unit]
    print("  [i] noise_weight unit: %s (factor %g)" % (unit, NOISE_NS_SCALE))
    return NOISE_NS_SCALE


def genie_weight(d):
    """
    w [Hz] = OneWeight * flux(E) / n_flux / n_files,  flux = NORM * E^GAMMA.

    VERIFIED against the official project (icetray-oscNext,
    oscNext/frame_objects/).  weighting.py adds, for exactly our purpose --
    its own comment says "useful for unbiased samples for training machine
    learning algorithms" -- a single power law with norm=2.e-2 and
    spectral_index=-3., which is our NORM and GAMMA to the digit.  neutrinos.py
    then computes

        weight = OneWeight * flux / ( NEvents * gen_ratio )        (line 362)
        flux   = norm * energy ** spectral_index                   (line 334)
        gen_ratio = 0.7 for GENIE, and 1 - 0.7 for antineutrinos    (line 132)

    and weighting.py divides by the file count at the end, as we do.

    SO THE "FALLBACK" BELOW IS THE PRODUCTION FORMULA.  It is our n_flux_events
    path -- taken from I3GenieInfo -- that is the deviation; the official code
    never reads that field.  The two agree only if
    n_flux_events == NEvents * gen_ratio, which has NOT been measured.  pass3
    files carry both, so it can be.  pass2 GENIE L3 has no I3GenieInfo at all,
    so a pass2 run uses the production formula throughout.
    """
    E = d["true_energy"]
    ow = d["OneWeight"]
    flux = NORM * np.power(E, GAMMA, where=E > 0, out=np.full_like(E, np.nan))

    n_flux = d.get("n_flux_events", np.full_like(E, np.nan)).copy()
    missing = ~np.isfinite(n_flux)
    if missing.any():
        frac = np.where(d["pdg"] < 0, NUBAR_FRAC, NU_FRAC)
        n_flux[missing] = (d["NEvents"] * frac)[missing]
        print("  [i] n_flux_events missing in %d of %d events -> NEvents * "
              "(%.1f/%.1f)" % (missing.sum(), missing.size, NU_FRAC,
                               NUBAR_FRAC))
        if missing.all():
            # Two very different causes, and blaming the flag for both sent
            # the reader after the wrong one: the pass2 GENIE L3 files simply
            # have no I3GenieInfo, so this path is the ONLY one there and
            # nothing was forgotten.
            print("      ALL events: either --genie was not passed, or the "
                  "input has no I3GenieInfo at all (pass2 GENIE L3 does not).")
            print("      This is NOT a degraded path: NEvents * gen_ratio is "
                  "what the official oscNext weighting does (neutrinos.py "
                  "line 362, gen_ratio 0.7 for GENIE).")
        else:
            print("      (a partial miss usually means --genie was forgotten "
                  "for some parts)")
    return ow * flux / n_flux / d["_n_files"]


def noise_weight(d):
    # Printed every time: the factor is a per-production choice and a silent
    # wrong one costs a factor of 1e9.
    print("  [i] noise weight: unit=%s factor=%g"
          % (NOISE_WEIGHT_UNIT, NOISE_NS_SCALE))
    return d["noise_weight"] * NOISE_NS_SCALE / d["_n_files"]


def _corsika_weight_manual(d):
    """
    Without simweights: CorsikaWeightMap.Weight / (NEvents * OverSampling).

    The same fallback the earlier LightGBM notebook used.  The absolute rate is
    unreliable, but at least the events are weighted by the SPECTRUM -- far
    better than handing every event a flat 1.0.
    """
    print("  [!] simweights MISSING -> approximate weights from CorsikaWeightMap.")
    print("      The absolute rate is unreliable; the spectrum is at least right.")
    w = np.asarray(d.get("cwm_Weight"), dtype=np.float64)
    nev = np.asarray(d.get("cwm_NEvents"), dtype=np.float64)
    osamp = np.asarray(d.get("cwm_OverSampling"), dtype=np.float64)
    osamp = np.where(np.isfinite(osamp) & (osamp > 0), osamp, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return w / (nev * osamp * d["_n_files"])


def _open_for_simweights(path):
    """Open a file object simweights accepts (pytables, else h5py)."""
    try:
        return tables.open_file(path, "r"), "tables"
    except Exception:
        pass
    try:
        import h5py
        return h5py.File(path, "r"), "h5py"
    except Exception as e:
        raise RuntimeError("could not open: %s" % e)


def corsika_weight(d):
    """
    CORSIKA weights -- simweights + GaisserH3a, the method of the earlier
    notebook.

    simweights reads the HDF5 DIRECTLY (CorsikaWeightMap, PolyplopiaPrimary,
    I3CorsikaInfo, ...), so the files are reopened here.  The result is matched
    back on Run/Event/SubEvent -- row order is not trusted.

    NORMALISATION -- deliberately DIFFERENT from the earlier notebook:
      That code passed nfiles=1 per HDF5 and divided by the HDF5 count at the
      end.  Our parts hold several L3 files each (--chunk-files), so each
      part's OWN n_l3_files is passed as nfiles and there is NO further
      division at the end.  Otherwise the weights would come out too small by
      the number of files per part.
    """
    try:
        import simweights
    except ImportError:
        return _corsika_weight_manual(d)

    files = d.get("_files") or []
    n_l3 = d.get("_n_l3_per_file") or []
    if not files:
        print("  [!] no file list -> falling back to approximate weights")
        return _corsika_weight_manual(d)

    key2w, n_fail = {}, 0
    for path, nl3 in zip(files, n_l3):
        if not nl3:
            print("  [!] %s: no meta.json, nfiles unknown -> skipped"
                  % os.path.basename(path))
            n_fail += 1
            continue
        fh = kind = None
        try:
            fh, kind = _open_for_simweights(path)
            wobj = simweights.CorsikaWeighter(fh, nfiles=nl3)
            w = np.asarray(wobj.get_weights(simweights.GaisserH3a()), dtype=np.float64)
            with tables.open_file(path, "r") as h5:
                eh = _table_nodes(h5)["I3EventHeader"]
                r, e, sub = _ids(eh)
            if len(w) != len(r):
                print("  [!] %s: simweights gave %d weights for %d events -> skipped"
                      % (os.path.basename(path), len(w), len(r)))
                n_fail += 1
                continue
            for k, ww in zip(zip(r, e, sub), w):
                key2w[k] = ww
        except Exception as ex:
            print("  [!] simweights failed (%s): %s"
                  % (os.path.basename(path), str(ex)[:120]))
            n_fail += 1
        finally:
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass

    if not key2w:
        print("  [!] simweights worked on no file -> approximate weights")
        return _corsika_weight_manual(d)

    out = np.array([key2w.get(k, np.nan) for k in
                    zip(d["Run"], d["Event"], d["SubEvent"])], dtype=np.float64)
    matched = np.isfinite(out).mean()
    print("  simweights + GaisserH3a: %d events weighted (%.0f%% matched%s)"
          % (len(key2w), 100 * matched,
             ", %d files failed" % n_fail if n_fail else ""))
    if matched < 0.99:
        print("      [!] unmatched events are NaN -> counted as w_phys 0")
    return out


def muongun_weight(d):
    """
    The pass2 muon background.  There is no CORSIKA at pass2, so the muon BDT
    needs this or it has no weighted background at all.

    MuonGun stores a per-event rate weight directly, unlike CORSIKA where the
    spectrum has to be folded in.  process_L4.py books three spellings because
    which one a production wrote is not fixed; the first one actually present is
    used, and which that was is printed -- the choice changes the normalisation
    and nothing downstream would catch it.

    Divided by the L3 file count, as every other weighter here is.

    NOT VERIFIED against a pass2 rate.  The production's own
    `frame_objects/muongun.py` uses `raw_weight / num_events /
    prob_passing_KDE`; the KDE passing probability is not in our booked keys, so
    an absolute-rate comparison is not available yet.  The SHAPE -- which event
    counts how much -- is what the BDT trains on, and that is right.
    """
    for col in ("MuonWeight", "MuonWeight_GaisserH4a", "MuonGunWeight"):
        w = d.get(col)
        if w is None:
            continue
        w = np.asarray(w, dtype=np.float64)
        if np.isfinite(w).any():
            print("  [i] muongun weight: using %s" % col)
            return w / d["_n_files"]
    print("  [!] no MuonGun weight column found (looked for MuonWeight, "
          "MuonWeight_GaisserH4a, MuonGunWeight)")
    print("      -> w_phys = 0; the muon BDT would see an unweighted background.")
    return np.zeros(len(d["Run"]), dtype=np.float64)


# Keyed by WEIGHT SCHEME, not by sample name -- see productions.py.  A sample
# called "corsika" at pass3 and "muongun" at pass2 is the same ROLE with two
# formulas, and nothing here should have to know the names.
WEIGHTERS = {"genie": genie_weight, "noise": noise_weight,
             "corsika": corsika_weight, "muongun": muongun_weight}

# Table 13 of the technical note, L3 rates [mHz] -- for the magnitude check.
# Keyed by sample name: the table is a per-sample fact, and it has no MuonGun
# or NuTau row, so those samples simply print no comparison.
L3_RATES_MHZ = {"nue": 0.95, "numu": 3.77, "corsika": 505.0, "noise": 36.6}


def add_weights(data, SAMPLES=None, weighters=None):
    """
    Add w_phys [Hz] to every sample and print the magnitude comparison
    against Table 13.

    The weighter is chosen by the sample's WEIGHT SCHEME (`SAMPLES[name]
    ["weight"]`, see productions.py), not by its name.  That is what lets the
    same call serve pass2 and pass3: "which formula" is a fact the production
    table declares, and the names differ between productions while the schemes
    do not.  Without SAMPLES the name is tried as a scheme, which still works
    for "noise" and "corsika".

    max/sum above 5% means a single event dominates the rate: either the
    statistics are inadequate or the weight calculation is wrong.
    """
    weighters = weighters or WEIGHTERS
    for name, d in data.items():
        print(name)
        scheme = scheme_of(SAMPLES[name]) if SAMPLES and name in SAMPLES else name
        fn = weighters.get(scheme)
        if fn is None:
            print("  [!] no weighting function for scheme %r -> w_phys = NaN"
                  % scheme)
            print("      Give this sample a `weight=` in productions.py; the "
                  "known schemes are %s." % ", ".join(sorted(weighters)))
            d["w_phys"] = np.full(len(d["Run"]), np.nan)
            continue
        w = np.asarray(fn(d), dtype=np.float64)
        w[~np.isfinite(w)] = 0.0
        d["w_phys"] = w

        tot = w.sum()
        frac = 100 * w.max() / tot if tot > 0 else np.nan
        print("  total rate = %.4e Hz  (%.3f mHz),  max/sum = %.2f%%"
              % (tot, 1e3 * tot, frac))
        if frac > 5:
            print("      [!] one event dominates the rate (>5%) -- inadequate "
                  "statistics or a wrong weight")

        ref = L3_RATES_MHZ.get(name)
        if ref:
            ratio = (1e3 * tot) / ref if ref else np.nan
            flag = "" if 0.1 <= ratio <= 10 else "   [!] ORDER OF MAGNITUDE OFF"
            print("      Table 13 (L3, pass2): %.3f mHz  ->  ours/note "
                  "= %.2f%s" % (ref, ratio, flag))
    return data


# ---------------------------------------------------------------------------
# REGISTRY <-> FEATURE_MAP consistency
# ---------------------------------------------------------------------------

def read_feature_map(path=None):
    """
    Read FEATURE_MAP out of classifier.py **with AST**.

    It is not imported because that module depends on icetray; parsing means
    this check also runs in a plain python kernel.
    """
    import ast
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "classifier.py")
    if not os.path.exists(path):
        return None
    tree = ast.parse(open(path).read())

    ns = {}          # module-level string constants (L3V, HITSTAT, ...)
    fmap = None

    def ev(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return ns[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return ev(node.left) + ev(node.right)
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(ev(e) for e in node.elts)
        raise ValueError

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name):
            continue
        if tgt.id == "FEATURE_MAP" and isinstance(node.value, ast.Dict):
            fmap = {}
            for k, v in zip(node.value.keys, node.value.values):
                try:
                    fmap[ev(k)] = tuple(ev(v))
                except (ValueError, KeyError, TypeError):
                    pass
        else:
            try:
                ns[tgt.id] = ev(node.value)
            except (ValueError, KeyError, TypeError):
                pass
    return fmap


def read_column_alts(path=None):
    """
    Read classifier.COLUMN_ALTS with AST.

    That module tries field-name variants on the frame side.  The HDF5 column
    name and the frame field name DO NOT HAVE TO MATCH -- the hdfwriter
    converter may rename (e.g. fill_ratio_from_mean -> fillratio_from_mean).
    So the consistency check has to know both sides' alternatives, otherwise a
    legitimate name difference is reported as a "conflict".
    """
    import ast
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "classifier.py")
    if not os.path.exists(path):
        return {}
    tree = ast.parse(open(path).read())
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "COLUMN_ALTS"
                and isinstance(node.value, ast.Dict)):
            out = {}
            for k, v in zip(node.value.keys, node.value.values):
                try:
                    out[tuple(ast.literal_eval(k))] = list(ast.literal_eval(v))
                except Exception:
                    pass
            return out
    return {}


def check_feature_map(verbose=True):
    """
    Do classifier.FEATURE_MAP and REGISTRY agree?

    If training reads one column and frame application another, the model
    produces nonsense silently -- it does not raise.  Hence the check in code.
    Returns the list of mismatches (empty = consistent, None = unreadable).
    """
    fmap = read_feature_map()
    if not fmap:
        if verbose:
            print("[!] FEATURE_MAP could not be read (classifier.py is missing "
                  "or its format changed)")
        return None

    # DANGEROUS: same name, DIFFERENT column -> training and application diverge
    conflict = []
    # Harmless: present in FEATURE_MAP but not in REGISTRY.  FEATURE_MAP is a
    # superset (it also holds candidate variables), which is expected.
    only_fmap = []

    colalts = read_column_alts()

    def _accepts(pair, extra_cols):
        """(table, column) plus the alternative column names for that pair."""
        tbl, col = pair
        return {(tbl, col)} | {(tbl, c) for c in extra_cols}

    for name, pair in sorted(fmap.items()):
        mine = REGISTRY.get(name)
        if mine is None:
            only_fmap.append(name)
            continue
        # Do the (table, column) sets each side accepts intersect?
        mine_set = set(ALTS.get(name, [tuple(mine)]))
        fmap_set = _accepts(tuple(pair), colalts.get(tuple(pair), []))
        if not (mine_set & fmap_set):
            conflict.append((name, tuple(mine), tuple(pair)))

    for name in sorted(set(NOISE_FEATURES) | set(MUON_FEATURES)):
        if name in REGISTRY and name not in fmap:
            conflict.append((name, tuple(REGISTRY[name]), "ABSENT from FEATURE_MAP"))

    if verbose:
        if conflict:
            print("[!] CONFLICT -- same variable, different column.")
            print("    Training reads one, frame application the other; the")
            print("    model produces nonsense without raising:")
            for name, a, b in conflict:
                print("    %-22s REGISTRY=%s  FEATURE_MAP=%s" % (name, a, b))
        else:
            print("REGISTRY <-> FEATURE_MAP: no conflict "
                  "(%d shared variables)" % (len(set(fmap) & set(REGISTRY))))
        if only_fmap:
            print("  (info) FEATURE_MAP holds %d extra candidate variables: %s"
                  % (len(only_fmap), ", ".join(only_fmap[:6])
                     + (" ..." if len(only_fmap) > 6 else "")))
    return conflict
