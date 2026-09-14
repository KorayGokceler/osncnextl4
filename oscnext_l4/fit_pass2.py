"""
Fit the pass2 definition of the variables our rewrite does NOT reproduce.

WHY
---
The first pass2 cross-check validated the setup and two of the rewrites:

    control rows (iLineFit_speed, cog_z, z_sigma, z_travel, n_hit_doms,
                  NchCleaned, ICVetoHits)                     all identical
    micro_count                                    identical (--micro-count-uncleaned)
    FullTimeLengthRatio                            identical
    VICH_nch / npulses / qtot                      ours systematically SMALLER
    accumulated_time                               ours systematically LARGER
    first_hlc_rho                                  differs in ~19% of events

Since the control rows are identical, the three failures are not an input
problem: both sides read the same pulses.  They are definition problems --
the technical note's one-line descriptions do not pin the calculation down,
and the original code that would have (analysis.event_selection for the
Dunkman variables, tau_bdt.I3CutL7Module for VICH) is not in this
meta-project.  CLAUDE.md open risks 1 and 2 say exactly this.

But guessing is no longer necessary.  The pass2 L4 files ARE the answer key,
and they contain both the reference value and the pulse series it was computed
from, in the same frame.  So instead of reasoning about what the C++ might
have done, compute MANY candidate definitions per event and ask which one
reproduces the stored number.

This module runs over the pass2 **L4** files for that reason: the reference
and the input sit in one frame, so there is no event matching and no doubt
about what went in.  The variants write keys of their own (FIT_*), so nothing
collides with what is already there.

HOW TO READ THE RESULT
----------------------
A variant that reproduces the reference in ~100% of events IS the pass2
definition.  Anything in between (say 60%) is a partial match: right in the
common case, wrong in a sub-case, which usually means one more condition is
missing rather than that the whole idea is wrong.  The report prints the
agreement fraction and the median absolute difference so the two can be told
apart -- a variant can agree rarely and still be close everywhere.
"""

import numpy as np

from .variables import (VICH_SPEED_MIN, VICH_SPEED_MAX,
                        STRING36_X, STRING36_Y)


# ---------------------------------------------------------------------------
# 1. What a variant receives
# ---------------------------------------------------------------------------
#
# Every variant is a pure function of plain numpy arrays -- no frame, no
# IceTray.  That is deliberate: the definitions can then be unit tested on
# synthetic hits, and a wrong answer is a bug in arithmetic rather than in
# frame handling.
#
# An "event" dict holds, for each of the cleaned and uncleaned series:
#     t, q      pulse time [ns] and charge [PE]
#     x, y, z   the DOM position of that pulse
#     dom       an integer DOM id, so pulses can be grouped per DOM
#     hlc       bool, the I3RecoPulse LC flag
#     veto      bool, the DOM is in the DeepCore Filter veto region
#     fid       bool, the DOM is in the DeepCore fiducial region
# plus:
#     twr       (start, stop) of the cleaned series' TimeRange, or None


def _cum_fraction_time(t, q, fraction):
    """
    Time from the first entry to the point where the cumulative charge first
    reaches `fraction` of the total.  The shared core of every
    accumulated_time variant: the variants differ only in WHICH entries they
    hand in and in the zero point, never in this step.
    """
    if t.size == 0:
        return np.nan
    order = np.argsort(t, kind="stable")
    t, q = t[order], np.maximum(q[order], 0.0)
    total = q.sum()
    if total <= 0:
        return np.nan
    cum = np.cumsum(q) / total
    idx = int(np.searchsorted(cum, fraction))
    idx = min(idx, t.size - 1)
    return float(t[idx] - t[0])


def _per_dom(dom, t, q):
    """Collapse pulses to one entry per DOM: total charge, earliest time."""
    if dom.size == 0:
        return dom, t, q
    order = np.argsort(dom, kind="stable")
    d, tt, qq = dom[order], t[order], q[order]
    edges = np.flatnonzero(np.diff(d)) + 1
    groups = np.split(np.arange(d.size), edges)
    out_t = np.array([tt[g].min() for g in groups])
    out_q = np.array([np.maximum(qq[g], 0.0).sum() for g in groups])
    return np.array([d[g[0]] for g in groups]), out_t, out_q


def _first_pulse_per_dom(dom, t, q):
    """Keep only each DOM's earliest pulse (charge of that pulse only)."""
    if dom.size == 0:
        return t, q
    order = np.lexsort((t, dom))
    d, tt, qq = dom[order], t[order], q[order]
    keep = np.ones(d.size, dtype=bool)
    keep[1:] = d[1:] != d[:-1]
    return tt[keep], qq[keep]


# ---------------------------------------------------------------------------
# 2. accumulated_time variants
# ---------------------------------------------------------------------------
#
# Observed: ours is systematically LARGER than pass2, and pass2 stores exact
# zeros in some events.  A zero means their 75% point IS their first entry, so
# the leading hypotheses are the ones that SHRINK the entry list (fewer, more
# concentrated entries reach 75% sooner) rather than the ones that move the
# zero point.

def acc_all_pulses(ev, fraction=0.75):
    """What we do now: every pulse of the cleaned series, time ordered."""
    c = ev["cleaned"]
    return _cum_fraction_time(c["t"], c["q"], fraction)


def acc_first_pulse_per_dom(ev, fraction=0.75):
    """One entry per DOM: its earliest pulse, that pulse's charge."""
    c = ev["cleaned"]
    t, q = _first_pulse_per_dom(c["dom"], c["t"], c["q"])
    return _cum_fraction_time(t, q, fraction)


def acc_per_dom_charge(ev, fraction=0.75):
    """One entry per DOM: the DOM's TOTAL charge at its earliest time."""
    c = ev["cleaned"]
    _, t, q = _per_dom(c["dom"], c["t"], c["q"])
    return _cum_fraction_time(t, q, fraction)


def acc_hlc_only(ev, fraction=0.75):
    """Only HLC pulses."""
    c = ev["cleaned"]
    m = c["hlc"]
    return _cum_fraction_time(c["t"][m], c["q"][m], fraction)


def acc_unit_charge(ev, fraction=0.75):
    """Ignore charge: 75% of the HIT COUNT rather than of the charge."""
    c = ev["cleaned"]
    return _cum_fraction_time(c["t"], np.ones_like(c["t"]), fraction)


def acc_from_window_start(ev, fraction=0.75):
    """Zero point = the cleaned series' TimeRange start, not the first pulse."""
    c = ev["cleaned"]
    if ev["twr"] is None or c["t"].size == 0:
        return np.nan
    v = _cum_fraction_time(c["t"], c["q"], fraction)
    if not np.isfinite(v):
        return np.nan
    return float(v + (np.sort(c["t"])[0] - ev["twr"][0]))


def acc_uncleaned(ev, fraction=0.75):
    """The same calculation on the UNCLEANED series."""
    u = ev["uncleaned"]
    return _cum_fraction_time(u["t"], u["q"], fraction)


def acc_interp(ev, fraction=0.75):
    """Linear interpolation to exactly 75% instead of the first pulse past it."""
    c = ev["cleaned"]
    t, q = c["t"], c["q"]
    if t.size == 0:
        return np.nan
    order = np.argsort(t, kind="stable")
    t, q = t[order], np.maximum(q[order], 0.0)
    total = q.sum()
    if total <= 0:
        return np.nan
    cum = np.cumsum(q) / total
    if cum[0] >= fraction:
        return 0.0
    return float(np.interp(fraction, cum, t) - t[0])


ACC_VARIANTS = {
    "all_pulses (ours)":   acc_all_pulses,
    "first_pulse_per_dom": acc_first_pulse_per_dom,
    "per_dom_charge":      acc_per_dom_charge,
    "hlc_only":            acc_hlc_only,
    "unit_charge":         acc_unit_charge,
    "from_window_start":   acc_from_window_start,
    "uncleaned_series":    acc_uncleaned,
    "interpolated":        acc_interp,
}


# ---------------------------------------------------------------------------
# 3. VICH variants
# ---------------------------------------------------------------------------
#
# Observed: ours counts far FEWER veto DOMs than pass2 (4 against 60 in the
# worst events).  So the leading hypotheses LOOSEN a condition: a wider speed
# window, no speed window at all, or no causal direction requirement.

def _vich_count(ev, cog, speed_min, speed_max, require_causal=True,
                source="uncleaned"):
    """Count veto DOMs / pulses / charge under one set of conditions."""
    s = ev[source]
    if cog is None or s["t"].size == 0:
        return np.nan, np.nan, np.nan
    cx, cy, cz, ct = cog
    m = s["veto"]
    if not m.any():
        return 0.0, 0.0, 0.0
    x, y, z = s["x"][m], s["y"][m], s["z"][m]
    t, q, dom = s["t"][m], s["q"][m], s["dom"][m]

    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
    dt = ct - t                      # the veto hit must precede the COG
    ok = dt > 0 if require_causal else np.abs(dt) > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(ok, d / np.where(ok, np.abs(dt), 1.0), np.nan)
    sel = ok & np.isfinite(speed)
    if speed_min is not None:
        sel &= speed >= speed_min
    if speed_max is not None:
        sel &= speed <= speed_max

    n_pulses = float(sel.sum())
    qtot = float(np.maximum(q[sel], 0.0).sum())
    n_dom = float(np.unique(dom[sel]).size)
    return n_dom, n_pulses, qtot


def _cog(ev, fiducial=True, charge_weighted=True):
    c = ev["cleaned"]
    m = c["fid"] if fiducial else np.ones(c["t"].size, dtype=bool)
    if not m.any():
        m = np.ones(c["t"].size, dtype=bool)
    if not m.any():
        return None
    w = np.maximum(c["q"][m], 0.0) if charge_weighted else np.ones(int(m.sum()))
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    return (float(w @ c["x"][m]), float(w @ c["y"][m]),
            float(w @ c["z"][m]), float(w @ c["t"][m]))


def _vich_variant(fiducial=True, charge_weighted=True,
                  speed_min=VICH_SPEED_MIN, speed_max=VICH_SPEED_MAX,
                  require_causal=True, source="uncleaned"):
    def f(ev):
        return _vich_count(ev, _cog(ev, fiducial, charge_weighted),
                           speed_min, speed_max, require_causal, source)
    return f


VICH_VARIANTS = {
    "ours (fid cog, 0.25-0.40)": _vich_variant(),
    "cog_all_hits":              _vich_variant(fiducial=False),
    "cog_unweighted":            _vich_variant(charge_weighted=False),
    "speed_max_only":            _vich_variant(speed_min=None),
    "speed_min_only":            _vich_variant(speed_max=None),
    "no_speed_cut":              _vich_variant(speed_min=None, speed_max=None),
    "no_causal_cut":             _vich_variant(speed_min=None, speed_max=None,
                                               require_causal=False),
    "cleaned_input":             _vich_variant(source="cleaned"),
    "cog_all_no_speed":          _vich_variant(fiducial=False, speed_min=None,
                                               speed_max=None),
}


# ---------------------------------------------------------------------------
# 4. first_hlc_rho variants
# ---------------------------------------------------------------------------
#
# Observed: agrees in ~80% of events.  A rule that is right most of the time
# and wrong in a fifth of events is usually one condition away, not wrong in
# kind -- so the variants change WHICH hit counts as "first", not the rho
# formula.

def _rho(x, y):
    return float(np.sqrt((x - STRING36_X) ** 2 + (y - STRING36_Y) ** 2))


def _earliest(ev, source, hlc_only, fiducial_only=False):
    s = ev[source]
    m = np.ones(s["t"].size, dtype=bool)
    if hlc_only:
        m &= s["hlc"]
    if fiducial_only:
        m &= s["fid"]
    if not m.any():
        return np.nan
    i = np.argmin(np.where(m, s["t"], np.inf))
    return _rho(s["x"][i], s["y"][i])


RHO_VARIANTS = {
    "hlc_cleaned (ours)":  lambda ev: _earliest(ev, "cleaned", True),
    "hlc_uncleaned":       lambda ev: _earliest(ev, "uncleaned", True),
    "any_hit_cleaned":     lambda ev: _earliest(ev, "cleaned", False),
    "any_hit_uncleaned":   lambda ev: _earliest(ev, "uncleaned", False),
    "hlc_cleaned_fid":     lambda ev: _earliest(ev, "cleaned", True, True),
    "hlc_uncleaned_fid":   lambda ev: _earliest(ev, "uncleaned", True, True),
}


# ---------------------------------------------------------------------------
# 5. Scoring
# ---------------------------------------------------------------------------

def score(ref, val, rtol=1e-6, atol=1e-9):
    """
    (fraction agreeing, median |diff|, n compared).

    Agreement is not exact equality: a float recomputed in numpy differs from
    a C++ float in the last bits even when the definition is identical.  What
    distinguishes a right definition from a wrong one is the FRACTION, which
    goes to ~1 for the right one and sits far below for every other.
    The median |diff| is printed alongside because a variant can agree rarely
    and still be close everywhere -- that is a near miss, not a wrong idea.
    """
    both = np.isfinite(ref) & np.isfinite(val)
    n = int(both.sum())
    if n == 0:
        return 0.0, float("nan"), 0
    a, b = val[both], ref[both]
    close = np.abs(a - b) <= (atol + rtol * np.abs(b))
    return float(close.mean()), float(np.median(np.abs(a - b))), n


# ---------------------------------------------------------------------------
# 6. Getting the arrays out of a frame
# ---------------------------------------------------------------------------

FIT_PREFIX = "FIT_"

# The reference each family is fitted against, and its (table, column) in the
# booked HDF5.
REFERENCES = {
    "acc":  ("accumulated_time", "L4_accumulated_time", "value"),
    "vich": ("VICH_nch",         "L4_VICH_nch",         "value"),
    "rho":  ("first_hlc_rho",    "L4_first_hlc_rho",    "value"),
}

# VICH variants return a triple, so they also feed npulses and qtot.
VICH_EXTRA = {
    "vich_npulses": ("L4_VICH_npulses", "value", 1),
    "vich_qtot":    ("L4_VICH_qtot",    "value", 2),
}


def _arrays(pulse_map, geometry, veto_doms, fid_doms):
    """One flat record per pulse.  Returns a dict of numpy arrays."""
    from .variables import iter_map
    from icecube import dataclasses  # noqa: F401

    omgeo = geometry.omgeo
    x, y, z, t, q, dom, hlc, veto, fid = [], [], [], [], [], [], [], [], []
    for i, (omkey, pulses) in enumerate(iter_map(pulse_map)):
        if omkey not in omgeo:
            continue
        pos = omgeo[omkey].position
        in_veto = omkey in veto_doms
        in_fid = omkey in fid_doms
        for p in pulses:
            x.append(pos.x); y.append(pos.y); z.append(pos.z)
            t.append(p.time); q.append(p.charge)
            dom.append(i)
            hlc.append(bool(p.flags & _LC()))
            veto.append(in_veto); fid.append(in_fid)
    f = np.asarray
    return {"x": f(x, float), "y": f(y, float), "z": f(z, float),
            "t": f(t, float), "q": f(q, float),
            "dom": f(dom, np.int64),
            "hlc": f(hlc, bool), "veto": f(veto, bool), "fid": f(fid, bool)}


def _LC():
    from icecube import dataclasses
    return dataclasses.I3RecoPulse.PulseFlags.LC


def extract(frame, cleaned_key, uncleaned_key, geometry_key="I3Geometry"):
    """Build the event dict the variants consume, or None when impossible."""
    from .variables import get_pulses
    from .env import deepcore_veto_domset, deepcore_fiducial_domset

    if geometry_key not in frame:
        return None
    geo = frame[geometry_key]
    cln = get_pulses(frame, cleaned_key)
    unc = get_pulses(frame, uncleaned_key)
    if cln is None or unc is None:
        return None

    veto = deepcore_veto_domset("IC86")
    fid = deepcore_fiducial_domset("IC86")

    twr = None
    tr_key = cleaned_key + "TimeRange"
    if tr_key in frame:
        try:
            tw = frame[tr_key]
            twr = (float(tw.start), float(tw.stop))
        except Exception:
            twr = None

    return {"cleaned": _arrays(cln, geo, veto, fid),
            "uncleaned": _arrays(unc, geo, veto, fid),
            "twr": twr}


def fit_variants(frame, cleaned_pulses, uncleaned_pulses):
    """
    Tray module: write every candidate definition into the frame.

    It runs over the pass2 L4 files, where the reference values already sit,
    so each FIT_* key lands in the same frame as the number it is being fitted
    against -- no event matching anywhere in this comparison.
    """
    from icecube import dataclasses

    ev = extract(frame, cleaned_pulses, uncleaned_pulses)
    if ev is None:
        return True

    for name, fn in ACC_VARIANTS.items():
        key = FIT_PREFIX + "acc_" + _slug(name)
        if key not in frame:
            frame[key] = dataclasses.I3Double(float(_safe(fn, ev)))

    for name, fn in VICH_VARIANTS.items():
        slug = _slug(name)
        try:
            n_dom, n_pulses, qtot = fn(ev)
        except Exception:
            n_dom = n_pulses = qtot = np.nan
        for tag, val in (("vich_", n_dom), ("vich_npulses_", n_pulses),
                         ("vich_qtot_", qtot)):
            key = FIT_PREFIX + tag + slug
            if key not in frame:
                frame[key] = dataclasses.I3Double(float(val))

    for name, fn in RHO_VARIANTS.items():
        key = FIT_PREFIX + "rho_" + _slug(name)
        if key not in frame:
            frame[key] = dataclasses.I3Double(float(_safe(fn, ev)))
    return True


def _safe(fn, ev):
    try:
        v = fn(ev)
    except Exception:
        return np.nan
    return v if v is not None else np.nan


def _slug(name):
    """A frame-key-safe form of a variant's display name."""
    out = []
    for ch in name:
        out.append(ch if (ch.isalnum() or ch == "_") else "_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_").lower()


def fit_keys():
    """Every key this module writes, plus the references to book beside them."""
    keys = ["I3EventHeader"]
    for name in ACC_VARIANTS:
        keys.append(FIT_PREFIX + "acc_" + _slug(name))
    for name in VICH_VARIANTS:
        s = _slug(name)
        keys += [FIT_PREFIX + "vich_" + s,
                 FIT_PREFIX + "vich_npulses_" + s,
                 FIT_PREFIX + "vich_qtot_" + s]
    for name in RHO_VARIANTS:
        keys.append(FIT_PREFIX + "rho_" + _slug(name))
    keys += ["L4_accumulated_time", "L4_VICH_nch", "L4_VICH_npulses",
             "L4_VICH_qtot", "L4_first_hlc_rho"]
    return list(dict.fromkeys(keys))


# ---------------------------------------------------------------------------
# 7. The report
# ---------------------------------------------------------------------------

def report(h5path, stream=None):
    """Rank every candidate definition against the pass2 reference."""
    import sys
    from .pass2 import read_pairs
    out = stream or sys.stdout

    families = [
        ("accumulated_time", "acc_", ACC_VARIANTS,
         ("L4_accumulated_time", "value")),
        ("VICH_nch", "vich_", VICH_VARIANTS, ("L4_VICH_nch", "value")),
        ("VICH_npulses", "vich_npulses_", VICH_VARIANTS,
         ("L4_VICH_npulses", "value")),
        ("VICH_qtot", "vich_qtot_", VICH_VARIANTS, ("L4_VICH_qtot", "value")),
        ("first_hlc_rho", "rho_", RHO_VARIANTS, ("L4_first_hlc_rho", "value")),
    ]

    pairs = []
    for _, prefix, variants, (rtab, rcol) in families:
        pairs.append((prefix + "__ref", rtab, rcol))
        for name in variants:
            pairs.append((prefix + name, FIT_PREFIX + prefix + _slug(name),
                          "value"))
    data, missing = read_pairs(h5path, pairs)

    print("Fitting the pass2 definitions: %s" % h5path, file=out)
    print("events: %d" % len(data["Run"]), file=out)
    if missing:
        print("\nnot found (-> NaN): %s"
              % ", ".join(sorted({m[0] for m in missing})), file=out)

    best = {}
    for label, prefix, variants, _ in families:
        ref = data[prefix + "__ref"]
        print("\n%s" % label, file=out)
        print("  %-30s %10s %14s %8s" % ("variant", "agree", "median|diff|",
                                         "n"), file=out)
        print("  " + "-" * 66, file=out)
        rows = []
        for name in variants:
            frac, med, n = score(ref, data[prefix + name])
            rows.append((frac, med, n, name))
        for frac, med, n, name in sorted(rows, key=lambda r: -r[0]):
            mark = "  <== pass2" if frac > 0.99 else ""
            print("  %-30s %9.2f%% %14.6g %8d%s"
                  % (name, 100 * frac, med, n, mark), file=out)
        if rows:
            top = max(rows, key=lambda r: r[0])
            best[label] = (top[3], top[0])

    print("", file=out)
    for label, (name, frac) in best.items():
        verdict = ("REPRODUCES pass2" if frac > 0.99
                   else "best so far, but NOT the definition")
        print("%-18s -> %-30s %6.2f%%  %s"
              % (label, name, 100 * frac, verdict), file=out)
    return best
