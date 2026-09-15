"""
TEMPORARY SCAFFOLDING -- DELETE THIS FILE once the definitions are settled.

This module exists to FIND two definitions, not to be part of the pipeline.
When VICH and accumulated_time are pinned down, the answer goes into
oscnext_l4/variables.py (the function plus a docstring recording what the
evidence was), the finding goes into CLAUDE.md, and then this file and the
fit/fit-report subcommands of scripts/compare_pass2.py are removed.  Nothing
in the production path imports it.

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


def _cum_fraction_time(t, q, fraction, mode="at"):
    """
    Time from the first entry to the point where the cumulative charge reaches
    `fraction` of the total.  The shared core of every accumulated_time
    variant: they differ in WHICH entries they hand in, in the zero point, and
    in `mode` -- never in this step.

    mode="at"   the first entry whose cumulative charge reaches the fraction
                (what our production code does)
    mode="prev" the last entry BEFORE that one -- an off-by-one convention

    "prev" is not a shot in the dark.  Inverting pass2's numbers showed the
    charge fraction at their stored time sits just BELOW 0.75 in at least 84%
    of events (median 0.7258, 84th pct 0.7449) and never above it, which is
    exactly what taking the entry before the crossing produces: one entry's
    charge share short of the target.  The implied median shortfall, 0.024, is
    the share of an average pulse in an event with this many hits.
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
    if mode == "prev":
        idx = max(idx - 1, 0)
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


def acc_prev_entry(ev, fraction=0.75):
    """The entry BEFORE the cumulative charge crosses 75% -- see the mode
    note in _cum_fraction_time: this is what the inversion points at."""
    c = ev["cleaned"]
    return _cum_fraction_time(c["t"], c["q"], fraction, mode="prev")


def acc_prev_per_dom(ev, fraction=0.75):
    """The same off-by-one, but over per-DOM total charge."""
    c = ev["cleaned"]
    _, t, q = _per_dom(c["dom"], c["t"], c["q"])
    return _cum_fraction_time(t, q, fraction, mode="prev")


ACC_VARIANTS = {
    "all_pulses (ours)":   acc_all_pulses,
    "prev_entry":          acc_prev_entry,
    "prev_entry_per_dom":  acc_prev_per_dom,
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
                source="uncleaned", veto_field="veto"):
    """Count veto DOMs / pulses / charge under one set of conditions."""
    s = ev[source]
    if cog is None or s["t"].size == 0:
        return np.nan, np.nan, np.nan
    cx, cy, cz, ct = cog
    m = s.get(veto_field)
    if m is None or not m.any():
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


def _cog(ev, fiducial=True, charge_weighted=True, cog_field=None):
    c = ev["cleaned"]
    if cog_field is not None:
        m = c.get(cog_field, np.ones(c["t"].size, dtype=bool))
    else:
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


def _first_hlc_ref(ev):
    """
    The earliest HLC hit of the cleaned series as (x, y, z, t).

    Worth trying as VICH's reference point instead of the COG: it is the
    event's entry point rather than its centre, it is the natural thing for a
    causality test to point at, and unlike the COG it is now VERIFIED against
    pass2 (first_hlc_rho reproduces at 99.85%).  So if the reference point is
    what differs, this is the candidate with evidence behind it.
    """
    c = ev["cleaned"]
    m = c["hlc"]
    if not m.any():
        return None
    i = int(np.argmin(np.where(m, c["t"], np.inf)))
    return (float(c["x"][i]), float(c["y"][i]), float(c["z"][i]),
            float(c["t"][i]))


def _vich_ref_variant(ref_fn, speed_min=VICH_SPEED_MIN,
                      speed_max=VICH_SPEED_MAX, require_causal=True,
                      source="uncleaned"):
    def f(ev):
        return _vich_count(ev, ref_fn(ev), speed_min, speed_max,
                           require_causal, source)
    return f


# Where the trigger hierarchy may live, in the order we try.  A pass2 file can
# carry the SuperDST series instead of the plain hierarchy.
TRIGGER_KEYS = ("I3TriggerHierarchy", "QTriggerHierarchy", "DSTTriggers",
                "I3SuperDSTTriggers")


def _causal_band_mask(x, y, z, t, rx, ry, rz, rt):
    """The four (distance, dt) conditions of LowEnVariables' VetoCausalHits."""
    d = np.sqrt((x - rx) ** 2 + (y - ry) ** 2 + (z - rz) ** 2)
    dt = rt - t
    sel = d < 750.0
    sel &= dt > (-5.0 * d + 500.0)
    sel &= dt < (d / 0.3 + 150.0)
    sel &= dt > (d / 0.3 - 1850.0)
    return sel


def _vich_causal_count(ev, source="uncleaned", veto_field="all",
                       config_ids=(1011,)):
    """
    VICH under LowEnVariables' CAUSALITY BANDS instead of a speed window.

    SOURCE.  `grecovariables.VetoCausalHits`, the pythonic reimplementation of
    the LowEnVariables algorithms used by the GRECO online filter.  The GRECO
    tray calls it twice on the SAME uncleaned series, extracted PULSE BY PULSE
    (`GetHitInformation(..., 0)`):

        VetoCausalCharge = VetoCausalHits(..., useCharge=True)    -> summed charge
        VetoCausalHits   = VetoCausalHits(..., useCharge=False)   -> pulse count

    which is exactly the shape of pass2's L4_VICH_qtot / L4_VICH_npulses, on
    exactly the series our _vich is already verified to use.

    HOW IT DIFFERS FROM OUR _vich -- structurally, in three ways:

      1. No speed window.  Selection is four conditions in the (distance, dt)
         plane: a 750 m sphere and three causality bands.
      2. No veto DOM list.  The region is implicit in the bands, so the default
         here is the WHOLE series (veto_field="all").
      3. The reference is the hit closest in time to the TRIGGER, not a
         charge-weighted fiducial COG.

    That matters because the evidence says the disagreement is structural: a
    36-cell sweep of both speed edges peaked at 18.0% with the median
    difference pinned at 2 DOMs in EVERY cell -- the signature of sweeping a
    knob that is not the one that differs.

    The production loops over every matching trigger and ACCUMULATES, so a hit
    seen by two triggers is counted twice; that is reproduced here for the
    charge and pulse counts.  n_dom takes the union instead, since counting one
    DOM twice has no sensible meaning.
    """
    s = ev[source]
    if s["t"].size == 0:
        return np.nan, np.nan, np.nan
    cfg, tt = ev.get("trig_cfg"), ev.get("trig_t")
    if cfg is None or cfg.size == 0:
        return np.nan, np.nan, np.nan
    keep = np.isin(cfg, np.asarray(config_ids, dtype=np.int64))
    if not keep.any():
        return np.nan, np.nan, np.nan

    m = s.get(veto_field)
    if m is None or not m.any():
        return 0.0, 0.0, 0.0
    x, y, z = s["x"][m], s["y"][m], s["z"][m]
    t, q, dom = s["t"][m], s["q"][m], s["dom"][m]

    n_pulses, qtot = 0.0, 0.0
    union = np.zeros(t.size, dtype=bool)
    for trigger_time in tt[keep]:
        i = int(np.argmin(np.abs(t - trigger_time)))
        sel = _causal_band_mask(x, y, z, t, x[i], y[i], z[i], t[i])
        n_pulses += float(sel.sum())
        qtot += float(np.maximum(q[sel], 0.0).sum())
        union |= sel

    n_dom = float(np.unique(dom[union]).size)
    return n_dom, n_pulses, qtot


def _vich_causal_variant(source="uncleaned", veto_field="all",
                         config_ids=(1011,)):
    def f(ev):
        return _vich_causal_count(ev, source, veto_field, config_ids)
    return f


def _vich_variant(fiducial=True, charge_weighted=True,
                  speed_min=VICH_SPEED_MIN, speed_max=VICH_SPEED_MAX,
                  require_causal=True, source="uncleaned",
                  veto_field="veto", cog_field=None):
    def f(ev):
        cog = _cog(ev, fiducial, charge_weighted, cog_field)
        return _vich_count(ev, cog, speed_min, speed_max, require_causal,
                           source, veto_field)
    return f


VICH_VARIANTS = {
    "ours (fid cog, 0.25-0.40)": _vich_variant(),
    # LowEnVariables' causality bands -- the first STRUCTURALLY different
    # candidate.  See _vich_causal_count for the source and the reasoning.
    "causal_all_smt3":        _vich_causal_variant(),
    "causal_all_both_trig":   _vich_causal_variant(config_ids=(1010, 1011)),
    "causal_veto_smt3":       _vich_causal_variant(veto_field="veto"),
    "causal_all_cleaned":     _vich_causal_variant(source="cleaned"),
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
    # THE VETO REGION AS THE PRODUCTION CODE BUILDS IT.
    #
    # oscNext_L3.py (the real, uncommented script, now in
    # icetray-oscNext/oscNext/python/selection/) makes its veto series by
    # REMOVING the fiducial DOMs from the whole pulse series:
    #
    #   tray.AddModule("I3OMSelection<I3RecoPulseSeries>", "GenICVetoPulses_0",
    #                  selectInverse  = False,
    #                  InputResponse  = splituncleaned,
    #                  OutputResponse = "OfflinePulsesICVeto",
    #                  OmittedKeys    = DOMList.DeepCoreFiducialDOMs)
    #
    # i.e. "veto" means NOT FIDUCIAL, not a separate list.
    #
    # RETIRED before it was ever run.  DOMS.DOMS("IC86") was measured: 554
    # fiducial DOMs, 4606 veto DOMs, disjoint, 5160 together = 86 x 60, the
    # whole in-ice detector.  So DeepCoreVetoDOMs ALREADY IS the complement of
    # the fiducial list and these variants are identical to the production
    # _vich for any in-ice pulse series.  Kept only so the report shows the
    # equality rather than leaving the question open; they cost a row each.
    "veto_not_fiducial":         _vich_variant(veto_field="notfid"),
    "veto_not_fiducial_no_speed": _vich_variant(veto_field="notfid",
                                                speed_min=None, speed_max=None),
    "veto_not_fid_cog_all":      _vich_variant(veto_field="notfid",
                                               fiducial=False),
    "l3_veto_region":            _vich_variant(veto_field="l3veto"),
    "l3_veto_no_speed":          _vich_variant(veto_field="l3veto",
                                               speed_min=None, speed_max=None),
    "l3_veto_l3_cog":            _vich_variant(veto_field="l3veto",
                                               cog_field="l3fid"),
    "l3_veto_l3_cog_no_speed":   _vich_variant(veto_field="l3veto",
                                               cog_field="l3fid",
                                               speed_min=None, speed_max=None),
    "ref_first_hlc":             _vich_ref_variant(_first_hlc_ref),
    "ref_first_hlc_no_speed":    _vich_ref_variant(_first_hlc_ref,
                                                   speed_min=None,
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


# Technical note Table 7 (p.27): the DeepCore LEVEL 3 fiducial volume.
#   DeepCore strings 79-86, DOMs 11-60
#   IceCube strings 25-27, 34-37, 44-47, 54, DOMs 39-60
# Everything else is a veto DOM *for L3*.  This is a DIFFERENT region from the
# DeepCore Filter's (L2) one that our VICH uses, and the note says outright
# (p.60, line 880) that "some algorithms used throughout their event selection
# may use differing definitions of what precisely constitutes the veto region".
L3_FID_DEEPCORE_STRINGS = tuple(range(79, 87))
L3_FID_ICECUBE_STRINGS = (25, 26, 27, 34, 35, 36, 37, 44, 45, 46, 47, 54)


def _l3_fiducial(string, om):
    """Table 7 membership, as boolean arrays."""
    dc = np.isin(string, L3_FID_DEEPCORE_STRINGS) & (om >= 11) & (om <= 60)
    ic = np.isin(string, L3_FID_ICECUBE_STRINGS) & (om >= 39) & (om <= 60)
    return dc | ic


def _arrays(pulse_map, geometry, veto_doms, fid_doms):
    """One flat record per pulse.  Returns a dict of numpy arrays."""
    from .variables import iter_map
    from icecube import dataclasses  # noqa: F401

    omgeo = geometry.omgeo
    x, y, z, t, q, dom, hlc, veto, fid = [], [], [], [], [], [], [], [], []
    st, om = [], []
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
            st.append(omkey.string); om.append(omkey.om)
            hlc.append(bool(p.flags & _LC()))
            veto.append(in_veto); fid.append(in_fid)
    f = np.asarray
    st = f(st, np.int64); om = f(om, np.int64)
    l3fid = (_l3_fiducial(st, om) if st.size
             else np.zeros(0, dtype=bool))
    fid_arr = f(fid, bool)
    return {"x": f(x, float), "y": f(y, float), "z": f(z, float),
            "t": f(t, float), "q": f(q, float),
            "dom": f(dom, np.int64), "string": st, "om": om,
            "hlc": f(hlc, bool), "veto": f(veto, bool), "fid": fid_arr,
            # "everything that is NOT fiducial".  This is how the PRODUCTION
            # L3 script builds its veto series -- see the variant note below.
            # MEASURED: it is the SAME set as DeepCore_Filter's own
            # DeepCoreVetoDOMs (554 + 4606 = 5160 = 86 x 60, disjoint), not a
            # wider one, so "veto" and "notfid" agree on in-ice pulses.
            "notfid": ~fid_arr,
            # No region cut at all.  LowEnVariables' VetoCausalHits applies its
            # causality bands to the WHOLE series -- the region is implicit in
            # the bands, not a DOM list -- so a faithful variant needs this.
            "all": np.ones(fid_arr.size, dtype=bool),
            "l3fid": l3fid, "l3veto": ~l3fid}


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

    # Trigger times, for the causal-band variants.  LowEnVariables'
    # VetoCausalHits anchors itself on the hit closest in time to the trigger,
    # not on a COG, so the variant cannot be built from the pulses alone.
    # The key name is not the same in every production: pass3 L3 keeps
    # I3TriggerHierarchy, while a pass2 file may carry only the SuperDST
    # trigger series.  Try them in order and take the first that yields
    # anything, so a rename shows up as "no trigger key" rather than as a
    # silently empty variant.
    tr_cfg, tr_t = [], []
    for tkey in TRIGGER_KEYS:
        if tkey not in frame:
            continue
        try:
            cfg, tt = [], []
            for trig in frame[tkey]:
                cid = getattr(getattr(trig, "key", None), "config_id", None)
                if cid is None:
                    continue
                cfg.append(int(cid))
                tt.append(float(trig.time))
        except Exception:
            continue
        if cfg:
            tr_cfg, tr_t = cfg, tt
            break

    return {"cleaned": _arrays(cln, geo, veto, fid),
            "uncleaned": _arrays(unc, geo, veto, fid),
            "trig_cfg": np.asarray(tr_cfg, dtype=np.int64),
            "trig_t": np.asarray(tr_t, dtype=float),
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
    keys += DIAG_KEYS
    keys += ["L4_accumulated_time", "L4_VICH_nch", "L4_VICH_npulses",
             "L4_VICH_qtot", "L4_first_hlc_rho"]
    return list(dict.fromkeys(keys))


# ---------------------------------------------------------------------------
# 7. The report
# ---------------------------------------------------------------------------

def report(h5path, stream=None):
    """Rank every candidate definition against the pass2 reference."""
    import sys
    from .pass2 import read_pairs, MOSTLY_FRACTION
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
            mark = "  <== pass2" if frac >= MOSTLY_FRACTION else ""
            print("  %-30s %9.2f%% %14.6g %8d%s"
                  % (name, 100 * frac, med, n, mark), file=out)
        if rows:
            top = max(rows, key=lambda r: r[0])
            best[label] = (top[3], top[0])

    print("", file=out)
    for label, (name, frac) in best.items():
        verdict = ("REPRODUCES pass2" if frac >= MOSTLY_FRACTION
                   else "best so far, but NOT the definition")
        print("%-18s -> %-30s %6.2f%%  %s"
              % (label, name, 100 * frac, verdict), file=out)
    return best


# ---------------------------------------------------------------------------
# 8. Inversion: ask what pass2's number IMPLIES
# ---------------------------------------------------------------------------
#
# Adding more guesses has diminishing returns -- eight accumulated_time
# variants and nine VICH variants all failed.  So invert the question: the
# reference value is known per event, so compute what it would have to MEAN
# under our arithmetic.  If the implied quantity concentrates on one value,
# that value is the missing piece; if it is scattered, the whole framing is
# wrong and the scatter says so honestly.

def implied_fraction(t, q, ref):
    """
    The charge fraction accumulated by t[0] + ref.

    If pass2 used these same entries and some fixed fraction f, this returns
    f in every event.  So the median over events IS their fraction, and the
    spread says whether the entry set is right: a right entry set with a
    different fraction gives a tight distribution away from 0.75; a wrong
    entry set gives a broad one.
    """
    if t.size == 0 or not np.isfinite(ref):
        return np.nan
    order = np.argsort(t, kind="stable")
    t, q = t[order], np.maximum(q[order], 0.0)
    total = q.sum()
    if total <= 0:
        return np.nan
    return float(q[t <= t[0] + ref].sum() / total)


ACC_ENTRY_SETS = {
    "all_pulses": lambda c: (c["t"], c["q"]),
    "first_pulse_per_dom": lambda c: _first_pulse_per_dom(c["dom"], c["t"],
                                                          c["q"]),
    "per_dom_charge": lambda c: _per_dom(c["dom"], c["t"], c["q"])[1:],
    "hlc_only": lambda c: (c["t"][c["hlc"]], c["q"][c["hlc"]]),
}


def vich_bounds(ev, source="uncleaned"):
    """
    Ceilings for VICH_nch, to test whether pass2's count can fit inside our
    veto region at all.

    This is the decisive test for the veto-region hypothesis.  No speed
    window, no causality requirement and no COG choice can ever raise the
    count above the number of veto DOMs that were hit -- so if pass2's stored
    value EXCEEDS that ceiling, our veto region is missing DOMs and no amount
    of tuning the cuts will close the gap.
    """
    s = ev[source]
    if s["t"].size == 0:
        return 0.0, 0.0
    veto_hit = float(np.unique(s["dom"][s["veto"]]).size)
    any_hit = float(np.unique(s["dom"]).size)
    return veto_hit, any_hit


def implied_speed_cut(ev, ref, cog=None, source="uncleaned"):
    """
    The upper speed cut that would make our veto hits count to `ref` DOMs.

    Per DOM we take its smallest speed (the easiest one to admit), sort them,
    and read off the ref-th.  If pass2 used an upper speed cut on our veto
    region, this concentrates on their threshold across events.
    """
    s = ev[source]
    if cog is None:
        cog = _cog(ev)
    if cog is None or not np.isfinite(ref) or ref < 1:
        return np.nan
    cx, cy, cz, ct = cog
    m = s["veto"]
    if not m.any():
        return np.nan
    d = np.sqrt((s["x"][m] - cx) ** 2 + (s["y"][m] - cy) ** 2
                + (s["z"][m] - cz) ** 2)
    dt = ct - s["t"][m]
    ok = dt > 0
    if not ok.any():
        return np.nan
    speed = d[ok] / dt[ok]
    dom = s["dom"][m][ok]
    order = np.argsort(dom, kind="stable")
    dom, speed = dom[order], speed[order]
    edges = np.flatnonzero(np.diff(dom)) + 1
    per_dom = np.array([g.min() for g in np.split(speed, edges)])
    per_dom.sort()
    k = int(ref) - 1
    if k < 0 or k >= per_dom.size:
        return np.nan
    return float(per_dom[k])


def diagnose(frame, cleaned_pulses, uncleaned_pulses,
             acc_ref_key="L4_accumulated_time", vich_ref_key="L4_VICH_nch"):
    """Tray module: write the inverted quantities beside the references."""
    from icecube import dataclasses

    ev = extract(frame, cleaned_pulses, uncleaned_pulses)
    if ev is None:
        return True
    c = ev["cleaned"]

    acc_ref = float(frame[acc_ref_key].value) if acc_ref_key in frame else np.nan
    for name, sel in ACC_ENTRY_SETS.items():
        try:
            t, q = sel(c)
            v = implied_fraction(t, q, acc_ref)
        except Exception:
            v = np.nan
        key = FIT_PREFIX + "diag_accfrac_" + _slug(name)
        if key not in frame:
            frame[key] = dataclasses.I3Double(float(v))

    vich_ref = float(frame[vich_ref_key].value) if vich_ref_key in frame else np.nan
    veto_hit, any_hit = vich_bounds(ev)
    for key, val in ((FIT_PREFIX + "diag_veto_doms_hit", veto_hit),
                     (FIT_PREFIX + "diag_all_doms_hit", any_hit),
                     (FIT_PREFIX + "diag_speed_needed",
                      implied_speed_cut(ev, vich_ref))):
        if key not in frame:
            frame[key] = dataclasses.I3Double(float(val))
    return True


DIAG_KEYS = ([FIT_PREFIX + "diag_accfrac_" + _slug(n) for n in ACC_ENTRY_SETS]
             + [FIT_PREFIX + "diag_veto_doms_hit",
                FIT_PREFIX + "diag_all_doms_hit",
                FIT_PREFIX + "diag_speed_needed"])


def diagnose_report(h5path, stream=None):
    """Read the inverted quantities and say what they imply."""
    import sys
    from .pass2 import read_pairs
    out = stream or sys.stdout

    pairs = [(k, k, "value") for k in DIAG_KEYS]
    pairs += [("ref_acc", "L4_accumulated_time", "value"),
              ("ref_vich", "L4_VICH_nch", "value")]
    data, missing = read_pairs(h5path, pairs)
    n = len(data["Run"])
    print("Inverting the pass2 numbers: %s" % h5path, file=out)
    print("events: %d" % n, file=out)
    if missing:
        print("not found: %s" % ", ".join(sorted({m[0] for m in missing})),
              file=out)

    print("\naccumulated_time -- the charge fraction pass2's value implies",
          file=out)
    print("  (0.75 would mean the fraction is right and the ENTRY SET is what "
          "differs)", file=out)
    print("  %-24s %9s %9s %9s %9s %8s"
          % ("entry set", "median", "16%", "84%", "frac@.75", "n"), file=out)
    print("  " + "-" * 74, file=out)
    for name in ACC_ENTRY_SETS:
        v = data[FIT_PREFIX + "diag_accfrac_" + _slug(name)]
        v = v[np.isfinite(v)]
        if v.size == 0:
            print("  %-24s %9s" % (name, "no data"), file=out)
            continue
        at75 = float((np.abs(v - 0.75) < 0.01).mean())
        print("  %-24s %9.4f %9.4f %9.4f %8.1f%% %8d"
              % (name, np.median(v), np.percentile(v, 16),
                 np.percentile(v, 84), 100 * at75, v.size), file=out)

    print("\nVICH_nch -- can pass2's count fit inside our veto region?",
          file=out)
    ref = data["ref_vich"]
    veto = data[FIT_PREFIX + "diag_veto_doms_hit"]
    alld = data[FIT_PREFIX + "diag_all_doms_hit"]
    ok = np.isfinite(ref) & np.isfinite(veto)
    if ok.any():
        over_veto = float((ref[ok] > veto[ok]).mean())
        over_all = float((ref[ok] > alld[ok]).mean())
        print("  pass2 count > DOMs hit in OUR veto region : %6.2f%% of events"
              % (100 * over_veto), file=out)
        print("  pass2 count > ALL hit DOMs in the event    : %6.2f%% of events"
              % (100 * over_all), file=out)
        print("  median pass2 count %.1f, median veto DOMs hit %.1f, "
              "median DOMs hit %.1f"
              % (np.median(ref[ok]), np.median(veto[ok]), np.median(alld[ok])),
              file=out)
        if over_veto > 0.05:
            print("  -> OUR VETO REGION IS TOO SMALL.  No speed window or COG "
                  "choice can reach a count our region cannot hold, so the "
                  "region definition is the thing to fix.", file=out)
        elif over_all > 0.0:
            print("  -> pass2 counts more DOMs than the event has hits, so it "
                  "is not counting hit DOMs the way we assume at all.",
                  file=out)
        else:
            print("  -> the count FITS inside our region, so the region is "
                  "big enough and the selection inside it is what differs.",
                  file=out)

    sp = data[FIT_PREFIX + "diag_speed_needed"]
    sp = sp[np.isfinite(sp)]
    if sp.size:
        print("\n  implied upper speed cut (m/ns), over %d events:" % sp.size,
              file=out)
        print("    median %.4f   16%% %.4f   84%% %.4f   (we use %.2f-%.2f)"
              % (np.median(sp), np.percentile(sp, 16), np.percentile(sp, 84),
                 VICH_SPEED_MIN, VICH_SPEED_MAX), file=out)
        print("    A tight distribution here is their threshold; a broad one "
              "means the speed cut is not what separates us.", file=out)


# ---------------------------------------------------------------------------
# 9. Grid scans
# ---------------------------------------------------------------------------
#
# The inversion narrowed both failures to one parameter each, so sweep that
# parameter instead of hand-picking more variants:
#
#   accumulated_time -- the implied charge fraction sits just BELOW 0.75 in
#     most events, which is the signature of interpolating to the crossing
#     rather than snapping to the pulse that crosses it.  So sweep the
#     fraction AND the snap/interp choice, over each entry set.
#
#   VICH -- the veto region is big enough (pass2's count exceeds our ceiling
#     in 0.16% of events), so the selection inside it is what differs.  The
#     implied one-sided cut came out at 0.2075, below our 0.25 lower bound,
#     but broad -- which a one-sided construction would produce even for a
#     two-sided truth.  So sweep the window in both directions at once.
#
# A grid that peaks near 100% names the definition.  A grid whose best cell is
# still ~15% proves the parameter is not what separates us, which is worth as
# much: it retires the hypothesis instead of leaving it open.

ACC_FRACTIONS = (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
VICH_SPEED_MINS = (None, 0.0, 0.10, 0.15, 0.20, 0.25)
VICH_SPEED_MAXS = (None, 0.30, 0.35, 0.40, 0.50, 1.00)


def _acc_grid_fn(sel, fraction, mode):
    def f(ev):
        t, q = sel(ev["cleaned"])
        if t.size == 0:
            return np.nan
        if mode != "interp":
            return _cum_fraction_time(t, q, fraction, mode=mode)
        order = np.argsort(t, kind="stable")
        tt, qq = t[order], np.maximum(q[order], 0.0)
        total = qq.sum()
        if total <= 0:
            return np.nan
        cum = np.cumsum(qq) / total
        if cum[0] >= fraction:
            return 0.0
        return float(np.interp(fraction, cum, tt) - tt[0])
    return f


ACC_MODES = ("at", "prev", "interp")


def acc_grid():
    out = {}
    for es_name, sel in ACC_ENTRY_SETS.items():
        for fr in ACC_FRACTIONS:
            for mode in ACC_MODES:
                name = "%s@%.2f%s" % (es_name, fr,
                                      "" if mode == "at" else "_" + mode)
                out[name] = _acc_grid_fn(sel, fr, mode)
    return out


def vich_grid():
    out = {}
    for lo in VICH_SPEED_MINS:
        for hi in VICH_SPEED_MAXS:
            if lo is not None and hi is not None and lo >= hi:
                continue
            name = "s_%s_%s" % ("none" if lo is None else ("%.2f" % lo),
                                "none" if hi is None else ("%.2f" % hi))
            out[name] = _vich_variant(speed_min=lo, speed_max=hi)
    return out


ACC_GRID = acc_grid()
VICH_GRID = vich_grid()


def grid_keys():
    keys = [FIT_PREFIX + "gacc_" + _slug(n) for n in ACC_GRID]
    keys += [FIT_PREFIX + "gvich_" + _slug(n) for n in VICH_GRID]
    return keys


def grid_variants(frame, cleaned_pulses, uncleaned_pulses):
    """Tray module: evaluate every grid cell."""
    from icecube import dataclasses
    ev = extract(frame, cleaned_pulses, uncleaned_pulses)
    if ev is None:
        return True
    for name, fn in ACC_GRID.items():
        key = FIT_PREFIX + "gacc_" + _slug(name)
        if key not in frame:
            frame[key] = dataclasses.I3Double(float(_safe(fn, ev)))
    for name, fn in VICH_GRID.items():
        key = FIT_PREFIX + "gvich_" + _slug(name)
        if key in frame:
            continue
        try:
            n_dom, _, _ = fn(ev)
        except Exception:
            n_dom = np.nan
        frame[key] = dataclasses.I3Double(float(n_dom))
    return True


def grid_report(h5path, top=12, stream=None):
    import sys
    from .pass2 import read_pairs, MOSTLY_FRACTION
    out = stream or sys.stdout

    pairs = [("ref_acc", "L4_accumulated_time", "value"),
             ("ref_vich", "L4_VICH_nch", "value")]
    pairs += [("acc:" + n, FIT_PREFIX + "gacc_" + _slug(n), "value")
              for n in ACC_GRID]
    pairs += [("vich:" + n, FIT_PREFIX + "gvich_" + _slug(n), "value")
              for n in VICH_GRID]
    data, missing = read_pairs(h5path, pairs)

    print("Grid scan: %s" % h5path, file=out)
    print("events: %d" % len(data["Run"]), file=out)
    if missing:
        print("not found: %d key(s) -- rerun `fit` with --grid"
              % len(missing), file=out)

    for label, prefix, grid, ref_key in (
            ("accumulated_time", "acc:", ACC_GRID, "ref_acc"),
            ("VICH_nch", "vich:", VICH_GRID, "ref_vich")):
        ref = data[ref_key]
        rows = []
        for name in grid:
            frac, med, n = score(ref, data[prefix + name])
            rows.append((frac, med, n, name))
        rows.sort(key=lambda r: -r[0])
        print("\n%s -- best %d of %d cells" % (label, min(top, len(rows)),
                                               len(rows)), file=out)
        print("  %-28s %10s %14s" % ("cell", "agree", "median|diff|"),
              file=out)
        print("  " + "-" * 56, file=out)
        for frac, med, n, name in rows[:top]:
            mark = "  <== pass2" if frac >= MOSTLY_FRACTION else ""
            print("  %-28s %9.2f%% %14.6g%s"
                  % (name, 100 * frac, med, mark), file=out)
        if rows and rows[0][0] <= 0.5:
            print("  -> the best cell is only %.1f%%, so this parameter is NOT "
                  "what separates us: the sweep retires it."
                  % (100 * rows[0][0]), file=out)
