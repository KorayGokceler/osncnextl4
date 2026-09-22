"""
Pure-Python rewrites of the icetray modules this meta-project does not have.

Each section below replaces a single `tray.AddModule(...)` line of the
original L4 script, and each was verified against the real pass2 L4 output:

    first_hlc  <- SimpleVertex  FirstHLC<I3RecoPulse>  (bitwise identical)
    dunkman    <- analysis      CalculateVariables     (accumulated_time 99.84%,
                                                        separation not a BDT input)
    vich       <- tau_bdt       I3CutL7Module          (100.00%)

They were three files in a package; they are one file because they exist for
one reason.  READ THE DOCSTRINGS BEFORE CHANGING ANYTHING: they carry where
each definition came from, which of the several readings the production
actually ran, and what the measurement was.
"""

import numpy as np

from .frame_objects import iter_map, get_pulses, _LC_FLAG

from icecube import dataclasses


# =========================================================================
# first_hlc
# replaces: SimpleVertex  FirstHLC<I3RecoPulse>
# =========================================================================

'''
Replaces `tray.AddModule("FirstHLC<I3RecoPulse>", ...)` -- the C++ module
`SimpleVertex/private/SimpleVertex/FirstHLC.cxx`, which this meta-project does
not carry.

It reproduces TWO bugs of the original deliberately; see the function docstring
and CLAUDE.md open risk 2a.
'''

# What the original writes when the event has NO HLC hit: GetFirstHLCHit returns
# a bool that Physics() ignores, so the search's starting value survives into
# the frame.  first_hlc_rho then comes out as 1407.3.
FIRST_HLC_SENTINEL_POS = (1000.0, 1000.0, 1000.0)
FIRST_HLC_SENTINEL_TIME = 1e10


def _first_hlc(frame, pulses_key, output_key, geometry_key="I3Geometry"):
    '''
    Earliest HLC hit of the cleaned series, written as an I3Particle.

    VERIFIED against the module the production actually ran:
    `SimpleVertex/private/SimpleVertex/FirstHLC.cxx`, registered as
    `I3_MODULE(FirstHLC<I3RecoPulse>)`, which is what
    `tray.AddModule("FirstHLC<I3RecoPulse>", ...)` instantiates.  (A Python
    class of the same name exists in `analysis/python/yanez/FirstHLC.py`, but
    its parameters are `InputPulseSeries`/`Vertex` where the L4 tray passes
    `HitSeriesName`/`OutputName`, so it is a different author's module and not
    the one that produced pass2.)

    Two things came out of reading it, and this rewrite had BOTH wrong:

    1. TIES GO TO THE LAST DOM, not the first.  The C++ reads

           if (hitTime > hlc_time) continue;
           hlc_time = hitTime;  hlc_position = dom_position;

       -- it skips only a STRICTLY later hit, so a hit at exactly the current
       best time overwrites it.  Iteration is over an I3Map, i.e. ascending
       OMKey, so the highest OMKey among tied hits wins.  This code used
       `p.time < best[0]`, which keeps the first.  That is the whole of the
       146 m disagreements: a tie resolved to a different string.

    2. AN EVENT WITH NO HLC HIT STILL GETS A VERTEX -- the sentinel one.  See
       FIRST_HLC_SENTINEL_POS above.

    Everything else matches: the C++ scans every pulse of every DOM with no
    early exit (equivalent to stopping at a DOM's first HLC pulse, since
    pulses are time-ordered within a DOM), skips DOMs absent from the geometry,
    takes the DOM position rather than anything charge weighted, and sets
    fit_status OK.
    '''
    if output_key in frame:
        return True
    pmap = get_pulses(frame, pulses_key)
    if pmap is None or geometry_key not in frame:
        return True
    omgeo = frame[geometry_key].omgeo

    best_time = FIRST_HLC_SENTINEL_TIME
    best_pos = None
    for omkey, pulses in iter_map(pmap):
        if omkey not in omgeo:
            continue
        for p in pulses:
            if not (p.flags & _LC_FLAG):
                continue
            # `<=`, not `<`: a tie overwrites, so the last DOM in OMKey order
            # wins.  This mirrors the original's `if (hitTime > hlc_time)
            # continue`.
            if p.time <= best_time:
                best_time = p.time
                best_pos = omgeo[omkey].position
            break          # a DOM's first HLC pulse is its earliest

    part = dataclasses.I3Particle()
    if best_pos is None:
        part.pos = dataclasses.I3Position(*FIRST_HLC_SENTINEL_POS)
        part.time = FIRST_HLC_SENTINEL_TIME
    else:
        part.pos = best_pos
        part.time = best_time
    part.shape = dataclasses.I3Particle.ParticleShape.Cascade
    part.fit_status = dataclasses.I3Particle.FitStatus.OK
    frame[output_key] = part
    return True


# =========================================================================
# dunkman
# replaces: analysis      CalculateVariables
# =========================================================================

'''
Replaces `tray.AddModule("CalculateVariables", ...)` -- the C++ module
`analysis/private/analysis/event_selection/CalculateVariables.cxx`, from the
`analysis` project this meta-project does not carry.

The original writes one compound `Variables` object and the L4 script extracts
two fields from it; here each field is computed by its own function.
'''

def _accumulated_time(frame, pulses_key, output_key, fraction=0.75,
                      before_crossing=True):
    '''
    Time [ns] from the first pulse to the point where the event has ~75% of
    its charge.

    VERIFIED against the module the production ran:
    `analysis/private/analysis/event_selection/CalculateVariables.cxx`
    (`I3_MODULE(CalculateVariables)`), which the L4 tray calls as
    `tray.AddModule("CalculateVariables", "DunkVars", PulseSeries=cleaned)`.

    THE PRODUCTION DOES NOT LOOK FOR A 75% CROSSING AT ALL.  It bins the
    time-sorted pulses into CHARGE QUARTILES and assigns the variable inside
    the third one:

        bin_edges = [0, Q/4, Q/2, 3Q/4, Q]
        accumulated_charge += current_charge
        part_index = index of the first edge >= accumulated_charge

        } else if ( part_index == 3 ) {
            variables.accumulated_time = pulse.GetTime() - first_slc.t();

    `part_index == 3` means `0.5Q < accumulated_charge <= 0.75Q`, and the line
    runs for EVERY pulse in that band, so what survives is the LAST pulse
    whose cumulative charge has not yet passed 0.75Q.

    That is what the pass2 fit found empirically -- "the pulse before the
    crossing", 99.40% over 8144 events against 0.98% for the crossing itself.
    It is therefore NOT an off-by-one bug, as this docstring previously
    claimed: it is what quartile binning gives when the variable is written in
    the Q3 branch.  Two details the fit could not see, both fixed here:

      * the bound is `<=`, not `<` (lower_bound returns bin_edges[3] when the
        cumulative charge lands exactly on 0.75Q), where searchsorted's
        default gave the pulse before that one;
      * when NO pulse falls in the Q3 band the variable keeps its default,
        which is why pass2 stores exactly 0 in a fraction of events.  That
        needs one pulse to carry the cumulative sum from below 0.5Q to above
        0.75Q, i.e. more than a QUARTER of the event's charge -- not three
        quarters, as was written here before.

    `first_slc` is the first entry of the time-sorted list, HLC or not, so the
    zero point is the earliest pulse -- the same `t[0]` used here.

    Charges are summed as they come, with no clamping of negatives, as the
    original does.

    `before_crossing=True` (DEFAULT) is the production rule above.
    `before_crossing=False` follows technical note Table 12 instead -- "time to
    reach 75% of an event's charge" -- taking the first pulse that reaches the
    fraction.  That agrees with pass2 in 0.98% of events.

    ONE DEVIATION REMAINS.  The original sorts with `std::sort`, which is not
    stable, so pulses sharing a time can land in either order; this uses a
    stable sort so the index is reproducible run to run.  There is no way to
    reproduce an unspecified order, and a reproducible answer is worth more.
    '''
    if output_key in frame:
        return True
    pmap = get_pulses(frame, pulses_key)
    if pmap is None:
        return True

    n_doms = 0
    times, charges = [], []
    # The total is summed HERE, in map order, one pulse at a time -- because
    # that is what the original does, in its first loop over the map:
    #
    #     variables.total_charge += current_charge;      // OMKey order
    #
    # while the cumulative sum below runs in TIME order.  Two different
    # summation orders give totals that differ in the last bit, and
    # `np.sum` differs again because it sums pairwise.  That matters: the
    # quartile edge is 0.75 * total, so a last-bit shift moves the boundary
    # and flips the index in events whose cumulative charge lands on it.
    total = 0.0
    for _, pulses in iter_map(pmap):
        n_doms += 1
        for p in pulses:
            times.append(p.time)
            charges.append(p.charge)
            total += p.charge

    # `if (map_of_pulses.size() <= 4) { PushFrame; return; }` -- with four or
    # fewer DOMs the original writes no Variables object at all, so the L4
    # script's ExtractDunk finds nothing and neither key reaches the frame.
    if n_doms <= 4 or not times:
        return True

    if not np.isfinite(total) or total <= 0:
        return True

    t = np.asarray(times)
    q = np.asarray(charges)
    order = np.argsort(t, kind="stable")
    t, q = t[order], q[order]
    cum = np.cumsum(q)          # sequential, like the original's running sum

    if before_crossing:
        # The Q3 band: 0.5Q < cumulative <= 0.75Q, last pulse in it.
        in_q3 = (cum > 0.5 * total) & (cum <= 0.75 * total)
        if not in_q3.any():
            # No pulse landed in Q3, so the struct default survives.
            frame[output_key] = dataclasses.I3Double(0.0)
            return True
        idx = int(np.flatnonzero(in_q3)[-1])
    else:
        idx = min(int(np.searchsorted(cum, fraction * total)), t.size - 1)

    frame[output_key] = dataclasses.I3Double(float(t[idx] - t[0]))
    return True


def _separation_in_cogs(frame, pulses_key, output_key,
                        geometry_key="I3Geometry"):
    '''
    Distance between the charge-weighted COG of the event's FIRST charge
    quartile and that of its FOURTH.

    A track moves on  -> the two COGs sit apart -> large separation.
    A cascade is local -> they coincide          -> small separation.

    VERIFIED against `CalculateVariables.cxx`, which the L4 tray runs as
    `CalculateVariables(PulseSeries=cleaned_pulses)`:

        variables.separation = CalcDistance(variables.cog_q1, variables.cog_q4);

    with `cog_q1` and `cog_q4` accumulated in the same quartile loop that
    produces accumulated_time: pulses are sorted by time, the cumulative
    charge is binned against `[0, Q/4, Q/2, 3Q/4, Q]`, and each pulse updates
    the COG of the quartile it falls in.

    THIS REPLACES A GUESS.  The earlier version split the event into two
    halves by HIT COUNT and compared those two COGs, with a docstring saying
    the definition "could not be verified (the project is unavailable)".  The
    project is available now, and the definition is different in two ways: the
    split is by CHARGE, not by count, and it compares the outer quartiles, not
    two halves -- so the middle half of the charge is excluded entirely.

    Not a BDT input (it is in neither Table 12 nor the production's
    L4_MUON_MODEL_INPUT_VARIABLES), so it is behind --run-optional.
    '''
    if output_key in frame:
        return True
    pmap = get_pulses(frame, pulses_key)
    if pmap is None or geometry_key not in frame:
        return True
    geo = frame[geometry_key]

    n_doms = 0
    hits = []
    total = 0.0          # summed in map order, as the original does
    for omkey, pulses in iter_map(pmap):
        n_doms += 1
        if omkey not in geo.omgeo:
            continue
        pos = geo.omgeo[omkey].position
        for p in pulses:
            hits.append((p.time, p.charge, pos.x, pos.y, pos.z))
            total += p.charge

    # Same guard as accumulated_time: the original computes no Variables at
    # all for four or fewer DOMs.
    if n_doms <= 4 or not hits:
        return True

    if not np.isfinite(total) or total <= 0:
        return True

    hits.sort(key=lambda h: h[0])
    a = np.asarray(hits, dtype=float)
    q = a[:, 1]

    cum = np.cumsum(q)
    q1 = cum <= 0.25 * total
    q4 = cum > 0.75 * total
    if not q1.any() or not q4.any():
        return True

    def cog(mask):
        w = q[mask]
        if w.sum() <= 0:
            return None
        w = w / w.sum()
        return np.array([w @ a[mask, 2], w @ a[mask, 3], w @ a[mask, 4]])

    c1, c4 = cog(q1), cog(q4)
    if c1 is None or c4 is None:
        return True
    frame[output_key] = dataclasses.I3Double(float(np.linalg.norm(c4 - c1)))
    return True


# =========================================================================
# vich
# replaces: tau_bdt       I3CutL7Module
# =========================================================================

'''
Replaces `tray.AddModule(I3CutL7Module, ...)` -- `tau-bdt/python/I3CutL7Module.py`
from the `tau_bdt` project this meta-project does not carry.

Veto Identified Causal Hits.  Four conditions in the (distance, dt) plane around
the pulse closest in time to the first matching trigger.  100.00% against the
real pass2 L4 files over 8144 events on all three outputs.
'''

def _vich(frame, uncleaned_pulses,
          nch_key, npulses_key, qtot_key, geometry_key="I3Geometry",
          trigger_key="I3TriggerHierarchy", config_ids=(1010, 1011)):
    '''
    Veto Identified Causal Hits -- REPRODUCES pass2 EXACTLY (100.00%).

    THE ALGORITHM, AND HOW IT WAS FOUND
    -----------------------------------
    This is `VetoCausalHits` from the pythonic reimplementation of the
    LowEnVariables algorithms used by the GRECO online filter
    (`grecovariables.py`).  The GRECO tray calls it twice on the same
    uncleaned series, extracted pulse by pulse:

        VetoCausalCharge = VetoCausalHits(..., useCharge=True)    -> qtot
        VetoCausalHits   = VetoCausalHits(..., useCharge=False)   -> npulses

    For every trigger with a matching config id, the hit CLOSEST IN TIME to
    that trigger is taken as the reference, and with `dt = t_ref - t_hit` and
    `d` the distance to it, a hit counts when

        d  < 750 m
        dt > -5 d + 500
        dt < d / 0.3 + 150
        dt > d / 0.3 - 1850

    Only the FIRST matching trigger is used, and `TriggerConfigIDs` defaults to
    **[1010, 1011]**, which the L4 tray does not override.

    CONFIRMED AGAINST THE MODULE ITSELF.  The algorithm was found by fitting
    candidate definitions against pass2 and reaching 100.00%; the oscNext_meta
    V01-00-07 build that produced pass2 then turned out to carry
    `tau_bdt/python/I3CutL7Module.py` -- Python, not C++, and exactly what
    `from icecube.tau_bdt import I3CutL7Module` imports.  It settles every
    detail this function had to assume:

      - `break` after the first trigger whose config id is in the list;
      - `nch` counts DOMs with at least one selected pulse (`foundOne`);
      - `npulses` counts the selected pulses;
      - `nVetoHitsTotalPE += pulse.charge`, with no clamping of negatives;
      - the reference is the pulse closest in time to the trigger, ties going
        to the first in map order (a strict `<`, as `np.argmin` does).

    One deviation is deliberate.  The original seeds its reference search with
    `refPulseTime = 0` and `refPulsePos = (0,0,0)`, so if NO pulse is closer to
    the trigger than t=0 is, that sentinel survives and the whole calculation
    is done against the origin at t=0.  With a trigger at ~10 us and pulses in
    the readout window this cannot happen, and it did not in any of the 8144
    verified events; reproducing it would only copy a latent bug.

    A second, structurally identical implementation exists as `CausalTrackVeto`
    in `analysis/python/event_selection/I3CutL7Module_JP_Matt.py`:

        def mightBeBackground(distance, timeDiff):
            return ((distance < 750) and (timeDiff > (-5*distance + 500)) and
                    ((distance/0.3 - 1850) < timeDiff < (distance/0.3 + 150)))

    Its docstring names the physics: "charge that might have caused the
    DeepCore trigger ... direct or scattered light that could have come from a
    muon track approaching the trigger position".  It restricts itself to
    config id 1011 and returns charge only, so it corroborates the bands but
    is NOT the module L4 runs; where the two differ, I3CutL7Module.py wins.
    (GRECO's `VetoCausalHits` is a third implementation of the same bands, and
    it loops over every matching trigger with no `break` -- which is why the
    trigger question had to be settled from the module itself rather than by
    majority.)

    MEASURED against the real pass2 L4 files, 8144 events, all three outputs:
    nch 100.00%, npulses 100.00%, qtot 100.00%, median difference 0.

    WHAT THIS REPLACED, AND WHY THE OLD VERSION COULD NOT WORK
    ---------------------------------------------------------
    The first implementation followed technical note sec. 3.4: for every hit in
    the veto region, the speed between it and the event's COG vertex, kept when
    that speed fell in [0.25, 0.40] m/ns.  It reproduced pass2 in ~12% of
    events with the median difference pinned at 2 DOMs.

    Reading sec. 3.4 in full (p.26-27) showed why: that passage describes the
    LEVEL 2 DeepCore Filter, a different algorithm, which uses the window to
    DISCARD hits.  The note's only description of `L4_VICH_nch` is Table 12's
    one line -- no window, no region, no reference point.

    Five hypotheses were then retired by measurement, all landing at ~12%:
    the speed window (a 36-cell sweep of both edges peaked at 18.0%, with the
    median difference pinned at 2 in EVERY cell -- the signature of sweeping
    the wrong knob), the veto region (closed by construction: DOMS.DOMS("IC86")
    gives 554 fiducial + 4606 veto DOMs, disjoint, 5160 = the whole in-ice
    detector, so DeepCoreVetoDOMs already IS "not fiducial"), the COG
    (unweighted 12.0%, non-fiducial 11.9%), the reference point (first HLC hit,
    11.7%) and the input series (cleaned, 2.8%).

    Three structural differences separate the real algorithm from that one, and
    none of them is reachable by tuning a parameter:

      1. There is NO speed window.  Selection is four conditions in the
         (distance, dt) plane -- a 750 m sphere and three causality bands.
      2. There is NO veto DOM list.  The region is implicit in the bands, so
         the whole series is scanned.  Restricting to the veto DOMs drops
         agreement from 100% to 19.4%.
      3. The reference is the hit nearest the TRIGGER, not a charge-weighted
         fiducial COG.

    Hence this function no longer takes `cleaned_pulses` or `fiducial_cog`:
    neither plays any part.  The uncleaned series was already verified --
    `reference/oscNext_L4_pass2_original.py` passes
    `InputPulses=uncleaned_pulses  # Use uncleaned hits` to `I3CutL7Module`.

    Originally done by tau_bdt.I3CutL7Module, which was never found.
    '''
    if nch_key in frame:
        return True
    unc = get_pulses(frame, uncleaned_pulses)
    if unc is None or geometry_key not in frame:
        return True
    geo = frame[geometry_key]

    trigger_times = []
    if trigger_key in frame:
        for trig in frame[trigger_key]:
            # MERGED and THROUGHPUT triggers carry no config id; only the
            # SIMPLE_MULTIPLICITY one does.  Skipping them matters: reading
            # config_id unguarded raises and loses every trigger in the event.
            cid = getattr(getattr(trig, "key", None), "config_id", None)
            if cid is not None and int(cid) in config_ids:
                trigger_times.append(float(trig.time))

    # Pulse by pulse, as GetHitInformation(..., hitMode=0) does.
    xs, ys, zs, ts, qs, doms = [], [], [], [], [], []
    for omkey, pulses in iter_map(unc):
        if omkey not in geo.omgeo:
            continue
        pos = geo.omgeo[omkey].position
        key = (omkey.string, omkey.om)
        for pulse in pulses:
            xs.append(pos.x); ys.append(pos.y); zs.append(pos.z)
            ts.append(pulse.time); qs.append(pulse.charge); doms.append(key)

    if not ts or not trigger_times:
        frame[nch_key]     = dataclasses.I3Double(0.0)
        frame[npulses_key] = dataclasses.I3Double(0.0)
        frame[qtot_key]    = dataclasses.I3Double(0.0)
        return True

    x = np.asarray(xs); y = np.asarray(ys); z = np.asarray(zs)
    t = np.asarray(ts); q = np.asarray(qs)

    # ONLY THE FIRST matching trigger, per the module's own `break`.  An
    # earlier version looped over every matching trigger and accumulated.  It
    # still agreed with pass2 in 100% of 8144 events, because those events
    # carry exactly one matching trigger (next to a MERGED and a THROUGHPUT
    # one, which have no config id at all) -- so the bug was invisible in the
    # data that verified it, and an event with two would have been double
    # counted.  Found by reading the original, not by measurement; the same
    # goes for the [1010, 1011] default, which pass2 simulation never exercises
    # because 1010 does not appear in it.
    trigger_time = trigger_times[0]
    i = int(np.argmin(np.abs(t - trigger_time)))
    d = np.sqrt((x - x[i]) ** 2 + (y - y[i]) ** 2 + (z - z[i]) ** 2)
    dt = t[i] - t
    sel = d < 750.0
    sel &= dt > (-5.0 * d + 500.0)
    sel &= dt < (d / 0.3 + 150.0)
    sel &= dt > (d / 0.3 - 1850.0)

    n_pulses = float(sel.sum())
    # The original sums pulse.charge as it is -- no clamping of negatives.
    qtot = float(q[sel].sum())
    n_doms = len({doms[j] for j in np.flatnonzero(sel)})

    frame[nch_key]     = dataclasses.I3Double(float(n_doms))
    frame[npulses_key] = dataclasses.I3Double(float(n_pulses))
    frame[qtot_key]    = dataclasses.I3Double(float(qtot))
    return True
