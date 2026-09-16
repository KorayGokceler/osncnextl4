'''
Replaces `tray.AddModule(I3CutL7Module, ...)` -- `tau-bdt/python/I3CutL7Module.py`
from the `tau_bdt` project this meta-project does not carry.

Veto Identified Causal Hits.  Four conditions in the (distance, dt) plane around
the pulse closest in time to the first matching trigger.  100.00% against the
real pass2 L4 files over 8144 events on all three outputs.
'''

import numpy as np

from ..env import require_icetray
from ..pulses import iter_map, get_pulses

require_icetray()
from icecube import dataclasses


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
