'''
Replaces `tray.AddModule("CalculateVariables", ...)` -- the C++ module
`analysis/private/analysis/event_selection/CalculateVariables.cxx`, from the
`analysis` project this meta-project does not carry.

The original writes one compound `Variables` object and the L4 script extracts
two fields from it; here each field is computed by its own function.
'''

import numpy as np

from ..env import require_icetray
from ..pulses import iter_map, get_pulses

require_icetray()
from icecube import dataclasses


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
