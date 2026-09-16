'''
Replaces `tray.AddModule("FirstHLC<I3RecoPulse>", ...)` -- the C++ module
`SimpleVertex/private/SimpleVertex/FirstHLC.cxx`, which this meta-project does
not carry.

It reproduces TWO bugs of the original deliberately; see the function docstring
and CLAUDE.md open risk 2a.
'''

from ..env import require_icetray
from ..pulses import iter_map, get_pulses, _LC_FLAG

require_icetray()
from icecube import dataclasses

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
