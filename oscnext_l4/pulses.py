'''
Pulse-series helpers.

The counterpart of the production's `oscNext/python/frame_objects/pulses.py`.
Both functions exist because a pulse series in the frame may be a
`I3RecoPulseSeriesMapMask` rather than the map itself, and because a map
iterates differently across icetray versions.
'''

from .env import require_icetray

require_icetray()
from icecube import dataclasses

# A pulse belongs to a local-coincidence (HLC) hit when this flag is set.
_LC_FLAG = dataclasses.I3RecoPulse.PulseFlags.LC


def iter_map(pulse_map):
    '''
    Iterate over (omkey, pulses) pairs.

    CAREFUL: depending on the IceTray version, iterating an
    I3RecoPulseSeriesMap directly may yield KEYS rather than pairs.  Then
    "for omkey, pulses in pmap" fails with "too many values to unpack
    (expected 2)", because an OMKey unpacks into three components
    (string, om, pmt).  .items() is correct in every version.
    '''
    if pulse_map is None:
        return []
    try:
        return pulse_map.items()
    except AttributeError:
        return iter(pulse_map)


def get_pulses(frame, key):
    '''Pulse serisini al; mask/union ise frame'e uygula.'''
    if key not in frame:
        return None
    obj = frame[key]
    if hasattr(obj, "apply"):
        try:
            return obj.apply(frame)
        except Exception:
            return None
    return obj
