'''
L3 variables this pipeline has to produce itself.

`FullTimeLengthRatio` is an L3 variable (the production writes it in
`oscNext_L3.py` as CleanedFullTimeLength / UncleanedFullTimeLength), and pass2's
L3 map carries it.  The pass3 L3 map carries only the two components, so the
ratio is divided out here at L4 instead.  Verified against the stored pass2
value: identical over 8144 events.
'''

import numpy as np

from .env import require_icetray
from .pulses import iter_map, get_pulses

require_icetray()
from icecube import dataclasses, icetray

L4_FTLR_KEY = "L4_FullTimeLengthRatio"


def _full_time_length_ratio(frame, output_key,
                            l3_key="IC2018_LE_L3_Vars",
                            cleaned_pulses=None, uncleaned_pulses=None):
    '''
    Ratio of cleaned to uncleaned event duration.  A noise BDT input.

    Technical note Table 11: "measure the total duration of the event in both
    the cleaned and the uncleaned pulse series (duration = max pulse time -
    min pulse time) and take their ratio."

    Direction: cleaned / uncleaned, i.e. within [0, 1] -- consistent with the
    x axis of Figure 13.

    MEASURED BEHAVIOUR (on one L3 file each: 126 nue, 17 noise).  The physics
    story in the earlier comment was WRONG and is corrected here:

                          ratio (median)   cleaned     uncleaned
        nue                        0.16     1626 ns     10100 ns
        noise (passing L3)         0.27     2780 ns     10290 ns

      * The ratio never approaches 1.  The uncleaned duration is ~10 us in
        every event, because SplitInIcePulses spans the whole readout window
        and noise hits are everywhere.  So the variable is effectively
        "cleaned duration / 10 us".
      * The separating direction is the OPPOSITE of what one expects: the
        cleaned series of noise events is LONGER than that of nue (a low
        energy cascade is compact in time, whereas a noise event that survived
        L3 is a few hits spread out in time).  It does separate, but not as
        "real event ~1 / noise ~0".
        (17 noise events is few -- the ordering is what this file shows, while
        the magnitude finding is structural and solid.)

    The pass3 L3 output has CleanedFullTimeLength and UncleanedFullTimeLength
    SEPARATELY inside IC2018_LE_L3_Vars but NOT their ratio, which is why it is
    computed here.  When they are absent from the L3 map it is measured
    directly from the pulse series.
    '''
    if output_key in frame:
        return True

    cleaned = uncleaned = None

    if l3_key in frame:
        v = frame[l3_key]
        if "CleanedFullTimeLength" in v and "UncleanedFullTimeLength" in v:
            cleaned = float(v["CleanedFullTimeLength"])
            uncleaned = float(v["UncleanedFullTimeLength"])

    if cleaned is None and cleaned_pulses and uncleaned_pulses:
        def duration(key):
            pmap = get_pulses(frame, key)
            if pmap is None:
                return None
            times = [p.time for _, pulses in iter_map(pmap) for p in pulses]
            return (max(times) - min(times)) if times else None
        cleaned = duration(cleaned_pulses)
        uncleaned = duration(uncleaned_pulses)

    if cleaned is None or uncleaned is None or uncleaned <= 0:
        return True

    # `uncleaned <= 0` prevents division by zero but does NOT prevent NaN:
    # NaN <= 0 is False, so a NaN denominator passed the check and silently
    # wrote NaN into the result.  Check the result itself -- inf and NaN are
    # both rejected, the variable is not written, and it shows up as missing
    # in the HDF5 (better than a wrong number).
    ratio = float(cleaned) / float(uncleaned)
    if not np.isfinite(ratio):
        return True

    # The cleaned series is a subset of the uncleaned one, so the ratio must
    # be <= 1.  If it exceeds that, the two durations are being measured from
    # different base series -- do not let it pass in silence.
    if ratio > 1.0 and not getattr(_full_time_length_ratio, "_warned", False):
        _full_time_length_ratio._warned = True
        icetray.logging.log_warn(
            "FullTimeLengthRatio > 1 (%.3f): the cleaned duration (%.1f ns) "
            "exceeds the uncleaned one (%.1f ns).  The two may not be measured "
            "from the same base series." % (ratio, cleaned, uncleaned))

    frame[output_key] = dataclasses.I3Double(ratio)
    return True
