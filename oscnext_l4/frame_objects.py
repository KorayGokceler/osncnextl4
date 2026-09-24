"""
Our counterparts of the production's `oscNext/python/frame_objects/`.

This meta-project has no `icecube.oscNext`, so the handful of frame-level
helpers the L4 tray needs are rewritten here.  One file mirrors their whole
directory, and each section below is named after the file it stands in for --
so it can still be read beside the original, which is the point.

    geom.py       -> calc_rho_36
    pulses.py     -> iter_map, get_pulses
    weighting.py  -> PropagateGenieInfo
"""

import numpy as np

from icecube import dataclasses, icetray


# -------------------------------------------------------------------------
# geom -- detector geometry
# oscNext/python/frame_objects/geom.py
# -------------------------------------------------------------------------

# Position of string 36 (the DeepCore centre) -- the reference point of rho_36.
STRING36_X = 46.29
STRING36_Y = -34.88


def calc_rho_36(x, y):
    '''
    Horizontal radial distance from string 36 (the DeepCore centre).

    VERBATIM from the official project, oscNext/frame_objects/geom.py:

        return np.sqrt( (x-46.29) ** 2 + (y+34.88) ** 2 )

    The constants were already right; the SQUARE ROOT was not.  We used
    np.hypot, which is a different algorithm (it rescales to avoid overflow)
    and disagrees with the naive form in the last bit.  Measured against pass2:
    our rho matched bitwise in 80.6% of 8144 events and to 1e-6 in 99.85% --
    the gap was this, not a different definition.  Using their form makes it
    exact.
    '''
    return float(np.sqrt((x - STRING36_X) ** 2 + (y - STRING36_Y) ** 2))

# The DeepCore DOM lists, built once.  DOMS.DOMS("IC86") is not cheap, and it
# used to be rebuilt on every frame inside VICH.  These lists are the
# DEFINITION of micro_count's fiducial selection; there is deliberately no
# fallback that guesses them.
_doms_cache = {}


def deepcore_doms(detector="IC86"):
    '''icecube.DeepCore_Filter.DOMS.DOMS(detector), cached.'''
    if detector not in _doms_cache:
        from icecube.DeepCore_Filter import DOMS
        _doms_cache[detector] = DOMS.DOMS(detector)
    return _doms_cache[detector]

# -------------------------------------------------------------------------
# pulses -- pulse-series helpers
# oscNext/python/frame_objects/pulses.py
# -------------------------------------------------------------------------

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
    '''Get the pulse series; apply it to the frame when it is a mask/union.'''
    if key not in frame:
        return None
    obj = frame[key]
    if hasattr(obj, "apply"):
        try:
            return obj.apply(frame)
        except Exception:
            return None
    return obj

# -------------------------------------------------------------------------
# weighting -- weight bookkeeping carried through the tray
# oscNext/python/frame_objects/weighting.py
# -------------------------------------------------------------------------

L4_NFLUX_KEY = "L4_n_flux_events"   # carried from I3GenieInfo into the P frame


class PropagateGenieInfo(icetray.I3Module):
    '''
    Write I3GenieInfo.n_flux_events into every Physics frame as an I3Double.

    WHY: I3GenieInfo appears once per file, in a NON-Physics frame (S/M).
    HDF5 booking is per event, so unless the value is carried into every P
    frame it cannot be used in the weight calculation.

    Weight convention (the same as the existing oscnext_rates.py):
        weight [Hz] = OneWeight * flux(E) / n_flux
        n_flux      = I3GenieInfo.n_flux_events                (when present)
                    = NEvents * 0.7 (nu) or 0.3 (nubar)        (otherwise)
    '''

    def __init__(self, context):
        icetray.I3Module.__init__(self, context)
        self.AddParameter("OutputKey", "frame key to write", L4_NFLUX_KEY)
        self.AddOutBox("OutBox")

    def Configure(self):
        self.output_key = self.GetParameter("OutputKey")
        self.n_flux = None
        self.warned = False
        self.n_seen = 0          # how many I3GenieInfo seen (= how many L3 files)

    def _grab(self, frame):
        '''
        Try to read I3GenieInfo.  A deserialisation error (genie_icetray not
        imported) or a missing field must NOT kill the job -- the weight
        calculation falls back to NEvents.

        CAREFUL: every L3 FILE has its own I3GenieInfo and one tray processes
        several files (--chunk-files).  The value must be UPDATED at every new
        I3GenieInfo.  It used to be read once and frozen, so the first file's
        n_flux_events was applied to the events of every later file and their
        weights came out wrong.
        '''
        if not frame.Has("I3GenieInfo"):
            return
        try:
            new_val = float(frame["I3GenieInfo"].n_flux_events)
            self.n_seen += 1
            if self.n_flux is not None and new_val != self.n_flux:
                icetray.logging.log_info(
                    "PropagateGenieInfo: n_flux_events changed %g -> %g "
                    "(file %d) -- updating"
                    % (self.n_flux, new_val, self.n_seen))
            self.n_flux = new_val
            icetray.logging.log_info(
                "PropagateGenieInfo: n_flux_events = %g" % self.n_flux)
        except Exception as e:
            if not self.warned:
                icetray.logging.log_warn(
                    "PropagateGenieInfo: could not read I3GenieInfo (%s: %s). "
                    "Was genie_icetray imported?  The weight calculation will "
                    "fall back to NEvents * nu/nubar fractions."
                    % (type(e).__name__, e))
                self.warned = True

    # NOTE: because Process() is overridden, stream methods such as DAQ() and
    # Simulation() are NEVER CALLED -- the Process below does the dispatch and
    # calls _grab on every stream.  (Both used to exist here; DAQ and
    # Simulation were dead code.)
    def Process(self):
        frame = self.PopFrame()
        if frame.Stop != icetray.I3Frame.Physics:
            self._grab(frame)
            self.PushFrame(frame)
            return
        self._grab(frame)
        if self.n_flux is not None and self.output_key not in frame:
            frame[self.output_key] = dataclasses.I3Double(self.n_flux)
        elif self.n_flux is None and not self.warned:
            icetray.logging.log_warn(
                "PropagateGenieInfo: no I3GenieInfo found -- the weight "
                "calculation will fall back to NEvents * nu/nubar fractions")
            self.warned = True
        self.PushFrame(frame)
