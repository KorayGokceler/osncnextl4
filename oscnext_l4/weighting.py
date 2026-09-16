'''
Weight bookkeeping carried through the L4 tray.

The counterpart of the production's `oscNext/python/frame_objects/weighting.py`
-- except that the production reads `NEvents * gen_ratio` where this reads
`I3GenieInfo`, a deviation recorded as open risk 4 in CLAUDE.md.

The production does this in `oscNext_master.py`, not in its L4 segment, which
is why the original L4 script has no counterpart to it.
'''

from .env import require_icetray

require_icetray()
from icecube import dataclasses, icetray

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
