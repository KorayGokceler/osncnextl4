# pybdtmodule.py

__doc__ = """Add a BDT score to I3Frames."""

import os
from icecube import icetray, dataclasses
from pybdt import util


class PyBDTModule (icetray.I3ConditionalModule):

    """Add a BDT score to I3Frames."""

    def __init__ (self, context):
        icetray.I3ConditionalModule.__init__ (self, context)
        self.AddParameter ('BDTFilename', '', None)
        self.AddParameter ('varsfunc', '', None)
        self.AddParameter ('OutputName', '', 'BDTScore')
        self.AddParameter ('UsePurity', '', False)

    def Configure (self):
        self.BDTFilename = self.GetParameter ('BDTFilename')
        self.varsfunc = self.GetParameter ('varsfunc')
        self.OutputName = self.GetParameter ('OutputName')
        self.UsePurity = self.GetParameter ('UsePurity')
        if not os.path.isfile (self.BDTFilename):
            raise ValueError ('BDTModule: BDTFilename must be a valid filename')
        if not self.varsfunc:
            raise ValueError ('BDTModule: must provide varsfunc')
        self.bdtmodel = util.load (self.BDTFilename)

    def Physics (self, frame):
        try:
            vals = self.varsfunc (frame)
            if not hasattr (vals, '__getitem__'):
                icetray.logging.log_error (
                        'varsfunc returned wrong type; mapping required')
                self.PushFrame (frame)
                return
            if vals:
                score = self.bdtmodel.score_event (vals, use_purity=self.UsePurity)
                frame.Put (self.OutputName, dataclasses.I3Double (score))
        except Exception as e:
            icetray.logging.log_warn (str (e)[1:-1])
            pass
        self.PushFrame (frame)

