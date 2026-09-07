import math, numpy, os
from .ml import DataSet
from .util import save
from icecube import icetray


class ConvertDS(icetray.I3ConditionalModule):
    """
    Extract varibales from the frame and store them in a pybdt.DataSet
    Just specify a function which takes a frame as input and returns a dictionary filled with the desired variables.
    This is intended to be the same function as you should later use for the scoring of the events by the pybdtmodule.
    Feel free to add a 'Livetime' or a 'ScaleFactor' (usually number of included simulation-files) to the dataset
    OBSERVE: put the weight of each event to the variable 'weight' in the dictionary
    """
    def __init__(self, ctx):
      icetray.I3ConditionalModule.__init__(self, ctx)
      self.AddParameter("OutfileName", "Write to this file", "")
      self.AddParameter("Livetime", "Add a livetime to the dataset", None)
      self.AddParameter("ScaleFactor", "Divide the total weight by this factor", None)
      self.AddParameter("varsfunc", "Function (Ptr) returning a dictionary of the BDT-variables from the frame")
      self.BDTarrays = {}
      self.pulled_events=0
      
    def Configure(self):
      icetray.I3ConditionalModule.Configure(self)
      self.outfileName = self.GetParameter("OutfileName")
      self.livetime = self.GetParameter("Livetime")
      self.scalefactor = self.GetParameter("ScaleFactor")
      self.varsfunc = self.GetParameter("varsfunc")
      if (self.outfileName==""):
          icetray.logging.log_fatal("Configure 'Outfile'")
      if (self.livetime is not None and (not numpy.isfinite(self.livetime) or self.livetime<=0.)):
          icetray.logging.log_fatal("Configure Livetime correctly")
      if (self.scalefactor is not None and (not numpy.isfinite(self.scalefactor) or self.scalefactor<=0.)):
          icetray.logging.log_fatal("Configure ScaleFactor correctly")
      if (not os.access(os.path.split(self.outfileName)[0], os.W_OK)):
          icetray.logging.log_fatal("Outfile destination inaccessible")
    def Finish(self):
      icetray.logging.log_info('Converting to numpy arrays')
      for key in self.BDTarrays:
          self.BDTarrays[key] = numpy.array (self.BDTarrays[key])
      
      if (self.scalefactor is not None and self.scalefactor!=1.):
        self.BDTarrays['weight'] = self.BDTarrays['weight']/self.scalefactor
      
      if (self.livetime is not None):
        self.BDTarrays["livetime"]=self.livetime
        
      icetray.logging.log_notice('Saving pybdt.ml.DataSet to '+self.outfileName)
      save (DataSet (self.BDTarrays), self.outfileName)
      icetray.logging.log_notice("%d events have been written" %(self.pulled_events))
      
    def Physics(self,frame):
      event = self.varsfunc(frame)

      #initialize the fields once
      if (self.pulled_events == 0):
        self.BDTarrays = dict ((key, []) for key in event.keys())
      
      #put variables in dedicated fields
      for key in event.keys():
        if (not numpy.isfinite(event[key])):
          icetray.logging.log_error("For this event the variable '%s' contains a problematic '%f'-value"%(key, event[key]))
        self.BDTarrays[key].append(event[key])
      
      self.pulled_events+=1
      self.PushFrame(frame)
      return
