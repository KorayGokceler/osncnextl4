'''
oscNext L4 -- Level 3 to Level 4 processing and classifier training.

The library lives here; the command line entry points are in ../scripts and
the notebook interface in ../notebooks.  Modules import `from icecube import
...` directly -- this runs inside an icetray environment, always.  What is NOT
guaranteed is an individual PROJECT (`oscNext`, `tau_bdt`, `slc-veto`, ...),
so those go through `env.optional_project`, which keeps one absent project
from making the whole package unimportable.
'''

__all__ = ["env", "variables", "frame_objects", "rewritten", "l3vars",
           "tray_io", "data", "dataset", "varmap", "runner", "classifier"]
