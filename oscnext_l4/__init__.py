'''
oscNext L4 -- Level 3 to Level 4 processing and classifier training.

The library lives here; the command line entry points are in ../scripts and
the notebook interface in ../notebooks.  Nothing in this package imports
IceTray at module level: `env` reports why an import failed instead of
letting a missing project take the whole repository down.
'''

__all__ = ["env", "variables", "frame_objects", "rewritten", "l3vars",
           "booker", "data", "varmap", "runner", "classifier", "filescan"]
