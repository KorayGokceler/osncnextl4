"""
Pure-Python rewrites of icetray modules this meta-project does not have.

Each one replaces a single `tray.AddModule(...)` line of the original L4
script, and each was verified against the real pass2 L4 output:

    first_hlc.py  <- SimpleVertex      FirstHLC<I3RecoPulse>   (bitwise identical)
    dunkman.py    <- analysis          CalculateVariables      (accumulated_time
                                       99.84%, separation not a BDT input)
    vich.py       <- tau_bdt           I3CutL7Module           (100.00%)

Read the docstrings before changing anything: they carry where each definition
came from, which reading of it the production actually used, and what the
measurement was.
"""
