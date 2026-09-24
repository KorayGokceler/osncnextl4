'''
oscNext L4 -- Level 3 to Level 4 processing and classifier training.

The library lives here; the command line entry points are in ../scripts and
the notebook interface in ../notebooks.  Modules import `from icecube import
...` directly -- this runs inside an icetray environment, always.  What is NOT
guaranteed is an individual PROJECT (`oscNext`, `tau_bdt`, `slc-veto`, ...),
so those go through `env.optional_project`, which keeps one absent project
from making the whole package unimportable.
'''

import os as _os

# The modules that are HERE, rather than a hand-kept list: the release
# (scripts/make_release.py) ships only the ones its scripts import, and a
# fixed list would advertise modules it left out.
__all__ = sorted(_f[:-3] for _f in _os.listdir(_os.path.dirname(__file__))
                 if _f.endswith(".py") and _f != "__init__.py")
