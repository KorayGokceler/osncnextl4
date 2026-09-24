'''
oscNext L4 -- Level 3 to Level 4 processing and classifier training.

The library lives here; the command line entry points are in ../scripts and
the notebook interface in ../notebooks.  Modules import `from icecube import
...` directly -- this runs inside an icetray environment (v1.17.0), always.
The projects that metaproject LACKS (oscNext, tau_bdt, analysis,
SimpleVertex) are never imported: rewritten.py stands in for them.
'''

import os as _os

# The modules that are HERE, rather than a hand-kept list: the release
# (scripts/make_release.py) ships only the ones its scripts import, and a
# fixed list would advertise modules it left out.
__all__ = sorted(_f[:-3] for _f in _os.listdir(_os.path.dirname(__file__))
                 if _f.endswith(".py") and _f != "__init__.py")
