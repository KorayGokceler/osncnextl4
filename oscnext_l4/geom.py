'''
Detector geometry helpers.

The counterpart of the production's `oscNext/python/frame_objects/geom.py`,
which this meta-project does not have.
'''

import numpy as np

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
