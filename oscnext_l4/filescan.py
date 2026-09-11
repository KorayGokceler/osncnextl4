"""
Find corrupt .i3 input files before they take a whole tray down.

I3Reader takes the entire file list at once, so ONE truncated file kills the
run.  This module is the pre-flight check; scripts/process_L4.py calls it
through --scan and scripts/scan_files.py exposes it standalone.

It lives in the package rather than in process_L4.py because two scripts need
it, and a script importing another script is a fragile arrangement.
"""

import os

from .env import require_icetray

require_icetray()
from icecube import dataio          # noqa: E402  -- after require_icetray


def validate_files(paths, n_frames=25, verbose=True):
    """
    Open each file and read its first `n_frames` frames.  Returns
    (healthy, corrupt).

    n_frames=0 -> read the WHOLE file (slow but certain).

    Note: this does not prove a file is entirely sound -- corruption in the
    middle is only caught by a full scan.  But the failures seen in practice
    (truncated writes) show up in the first few frames.
    """
    good, bad = [], []
    for i, path in enumerate(paths):
        try:
            if os.path.getsize(path) == 0:
                bad.append((path, "empty file (0 bytes)"))
                continue
        except OSError as e:
            bad.append((path, "stat: %s" % e))
            continue
        try:
            f = dataio.I3File(path)
            try:
                n = 0
                while f.more():
                    f.pop_frame()
                    n += 1
                    if n_frames and n >= n_frames:
                        break
            finally:
                f.close()
            good.append(path)
        except Exception as e:
            first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            bad.append((path, first[:160]))
        if verbose and (i + 1) % 100 == 0:
            print("  scanned: %d/%d" % (i + 1, len(paths)))
    return good, bad
