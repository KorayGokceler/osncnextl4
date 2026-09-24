"""
The tray's two file ends: what goes in, and what comes out.

Neither computes anything.  `validate_files` is the pre-flight check on the
input -- I3Reader takes the whole file list at once, so ONE truncated file
kills the run -- and `add_booker` is the writer that turns the frames into the
HDF5 the notebook then reads.  `scripts/process_L4.py` uses both, one before
the tray is built and one as its last module; `scan_files.py` and
`compare_pass2.py` use one each.

They live in the package rather than in a script because more than one script
needs them, and a script importing another script is a fragile arrangement.
"""

import os

from icecube import dataio


# -------------------------------------------------------------------------
# In: the pre-flight check on the input files
# -------------------------------------------------------------------------

# Libraries a frame object needs in order to be unpacked.  Never used by name,
# but without them reading a frame fails with "Deserialization failed for
# object at frame key 'X'".  Imported leniently: not every metaproject carries
# every one (genie_reader, for one), and a file only needs those whose objects
# it holds.
#   simclasses     -> I3MCPESeriesMap, noise_weight, ...
#   recclasses     -> I3HitStatisticsValues, I3DST, ...
#   genie_icetray  -> I3GenieInfo, I3GenieResult
_DESERIALIZE_LIBS = ("simclasses", "recclasses", "genie_icetray",
                     "genie_reader", "sim_services", "phys_services")


def load_deserialization_libs():
    """Import whichever of the deserialisation libraries exist; return them."""
    import importlib
    loaded = []
    for lib in _DESERIALIZE_LIBS:
        try:
            importlib.import_module("icecube." + lib)
            loaded.append(lib)
        except ImportError:
            pass
    return loaded


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

# -------------------------------------------------------------------------
# Out: booking to HDF5
# -------------------------------------------------------------------------
#
# `add_booker` is four lines, and it is a function rather than four lines in
# each caller because of what hdfwriter writes BESIDE the data.  For every key
# it also writes a `/__I3Index__/<key>` table, one row per FRAME, carrying
# `exists` and `start`/`stop`.  `data._index_node()` and `pass2.match()` both
# depend on it: it is the ONLY reliable way to line a table up with its events,
# because `(Run, Event, SubEvent)` is not unique across the L3 files a single
# `--chunk-files` part holds (booking audit, bug 2).  Whatever this function
# ever becomes, it has to keep writing that index.
#
# ON THE DEPRECATION WARNING, PRECISELY.  Importing hdfwriter under icetray
# v1.17.0 warns "icecube.hdfwriter is deprecated ... Use icecube.tableio
# instead".  That reads like "hdfwriter is going away", and it is worth being
# exact about what it means, because the two are layers rather than rivals:
# `tableio` is the generic engine (frames -> rows, through the per-type
# converters) and knows nothing about HDF5, while the HDF5 BACK END --
# `I3HDFTableService` -- lives inside hdfwriter.  The production's own booking
# code says so outright (icetray-oscNext/.../tools/processor.py):
#
#     from icecube.tableio  import I3TableWriter
#     from icecube.hdfwriter import I3HDFTableService
#
# So migrating to tableio means dropping the `I3HDFWriter` CONVENIENCE SEGMENT
# for the generic writer plus that service -- it does NOT reduce the dependency
# on hdfwriter, and it is not protection against hdfwriter being removed.  (Our
# own history confirms the segment is a wrapper: a booking run that called
# I3HDFWriter died with "I3TableWriter died mid-run", see
# verification/README.md.)  Worth doing to match the production's code, not
# as a fix for the warning; unmeasured which of the two the warning fires on.
#
#     THERE USED TO BE A FALLBACK HERE, AND IT WOULD NOT HAVE WORKED.
#     `SimpleBooker` (~250 lines) reproduced the table layout by hand through
#     pytables, for a meta-project built without HDF5 support.  hdfwriter IS in
#     the environment this pipeline runs on, so it never ran end to end and
#     every measurement in CLAUDE.md was made through hdfwriter -- and,
#     decisively, it wrote the data tables ONLY, never `/__I3Index__/`.  Files
#     booked through it would have been refused by `pass2.match()` and silently
#     triple-matched by `load_sample`.  It was not a safety net; it only looked
#     like one.

def add_booker(tray, name, output, keys, sub_event_streams=("InIceSplit",)):
    '''Add the HDF5 writer to the tray.'''
    from icecube import hdfwriter
    tray.Add(hdfwriter.I3HDFWriter, name,
             Output=output, Keys=keys,
             SubEventStreams=list(sub_event_streams))
