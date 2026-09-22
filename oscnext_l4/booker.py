'''
Booking the L4 variables to HDF5 -- the one place that knows how.

`add_booker` is a four-line wrapper around `hdfwriter.I3HDFWriter`, and it is a
function rather than four lines in each caller for one reason: hdfwriter is
DEPRECATED.  Importing it under icetray v1.17.0 warns

    icecube.hdfwriter is deprecated and will be removed in a future release.
    Use icecube.tableio instead.

and that is a real forward risk, not a style note.  hdfwriter writes a
`/__I3Index__/<key>` table beside each data table, holding one row per FRAME
with `exists` and `start`/`stop`.  `data._index_node()` and `pass2.match()`
both depend on it: it is the ONLY reliable way to line a table up with its
events, because `(Run, Event, SubEvent)` is not unique across the L3 files a
single `--chunk-files` part holds (booking audit, bug 2).  So a metaproject
that drops hdfwriter breaks booking AND event matching together, and the
migration to tableio has to keep the index.  When that day comes, it is this
function that changes.

    THERE USED TO BE A FALLBACK HERE, AND IT WOULD NOT HAVE WORKED.
    `SimpleBooker` (~250 lines) reproduced the table layout by hand through
    pytables, for a meta-project built without HDF5 support.  Three things
    were true of it: hdfwriter IS in the environment this pipeline runs on, so
    it never ran end to end; every measurement in CLAUDE.md was made through
    hdfwriter; and -- decisively -- it wrote the data tables ONLY, never
    `/__I3Index__/`.  Booking through it would have produced files that
    `pass2.match()` refuses to match and that `load_sample` can only fall back
    to triple-matching, which is the bug above.  It was not insurance against
    the deprecation either, for the same reason.  Removed rather than left to
    look like a safety net.
'''


def add_booker(tray, name, output, keys, sub_event_streams=("InIceSplit",)):
    '''Add the HDF5 writer to the tray.'''
    from icecube import hdfwriter
    tray.Add(hdfwriter.I3HDFWriter, name,
             Output=output, Keys=keys,
             SubEventStreams=list(sub_event_streams))
    return "hdfwriter"
