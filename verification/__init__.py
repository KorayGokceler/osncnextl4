"""
The pass2 cross-check -- the answer key this repository was verified against.

It is NOT part of the pipeline.  The pipeline turns L3 files into HDF5 and
trains two classifiers; this arm runs that same pipeline over the pass2 L3
files and holds the result against the real pass2 L4 files, which already
contain every variable:

    pass2 L3  --(our oscNext_L4 segment)-->  ours.hdf5   ┐
                                                         ├-> compare, column
    pass2 L4  --(book what is already there)-> pass2.hdf5 ┘   by column

That is what established the current status: 14 of 15 rows bitwise identical
over 56,301 events in five samples.  README.md here is the step-by-step record.

WHY IT SITS OUTSIDE `oscnext_l4/`: it is a CONSUMER of the package, like
`scripts/` and `notebooks/` are, not a part of it.  Nothing in the pipeline
imports it.  The dependency runs one way -- `pass2.py` reads
`oscnext_l4.data`'s HDF5 layout helpers (`_table_nodes`, `_index_node`,
`_ids`), deliberately, because that is where the knowledge of what hdfwriter
writes lives and the cross-check must read files exactly as loading does.
They are private names and this is an insider; if a second outside caller ever
wants them, promote them rather than widening this exception.

**Delete this arm LAST.**  It is the safety net: a simplification that
silently breaks a variable is caught here and nowhere else.
"""
