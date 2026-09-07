#!/usr/bin/env python3
# update_saved_files.py


from datetime import timedelta
from optparse import OptionParser
import time

import pybdt
from pybdt import ml
from pybdt import util

start_time = time.time ()

usage = '%prog [filename] {[filename]...}' + \
"""

This script updates pybdt files saved prior to the introduction of the "ml"
module.  Previously, the main classes were located in pybdt/wrapper.py, and
pybdt/__init__.py contained the line:

    from wrapper import *

Then users would use the classes like so:

    ds = pybdt.DataSet (...)

It was decided that this is poor form, and that a better idiom for users to
follow would be the following:

    from pybdt import ml

    ds = ml.DataSet (...)

So now there are no more package-level classes. This requires the small code
change listed above, but it also requires previously saved files to be updated.

This script tries to load each file passed to it. If it cannot do so, it
inserts ml.DataSet, ml.DTModel and ml.BDTModel into the pybdt package-level
namespace and tries again. If this works, it resaves the file; otherwise it
prints an error message."""

parser = OptionParser (usage=usage)

opts, args = parser.parse_args ()

filenames = args

print ()

for filename in filenames:
    try:
        util.load (filename)
    except:
        print ('Could not load {0} ; trying to update...'.format (filename))
        pybdt.DataSet = pybdt.ml.DataSet  # type: ignore[attr-defined]
        pybdt.DTModel = pybdt.ml.DTModel  # type: ignore[attr-defined]
        pybdt.BDTModel = pybdt.ml.BDTModel  # type: ignore[attr-defined]
        try:
            obj = util.load (filename)
            util.save (obj, filename)
            print ('File updated successfully.')
        except:
            print ('Could not update file, {}!'.format(filename))
        del pybdt.DataSet  # type: ignore[attr-defined]
        del pybdt.DTModel  # type: ignore[attr-defined]
        del pybdt.BDTModel  # type: ignore[attr-defined]
    else:
        print ('Loaded {0} successfully.'.format (filename))


end_time = time.time ()

duration = timedelta (seconds=end_time - start_time)

print ()
print ('Finished in {0}.'.format (duration))
