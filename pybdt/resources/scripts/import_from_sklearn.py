#!/usr/bin/env python3
# import_from_sklearn.py


from optparse import OptionParser
import os

from pybdt import ml, util

from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.ensemble import AdaBoostClassifier


usage = '%prog {options} [comma-sep\'d feature names (no spaces)] ' \
    '[signal_id] [sklearn_infile] [pybdt_outfile]' \
"""

Import a two-class BDT from scikit-learn to pybdt.

The first agrument is the feature names, specified as a comma-separated list
with no spaces.  This list must match the ORDER in which the features were
originally supplied to sklearn for training.

The second argument is the id of the signal class in the trained scikit-learn
BDT.  This will be either 0 or 1.

The third argument is the filename containing the scikit-learn BDT.  Either
pickle or joblib (standalone or from sklearn.externals.joblib) will suffice.
The file should load as a mapping with either a single key-value pair, or with
a key given with --key KEY.

The fourth argument is the output pybdt filename.
"""
parser = OptionParser (usage=usage)

parser.add_option ('-k', '--key', dest='key',
                   default=None, metavar='KEY',
                   help='the name of the BDT object in the input file')

opts, args = parser.parse_args ()

# right number of args?
if not len (args) == 4:
    parser.error ('must provide exactly 3 arguments')

# get feature names
try:
    feature_names = args[0].split (',')
except:
    parser.error ('could not parse feature names')

# get signal id
signal_id = int (args[1])

# get filenames
infile, outfile = args[2:]
if not os.path.isfile (infile):
    parser.error ('sklearn_infile "{0}" is not a file'.format (infile))

# convert
bdt = util.load_sklearn (infile, feature_names, signal_id, key=opts.key)
util.save (bdt, outfile)

