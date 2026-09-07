#!/usr/bin/env python3
# setup_sample_validator.py

import sys, os

__doc__ = """Setup a pybdt.validate.Validator for the sample BDT."""

try:
    import matplotlib
except ImportError:
    print("Matplotlib is missing, so setup_sample_validator.py cannot run.")
    sys.exit()

matplotlib.use ('Agg')

from pybdt.util import save
from pybdt.validate import Validator

I3_BUILD = os.getenv ('I3_BUILD')
data_dir = '{0}/pybdt/resources/examples/data'.format(I3_BUILD)
output_dir = '{0}/pybdt/resources/examples/output'.format(I3_BUILD)

print ('Creating Validator...')
v = Validator ('{0}/sample.bdt'.format(output_dir))

def add_data (name, label):
    filename = '{0}/{1}.ds'.format (data_dir, name)
    print ('Adding {0} from {1} ...'.format (label, filename))
    v.add_data (name, filename, label=label)

add_data ('train_sig', 'Training signal sim')
add_data ('train_data', 'Training data')
add_data ('test_sig', 'Testing signal sim')
add_data ('test_data', 'Testing data')
add_data ('bg', 'Background sim')

print ('Configuring weighting...')

v.add_weighting ('weight', 'train_sig',
        color='cyan',
        )
v.add_weighting ('livetime', 'train_data',
        line=False, markers=True, marker='.', color='.5',
        errorbars=True,
        )

v.add_weighting ('weight', 'test_sig',
        color='blue',
        add_to_mc=True,
        )
v.add_weighting ('livetime', 'test_data',
        line=False, markers=True, marker='.', color='black',
        errorbars=True,
        use_as_data=True,
        )

v.add_weighting ('livetime', 'bg',
        color='purple',
        add_to_mc=True,
        )

v.setup_total_mc (
        color='green',
        )

save (v, '{0}/sample.validator'.format(output_dir))
