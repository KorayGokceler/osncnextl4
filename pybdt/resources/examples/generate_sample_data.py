#!/usr/bin/env python3
# generate_sample_data.py


__doc__ = """Generate sample data for use with the other example scripts."""

from pybdt.ml import DataSet
from pybdt.util import mkdir, save

import numpy as np
import sys
import os

sig_rate = 3e-3
bg_rate = .2

# an argument to the script will be interpreted as the random seed.
if len (sys.argv) >= 2:
    random_seed = int (sys.argv[1])
    np.random.seed (random_seed)
else:
    np.random.seed (0)


# create some example signal events
def make_sig (N=int (.2e5)):
    N = int (N)
    a = 2 + 1 * np.random.randn (N)
    b = .8 * 1.2 * np.random.randn (N)
    c = 120 + 20 * np.random.randn (N)
    weight = np.ones (N)
    weight /= weight.sum () # now normalized to 1
    weight *= sig_rate
    return dict (a=a, b=b, c=c, weight=weight)

# create some example background events
def make_bg (N=int (1e5)):
    N = int (N)
    a = 2 * np.random.randn (N)
    b = 3 + .6 * np.random.randn (N)
    c = 70 + 30 * np.random.randn (N)
    livetime = N / bg_rate
    return dict (a=a, b=b, c=c, livetime=livetime)

# create some "data" events (using some signal + some background)
def make_data (N):
    N = int (N)
    frac_sig = sig_rate / (sig_rate + bg_rate)
    frac_bg = bg_rate / (sig_rate + bg_rate)
    N_sig = int (N * frac_sig)
    N_bg = int (N * frac_bg)
    sig_component = make_sig (N_sig)
    bg_component = make_bg (N_bg)
    data = sig_component
    for k, v in bg_component.items ():
        try:
            data[k] = np.r_[data[k], v]
        except KeyError:
            pass
    data['livetime'] = N_bg / (bg_rate)
    del data['weight']
    return data

# make "simulation" event samples
train_sig_sim = make_sig ()
test_sig_sim = make_sig ()
bg_sim = make_bg ()

# make "data" event sample
train_data = make_data (.5e5)
test_data = make_data (5e5)

# get training and testing samples
train_sig = DataSet (train_sig_sim)
train_data = DataSet (train_data)
test_sig = DataSet (test_sig_sim)
test_data = DataSet (test_data)

bg = DataSet (bg_sim)

# create a data directory, if one does not already exist
I3_BUILD = os.getenv ('I3_BUILD')
data_dir = '{0}/pybdt/resources/examples/data'.format(I3_BUILD)
mkdir (data_dir)

print ('Saving data in data/ ...')

# save these files
save (train_sig, '{0}/train_sig.ds'.format (data_dir))
save (train_data, '{0}/train_data.ds'.format (data_dir))
save (test_sig, '{0}/test_sig.ds'.format (data_dir))
save (test_data, '{0}/test_data.ds'.format (data_dir))
save (bg, '{0}/bg.ds'.format (data_dir))

