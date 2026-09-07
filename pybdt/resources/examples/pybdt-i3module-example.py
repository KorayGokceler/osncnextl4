#!/usr/bin/env python3


__doc__ = """Generate I3Frames with the sample BDT score included."""

import os
from icecube.icetray import I3Tray
from icecube import dataio, dataclasses, icetray, phys_services

import datetime, time
from optparse import OptionParser

import numpy as np

from icecube.pybdtmodule import PyBDTModule
from pybdt.util import load

start_time = time.time ()

parser = OptionParser (usage='%prog {[options]}')
parser.add_option ('-n', '--n-events', dest='n_events', type=int,
        default=50, metavar='N',
        help='generate and score N sample events')
parser.add_option ('-c', '--cut', dest='cut', type=float,
        default=-1.0, metavar='SCORE',
        help='cut on BDT score > SCORE')
opts, args = parser.parse_args ()

I3_BUILD = os.getenv ('I3_BUILD')
output_dir = '{0}/pybdt/resources/examples/output'.format(I3_BUILD)

print ('Initializing tray...')
tray = I3Tray ()

tray.AddModule ('I3InfiniteSource', 'source',
        stream=icetray.I3Frame.Physics)

def PopulateEvent (frame):
    """Generate either a signal like or background like event."""

    # generate a "high" "signal" rate so it's easy to find sample signal events
    # in output file
    sig_rate = .3
    bg_rate = 1.
    total_rate = sig_rate + bg_rate
    frac_sig = sig_rate / total_rate

    signal_like = np.random.random () < frac_sig
    if signal_like:
        a = 2 + 1 * np.random.randn ()
        b = .8 * 1.2 * np.random.randn ()
        c = 120 + 20 * np.random.randn ()
        frame.Put ('IsSignal', icetray.I3Bool (True))
    else:
        a = 2 * np.random.randn ()
        b = 3 + .6 * np.random.randn ()
        c = 70 + 30 * np.random.randn ()
        frame.Put ('IsBackground', icetray.I3Bool (True))
    frame.Put ('A', dataclasses.I3Double (a))
    frame.Put ('B', dataclasses.I3Double (b))
    frame.Put ('C', dataclasses.I3Double (c))
    frame.Put ('IsSignal_Truth', icetray.I3Bool (signal_like))
    frame.Put ('I3EventHeader', dataclasses.I3EventHeader ())

# create fake events just like generate_sample_data.py
tray.AddModule (PopulateEvent, 'populate_events')

def varsfunc (frame):
    a = frame['A'].value
    b = frame['B'].value
    c = frame['C'].value
    out = dict (a=a, b=b, c=c)
    return out

# score the fake events
tray.AddModule (PyBDTModule, 'bdt',
        BDTFilename='{0}/sample.bdt'.format(output_dir),
        varsfunc=varsfunc,
        OutputName='Score')

# apply a loose cut
def cutfunc (frame):
    if 'Score' in frame and frame['Score'].value > opts.cut:
        return True
    else:
        return False

tray.AddModule (cutfunc, 'cutter')

tray.AddModule ('I3Writer', 'writer',
                filename='{0}/sample.i3'.format(output_dir))



print ('Executing tray...')
tray.Execute (opts.n_events)


end_time = time.time ()

elapsed_time = end_time - start_time

print ('Processing complete ({0:.0f} s).'.format (elapsed_time))
