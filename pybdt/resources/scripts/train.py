#!/usr/bin/env python3
# train.py


from datetime import timedelta
from optparse import OptionGroup, OptionParser
import time

from pybdt import ml
from pybdt import util

start_time = time.time ()

usage = \
"""%prog {options} [comma-sep\'d variables] \\
         [training signal] [training background] [bdt filename]"""
parser = OptionParser (usage=usage)

data_group = OptionGroup (parser, 'Data options')
dt_group = OptionGroup (parser, 'Decision Tree options')
forest_group = OptionGroup (parser, 'Forest options')

dt_group.add_option ('-d', '--depth', dest='depth',
        default=None, type=int, metavar='N',
        help='make trees N levels deep')

dt_group.add_option ('-L', '--nonlinear-cuts', dest='linear_cuts',
        default=True, action='store_false',
        help='use nonlinear cut spacing')

dt_group.add_option ('-c', '--num-cuts', dest='num_cuts',
        default=None, type=int, metavar='N',
        help='try N cuts per var per node')

dt_group.add_option ('-s', '--min-split', dest='min_split',
        default=None, type=int, metavar='N',
        help='do not split if a node contains fewer than N events')

dt_group.add_option ('-p', '--prune-strength', dest='prune_strength',
        default=None, type=float, metavar='STRENGTH',
        help='use STRENGTH prune strength')

dt_group.add_option ('-v', '--num-random-variables', dest='num_random_variables',
        default=0, type=int, metavar='N',
        help='use N randomly selected variables at each node '
        '(0 to use every var at every node)')

forest_group.add_option ('-t', '--num-trees', dest='num_trees',
        default=None, type=int, metavar='N',
        help='use N trees')

forest_group.add_option ('-b', '--beta', dest='beta',
        default=None, type=float, metavar='BETA',
        help='use BETA boost parameter')

forest_group.add_option ('-e', '--frac-random-events', dest='frac_random_events',
        default=1, type=float, metavar='FRAC',
        help='use FRAC randomly selected fraction of events in each tree '
        '(1, the default, to use every event in every tree)')

forest_group.add_option ('-r', '--use-purity', dest='use_purity',
        default=None, action='store_true',
        help='use purity information during training as in SAMME.R')

data_group.add_option ('--sig-weight', dest='sig_weight',
        default='', metavar='COLNAME',
        help='the name of the variable in which the signal weights are stored')

data_group.add_option ('--bg-weight', dest='bg_weight',
        default='', metavar='COLNAME',
        help='the name of the variable in which the bg weights are stored')

parser.add_option_group (dt_group)
parser.add_option_group (forest_group)
parser.add_option_group (data_group)

opts, args = parser.parse_args ()

feature_list = args[0]
features = feature_list.split (',')
train_sig_filename = args[1]
train_bg_filename = args[2]
bdt_filename = args[3]

print ('Configuring BDTLearner...')

learner = ml.BDTLearner (features, opts.sig_weight, opts.bg_weight)

learner.dtlearner.linear_cuts = opts.linear_cuts
if opts.num_cuts is not None:
    learner.dtlearner.num_cuts = opts.num_cuts
if opts.min_split is not None:
    learner.dtlearner.min_split = opts.min_split
if opts.depth is not None:
    learner.dtlearner.max_depth = opts.depth
if opts.num_random_variables != 0:
    learner.dtlearner.num_random_variables = opts.num_random_variables

if opts.num_trees is not None:
    learner.num_trees = opts.num_trees
if opts.beta is not None:
    learner.beta = opts.beta
if opts.frac_random_events != 1:
    learner.frac_random_events = opts.frac_random_events
learner.add_before_pruner (
        ml.SameLeafPruner ())
if opts.prune_strength is not None:
    learner.add_before_pruner (
            ml.CostComplexityPruner (opts.prune_strength))
if opts.use_purity is not None:
    learner.use_purity = True

print ('Loading signal training events...')
train_sig = util.load (train_sig_filename)
print ('Loading background training events...')
train_bg = util.load (train_bg_filename)

print ('Training BDT...')
bdt = learner.train (train_sig, train_bg)

print ('Saving BDT as {0} ...'.format (bdt_filename))
util.save (bdt, bdt_filename)

end_time = time.time ()

duration = timedelta (seconds=end_time - start_time)

print ('Finished in {0}.'.format (duration))
