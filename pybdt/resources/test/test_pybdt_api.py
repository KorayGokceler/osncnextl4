#!/usr/bin/env python3
# pybdt_api_test.py


__doc__ = """Try to call as much of the PyBDT API as possible."""


import numpy as np
import os
from pybdt import ml
from pybdt import util
from pybdt import viz


# Load data from example scripts.
build_dir = os.getenv ('I3_BUILD')
data_dir = '{0}/pybdt/resources/examples/data'.format (build_dir)
train_sig = util.load ('{0}/train_sig.ds'.format (data_dir))
test_sig = util.load ('{0}/test_sig.ds'.format (data_dir))
train_bg = util.load ('{0}/train_data.ds'.format (data_dir))
test_bg = util.load ('{0}/test_data.ds'.format (data_dir))

def get_smaller_dataset (ds):
    return ds.get_subset (np.arange (len (ds)) % 10 == 0)

train_sig = get_smaller_dataset (train_sig)
test_sig = get_smaller_dataset (test_sig)
train_bg = get_smaller_dataset (train_bg)
test_bg = get_smaller_dataset (test_bg)

features = ['a', 'b', 'c']

def print_dtlearner (dtl):
    print ('* max_depth = {0}'.format (dtl.max_depth))
    print ('* min_split = {0}'.format (dtl.min_split))
    print ('* num_cuts = {0}'.format (dtl.num_cuts))
    print ('* num_random_variables = {0}'.format (dtl.num_random_variables))
    print ('* separation_type = {0}'.format (dtl.separation_type))

# Test a DTLearner
dtl = ml.DTLearner (features, 'weight', '')
print ('DTLearner defaults:')
print_dtlearner (dtl)
dtl.set_defaults ()
dtl.max_depth = 3
dtl.min_split = dtl.min_split // 2
dtl.num_cuts = 1000

print ('Training DTModels with each separation type...')
dtl.separation_type = 'cross_entropy'
dtm_cross_entropy = dtl.train (train_sig, train_bg)
dtl.separation_type = 'gini'
dtm_gini = dtl.train (train_sig, train_bg)
dtl.separation_type = 'misclass_error'
dtm_misclass_error = dtl.train (train_sig, train_bg)

# Test the DTModel some more
print ('* gini tree separation-weighted variable importance:')
print (viz.variable_importance_to_text (
    dtm_gini.variable_importance (True)))
print ('* gini tree number-of-use-weighted variable importance:')
print (viz.variable_importance_to_text (
    dtm_gini.variable_importance (False)))
print ('* feature_names = {0}'.format (', '.join (dtm_gini.feature_names)))
print ()

# Test scoring
print ('Testing scoring...')
dtm_gini.score (dict (a=.1, b=.2, c=.3))
dtm_gini.score (dict (a=test_sig['a'], b=test_sig['b'], c=test_sig['c']))
dtm_gini.score (test_sig)
print ()

# Test a DTNode in ways viz does not bother with
root = dtm_gini.root
print ('* feature_name = {0}'.format (root.feature_name))
print ('* max_depth = {0}'.format (root.max_depth))
print ('* n_sig = {0}'.format (root.n_sig))
print ('* n_bg = {0}'.format (root.n_bg))
print ('* n_total = {0}'.format (root.n_total))
print ('* w_total = {0:.3e}'.format (root.w_total))
print ('* w_sig = {0:.3e}'.format (root.w_sig))
print ('* w_bg = {0:.3e}'.format (root.w_bg))
print ('* n_leaves = {0}'.format (root.n_leaves))
print ('* sep_gain = {0:.3e}'.format (root.sep_gain))
print ('* sep_index = {0:.3e}'.format (root.sep_index))
print ('* tree_size = {0}'.format (root.tree_size))


# Test a BDTLearner
bdtl = ml.BDTLearner (features, 'weight', '')
print ('BDTLearner.dtlearner defaults:')
print_dtlearner (bdtl.dtlearner)
print ('BDTLearner other defaults:')
print ('* beta = {0}'.format (bdtl.beta))
print ('* num_trees = {0}'.format (bdtl.num_trees))
print ('* frac_random_events = {0}'.format (bdtl.frac_random_events))
print ('* quiet = {0}'.format (bdtl.quiet))
print ('* after_pruners = {0}'.format (bdtl.after_pruners))
print ('* before_pruners = {0}'.format (bdtl.before_pruners))
print ()

print ('Training a random forest BDTModel...')
bdtl.dtlearner.num_random_variables = 2
bdtl.frac_random_events = 0.5
bdtm_rf = bdtl.train (train_sig, train_bg)

print ('Training a BDTModel with no pruning...')
bdtl.set_defaults ()
bdtm_noprune = bdtl.train (train_sig, train_bg)

print ('Training a BDTModel with cost complexity pruning...')
bdtl.set_defaults ()
bdtl.add_before_pruner (ml.CostComplexityPruner (20))
bdtm_ccprune = bdtl.train (train_sig, train_bg)

print ('Training a BDTModel with cost complexity "after" pruning...')
bdtl.set_defaults ()
bdtl.add_after_pruner (ml.CostComplexityPruner (20))
bdtm_ccaprune = bdtl.train (train_sig, train_bg)

print ('Training a BDTModel with expected error pruning...')
bdtl.set_defaults ()
bdtl.add_before_pruner (ml.ErrorPruner (5))
bdtm_eeprune = bdtl.train (train_sig, train_bg)

print ('Training a BDTModel with expected error "after" pruning...')
bdtl.set_defaults ()
errorpruner = ml.ErrorPruner (5)
bdtl.add_after_pruner (errorpruner)
bdtm_eeaprune = bdtl.train (train_sig, train_bg)
bdtl.clear_after_pruners ()
bdtl.clear_before_pruners ()
print ()

print ('Checking Pruner methods on root of last tree:')
root = bdtm_eeaprune.dtmodels[0].root
print ('* sep gain = {0:.3e} and rho = {1:.3e}'.format (
    ml.CostComplexityPruner.gain (root),
    ml.CostComplexityPruner.rho (root)))
print ('* error = {0:.3e} and subtree_error = {1:.3e}'.format (
    errorpruner.node_error (root),
    errorpruner.subtree_error (root)))
print ()

# Test viz, and show that trees are different from each other
print ('Here is the first tree from the no-pruning BDTModel:')
print (viz.dtmodel_to_text (bdtm_noprune.dtmodels[0], 2, 4))
print ()
print ('Here is the second tree from the cost-complexity pruning BDTModel:')
print (viz.dtmodel_to_text (bdtm_ccprune.dtmodels[1], 2, 4))
print ()
print ('Here is the second tree from the error pruning BDTModel:')
print (viz.dtmodel_to_text (bdtm_eeprune.dtmodels[1], 2, 4))
print ()

# TODO:
print ('More no-pruning BDTModel stuff:')
print ('* sum(alphas) = {0:.3f}'.format (bdtm_noprune.alphas.sum ()))
print ('* n_dtmodels = {0}'.format (bdtm_noprune.n_dtmodels))
print ('* tree-weighted, separation-weighted variable importance:')
print (viz.variable_importance_to_text (
    bdtm_noprune.variable_importance (True, True)))
print ('* tree-unweighted, separation-weighted variable importance:')
print (viz.variable_importance_to_text (
    bdtm_noprune.variable_importance (True, False)))
print ('* tree-weighted, number-of-use-weighted variable importance:')
print (viz.variable_importance_to_text (
    bdtm_noprune.variable_importance (False, True)))
print ('* tree-unweighted, number-of-use-weighted variable importance:')
print (viz.variable_importance_to_text (
    bdtm_noprune.variable_importance (False, False)))
bdtm_subset = bdtm_noprune.get_subset_bdtmodel (5, 10)
print ('* feature_names = {0}'.format (', '.join (bdtm_subset.feature_names)))
print ()

# Test scoring
print ('Testing scoring...')
bdtm_noprune.score (test_sig)
bdtm_noprune.score (dict (a=test_sig['a'], b=test_sig['b'], c=test_sig['c']))
bdtm_noprune.score (dict (a=.1, b=.2, c=.3))
print ()

assert (np.isnan (bdtm_noprune.score (dict (a=np.nan, b=.2, c=.3))))
assert (np.isnan (bdtm_noprune.score (dict (a=np.inf, b=.2, c=.3))))
assert (np.isnan (bdtm_noprune.score (dict (a=-np.inf, b=.2, c=.3))))
