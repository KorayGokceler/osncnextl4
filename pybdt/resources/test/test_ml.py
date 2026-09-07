#!/usr/bin/env python3

# some of these tests are only concerned with object creation hence:
#   ruff: noqa: F841

import os
import tempfile
import unittest

import numpy as np

from pybdt import ml, util, viz

#np.random.seed (0)

class TestWithSimpleData (unittest.TestCase):

    def setUp (self):
        self.N = 1000
        bg = dict (
            a = np.linspace (0, .8, self.N),
            b = np.arange (0, self.N),
            c = np.logspace (3, 5, self.N),
            d = np.random.permutation (self.N) - 3 * self.N / 4.,
            e = np.linspace (-100, 100, self.N),
            f = np.linspace (-10, 5, self.N),
            g = np.linspace (10, 40, self.N),
        )
        bg['w'] = np.exp (-(bg['e'] + 10.) **2 / (2 * 20**2))
        sig = dict (
            a = np.linspace (1.2, 2, self.N),
            b = np.arange (self.N, 2 * self.N),
            c = np.logspace (0, 2, self.N),
            d = np.random.permutation (self.N) - self.N / 4.,
            e = np.linspace (-100, 100, self.N),
            f = np.linspace (-10, 5, self.N),
            g = np.linspace (10, 40, self.N),
        )
        sig['w'] = np.exp (-(sig['e'] - 10.) **2 / (2 * 20**2))
        self.bg = ml.DataSet (bg)
        self.sig = ml.DataSet (sig)
        self.n_features = len (bg)
        self.feature_names = sorted (sig)

    def tearDown (self):
        self.bg = self.sig = None


class TestDatSet (TestWithSimpleData):

    def test_features (self):
        self.assertEqual (self.bg.n_features, self.n_features)
        self.assertEqual (self.sig.n_features, self.n_features)
        self.assertEqual (self.bg.n_events, self.N)
        self.assertEqual (self.sig.n_events, self.N)
        self.assertEqual (self.bg.names, self.feature_names)
        self.assertEqual (self.sig.names, self.feature_names)
        self.assertEqual (len (self.bg), self.N)
        self.assertEqual (len (self.sig), self.N)
        self.assertEqual (self.bg['a'][0], 0)
        self.assertEqual (self.bg['b'][0], 0)
        self.assertEqual (self.sig['a'][-1], 2)
        self.assertEqual (self.sig['b'][0], self.N)

    def test_get_subset (self):
        mask = np.arange (len (self.bg)) % 2 == 0
        sub_bg = self.bg.get_subset (mask)
        self.assertEqual (sub_bg['b'][0], 0)
        self.assertEqual (sub_bg['b'][-1], self.N - 2)

    def test_to_dict (self):
        d_bg = self.bg.to_dict ()
        self.assertEqual (sorted (d_bg), self.feature_names)
        self.assertTrue (np.all (d_bg['a'] == self.bg['a']))

    def test_etc (self):
        bg = self.bg.to_dict ()
        bg['a'] = np.r_[bg['a'], 0]
        with self.assertRaises (RuntimeError):
            ml.DataSet (bg)
        bg['a'] = self.bg['a']
        bg_even = ml.DataSet (bg, subset='even')
        bg_odd = ml.DataSet (bg, subset='odd')
        self.assertEqual(bg_even['a'][0], self.bg['a'][0])
        self.assertEqual(bg_odd['a'][0], self.bg['a'][1])
        yr = 365.25 * 86400
        bg_even.livetime = yr
        self.assertTrue (bg_even.livetime, yr)
        with self.assertRaises (RuntimeError):
            bg_even['nope']
        bg['x'] = 'who_cares'
        ml.DataSet (bg)


class TestBDT_Simple (TestWithSimpleData):

    def test_learner_structors (self):
        fs = ['a', 'b', 'c']
        bdtl = ml.BDTLearner (fs)
        bdtl = ml.BDTLearner (fs, 'w')
        bdtl = ml.BDTLearner (fs, 'w', 'w')
        dtl = ml.DTLearner (fs)
        dtl = ml.DTLearner (fs, 'w')
        dtl = ml.DTLearner (fs, 'w', 'w')

    def test_onecut (self):
        bdtl = ml.BDTLearner (['a'])
        bdtl.num_trees = 1
        self.assertEqual (bdtl.num_trees, 1)
        bdtl.beta = 1
        self.assertEqual (bdtl.beta, 1)
        bdtl.frac_random_events = 1
        self.assertEqual (bdtl.frac_random_events, 1)
        bdtl.quiet = False # just once for coverage
        self.assertFalse (bdtl.quiet)
        bdtl.use_purity = False
        self.assertFalse (bdtl.use_purity)
        dtl = bdtl.dtlearner
        dtl.max_depth = 1
        self.assertEqual (dtl.max_depth, 1)
        dtl.num_cuts = 100
        self.assertEqual (dtl.num_cuts, 100)
        dtl.linear_cuts = True
        self.assertTrue (dtl.linear_cuts)
        dtl.num_random_variables = 0
        self.assertEqual (dtl.num_random_variables, 0)
        bdt = bdtl.train (self.sig, self.bg)
        self.assertEqual (len (bdt.dtmodels), 1)
        dt = bdt.dtmodels[0]
        cut = dt.root
        self.assertEqual (cut.feature_id, 0)
        self.assertEqual (cut.feature_name, 'a')
        self.assertEqual (cut.max_depth, 1)
        self.assertEqual (cut.tree_size, 3)
        self.assertGreater(cut.feature_val, .8)
        self.assertLess(cut.feature_val, 1.2)
        self.assertLess(cut.left.purity, .01)
        self.assertGreater(cut.right.purity, .99)
        self.assertEqual (cut.n_total, len (self.sig) + len (self.bg))
        self.assertEqual (cut.n_sig, len (self.sig))
        self.assertEqual (cut.n_bg, len (self.bg))
        self.assertTrue (1.999 < cut.w_total < 2.001)
        self.assertTrue (.999 < cut.w_sig < 1.001)
        self.assertTrue (.999 < cut.w_bg < 1.001)
        self.assertEqual (cut.left.feature_name, '[leaf]')

    def test_onecut_nonlinear (self):
        bdtl = ml.BDTLearner (['c'])
        bdtl.num_trees = 1
        bdtl.quiet = True
        dtl = bdtl.dtlearner
        dtl.max_depth = 1
        dtl.num_cuts = self.N
        dtl.linear_cuts = False
        bdt = bdtl.train (self.sig, self.bg)
        cut = bdt.dtmodels[0].root
        self.assertLessEqual(cut.feature_val, self.bg['c'].min())
        self.assertGreaterEqual(cut.feature_val, self.sig['c'].max())

    def test_onecut_weights (self):
        bdtl = ml.BDTLearner (['a'], 'w', 'w')
        bdtl.num_trees = 1
        bdtl.dtlearner.max_depth = 1
        bdtl.dtlearner.num_cuts = 100
        bdtl.quiet = True
        bdt = bdtl.train (self.sig, self.bg)
        dt = bdt.dtmodels[0]
        cut = dt.root
        self.assertGreater(cut.feature_val, .8)
        self.assertLess(cut.feature_val, 1.2)

    def test_onecut_rand (self):
        bdtl = ml.BDTLearner (['a', 'b', 'c'])
        bdtl.num_trees = 1
        bdtl.frac_random_events = .5
        bdtl.quiet = True
        dtl = bdtl.dtlearner
        dtl.num_cuts = 100
        dtl.max_depth = 1
        dtl.num_random_variables = 2
        bdt = bdtl.train (self.sig, self.bg)
        dt = bdt.dtmodels[0]
        left = dt.root.left
        right = dt.root.right
        p = left.purity
        self.assertGreater(p + right.purity, .9999)
        self.assertGreater(max (p, 1 - p), .9999)

    def test_onecut_sep (self):
        bdtl = ml.BDTLearner (['a'])
        bdtl.num_trees = 1
        bdtl.quiet = True
        for sep in 'gini', 'cross_entropy', 'misclass_error':
            dtl = bdtl.dtlearner
            dtl.max_depth = 1
            # fails for cross_entropy for too many cuts
            # because of behavior as min(p,1-p)->0
            dtl.num_cuts = 100
            dtl.separation_type = sep
            self.assertEqual (dtl.separation_type, sep)
            bdt = bdtl.train (self.sig, self.bg)
            cut = bdt.dtmodels[0].root
            self.assertGreater(
                cut.feature_val,
                .8,
                (cut.feature_val, cut.sep_index)
            )
            self.assertLess(cut.feature_val, 1.2, cut.feature_val)

    def test_threecut_overlap (self):
        bdtl = ml.BDTLearner (['d'])
        bdtl.num_trees = 1
        bdtl.quiet = True
        dtl = bdtl.dtlearner
        dtl.max_depth = 2
        dtl.num_cuts = 101
        bdt = bdtl.train (self.sig, self.bg)
        dt = bdt.dtmodels[0]
        cut = dt.root
        if cut.left.is_leaf:
            p1 = cut.left.purity
            p2a = cut.right.left.purity
            p2b = cut.right.right.purity
        else:
            p1 = cut.right.purity
            p2a = cut.left.left.purity
            p2b = cut.left.right.purity
        # d: bg on [-75, 24], bg on [-25, 74]
        # first cut chops off one overlap-free zone
        # second cut chops off the other overlap-free zone
        # leaves contain sig-pure, bg-pure, and 50/50 mix
        self.assertIn(p1, (0, 1))
        self.assertIn(1 - p1, (p2a, p2b))
        self.assertIn(.5, (p2a, p2b))

    def test_threecut_overlap_b (self):
        bdtl = ml.BDTLearner (['e'], 'w', 'w')
        bdtl.num_trees = 1
        bdtl.quiet = True
        dtl = bdtl.dtlearner
        dtl.max_depth = 3
        dtl.num_cuts = 101
        bdt = bdtl.train (self.sig, self.bg)
        dt = bdt.dtmodels[0]
        self.assertTrue(np.isclose(dt.root.feature_val, 0, atol=1e-14),
                        f"feature_val: {dt.root.feature_val}, not close enough to zero")

    def test_prune (self):
        # mostly check that it query works and pruning does *anything*
        bdtl = ml.BDTLearner (list ('efg'), 'w', 'w')
        bdtl.num_trees = 1
        bdtl.quiet = True
        dtl = bdtl.dtlearner
        dtl.max_depth = 10
        dtl.num_cuts = 40
        bdt1 = bdtl.train (self.sig, self.bg)
        depth1 = bdt1.dtmodels[0].root.max_depth
        bdtl.add_before_pruner (ml.SameLeafPruner ())
        bdt2 = bdtl.train (self.sig, self.bg)
        depth2 = bdt2.dtmodels[0].root.max_depth
        self.assertLess(depth2, depth1)
        bdtl.clear_before_pruners ()
        p = ml.CostComplexityPruner (30)
        p.strength = 20
        self.assertEqual (p.strength, 20)
        bdtl.add_after_pruner (p)
        bdt3 = bdtl.train (self.sig, self.bg)
        depth3 = bdt3.dtmodels[0].root.max_depth
        self.assertLess(depth3, depth1)
        bdtl.clear_after_pruners ()

        self.assertFalse (list(bdtl.before_pruners))
        self.assertFalse (list(bdtl.after_pruners))

        bdtl.clear_before_pruners ()
        bdtl.clear_after_pruners ()
        p = ml.ErrorPruner (1e-6)
        p.strength = 1e-7
        self.assertEqual (p.strength, 1e-7)
        bdtl.add_after_pruner (p)
        bdt4 = bdtl.train (self.sig, self.bg)
        depth4 = bdt4.dtmodels[0].root.max_depth
        self.assertLess(depth4, depth1)

    @unittest.skipIf("ICETRAY_RUNNER_OS" in os.environ and \
                     "alma8" in os.environ.get("ICETRAY_RUNNER_OS"),
                     "Skipping test on dist: alma8, assertion fails (float compare) in test_boost")
    def test_boost (self):
        names = list ('efg')
        bdtl = ml.BDTLearner (names, 'w', 'w')
        bdtl.num_trees = 30
        bdtl.quiet = True
        dtl = bdtl.dtlearner
        dtl.max_depth = 2
        dtl.num_cuts = 40
        bdt = bdt1 = bdtl.train (self.sig, self.bg)
        bdtl.use_purity = True
        self.assertTrue (bdtl.use_purity)
        bdt2 = bdtl.train (self.sig, self.bg)
        # first tree should always be best
        self.assertEqual(bdt.alphas[0], bdt.alphas.max())
        # 'e' should be most important
        imp1TT = bdt1.variable_importance (True, True)
        self.assertGreater(imp1TT['e'], imp1TT['f'])
        self.assertGreater(imp1TT['e'], imp1TT['g'])
        imp1T = bdt1.dtmodels[0].variable_importance (True)
        self.assertGreater(imp1T['e'], imp1T['f'])
        self.assertGreater(imp1T['e'], imp1T['g'])
        imp1T = bdt1.dtmodels[0].variable_importance (False)
        # get sig scores
        scores1 = bdt1.score (self.sig, quiet=True)
        scores2 = bdt2.score (self.sig, quiet=True)
        # try the most sig-like event
        event_vals = [self.sig[k][-1] for k in names]
        event = dict (zip (names, event_vals))
        # what about a trimmed BDT?
        bdt3 = bdt.get_trimmed_bdtmodel (.1)
        self.assertLess(len (bdt3), len (bdt))
        self.assertEqual (bdt3.score (event), 1)
        bdt4 = bdt.get_subset_bdtmodel (0, 5)
        self.assertEqual (len (bdt4), 5)
        self.assertTrue (bdt4.score (event), 1)
        bdt5 = bdt.get_subset_bdtmodel_list (np.arange (len (bdt)) < 5)
        self.assertEqual (len (bdt5), 5)
        self.assertEqual (bdt4.score (event), 1)
        with self.assertRaises (RuntimeError):
            bdt.get_subset_bdtmodel (0, 500)
        with self.assertRaises (RuntimeError):
            bdt.get_subset_bdtmodel_list ([0, 500])

        # more about the most sig-like event
        event_vals = [self.sig[k][-1] for k in names]
        event = dict (zip (names, event_vals))
        score1 = bdt1.score (event)
        pscore1 = bdt1.score (event, use_purity=True)
        score2 = bdt2.score (event)
        pscore2 = bdt2.score (event, use_purity=True)
        self.assertNotEqual (score1, pscore1)
        self.assertNotEqual (score2, pscore2)
        # FIXME: fails on some machines, cast/decay error?
        # self.assertNotEqual (score1, score2)
        self.assertNotEqual (pscore1, pscore2)
        self.assertEqual (score1, scores1.max())
        self.assertEqual (score2, scores2.max())
        event_imp1TT = bdt1.event_variable_importance (event, True, True)
        self.assertGreater(event_imp1TT['e'], event_imp1TT['f'])
        self.assertGreater(event_imp1TT['e'], event_imp1TT['g'])
        event_imp1T = bdt1.dtmodels[0].event_variable_importance (
            event, True)
        self.assertGreater(event_imp1T['e'], event_imp1T['f'])
        self.assertGreater(event_imp1T['e'], event_imp1T['g'])
        # try the most bg-like event
        event_vals = [self.bg[k][0] for k in names]
        event = dict (zip (names, event_vals))
        score1 = bdt1.score (event)
        pscore1 = bdt1.score (event, use_purity=True)
        score2 = bdt2.score (event)
        pscore2 = bdt2.score (event, use_purity=True)
        self.assertEqual (bdt1.dtmodels[0].score (event), -1)

    def test_pickle (self):
        bdtl = ml.BDTLearner (list ('efg'), 'w', 'w')
        bdtl.num_trees = 30
        bdtl.quiet = True
        dtl = bdtl.dtlearner
        dtl.max_depth = 4
        dtl.num_cuts = 40
        bdt1 = bdtl.train (self.sig, self.bg)
        handle, filename = tempfile.mkstemp ('.bdt', 'pybdt_test_')
        try:
            util.save (bdt1, filename)
            bdt2 = util.load (filename)
            self.assertEqual (viz.dtmodel_to_text (bdt1.dtmodels[5]),
                            viz.dtmodel_to_text (bdt2.dtmodels[5]))
        finally:
            if os.path.isfile (filename):
                os.unlink (filename)


class TestVineSimple (TestWithSimpleData):

    def test_vine (self):
        names = list ('efg')
        bdtl = ml.BDTLearner (names, 'w', 'w')
        bdtl.num_trees = 1
        bdtl.quiet = True
        dtl = bdtl.dtlearner
        dtl.max_depth = 2
        dtl.num_cuts = 40

        mi = min (self.sig['e'].min(), self.bg['e'].min())
        ma = max (self.sig['e'].max(), self.bg['e'].max())
        width = (ma - mi) / 5
        step = width / 2

        with self.assertRaises (ValueError):
            vl = ml.VineLearner ('a', mi, ma, width, step, bdtl)

        vl = ml.VineLearner ('e', mi, ma, width, step, bdtl)
        vl.quiet = True

        self.assertEqual (vl.vine_feature, 'e')
        self.assertEqual (vl.vine_feature_min, mi)
        self.assertEqual (vl.vine_feature_max, ma)
        self.assertEqual (vl.vine_feature_width, width)
        self.assertEqual (vl.vine_feature_step, step)
        self.assertEqual (vl.quiet, True)
        self.assertEqual (vl.learner._, bdtl._)

        vm = vl.train (self.sig, self.bg)

        vl.quiet = False
        vl.vine_feature = 'f'
        mi = min (self.sig['f'].min(), self.bg['f'].min())
        ma = max (self.sig['f'].max(), self.bg['f'].max())
        vl.vine_feature_min = mi
        vl.vine_feature_max = ma
        vl.vine_feature_width = (ma - mi) / 3
        vl.vine_feature_step = vl.vine_feature_width / 3

        vm2 = vl.train (self.sig, self.bg)

    def test_pickle (self):
        names = list ('efg')
        bdtl = ml.BDTLearner (names, 'w', 'w')
        bdtl.num_trees = 1
        bdtl.quiet = True
        dtl = bdtl.dtlearner
        dtl.max_depth = 2
        dtl.num_cuts = 40

        mi = min (self.sig['e'].min(), self.bg['e'].min())
        ma = max (self.sig['e'].max(), self.bg['e'].max())
        width = (ma - mi) / 5
        step = width / 2

        vl = ml.VineLearner ('e', mi, ma, width, step, bdtl)
        vl.quiet = True

        vm1 = vl.train (self.sig, self.bg)
        handle, filename = tempfile.mkstemp ('.vine', 'pybdt_test_')
        try:
            util.save (vm1, filename)
            vm2 = util.load (filename)
            dscore = np.abs (vm1.score (self.sig) - vm2.score (self.sig))
            self.assertEqual (np.sum (dscore < 1e-5), len (self.sig))
        finally:
            if os.path.isfile (filename):
                os.unlink (filename)


if __name__ == '__main__':
    unittest.main ()
