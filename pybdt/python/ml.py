# ml.py

__doc__ = """Wrapper for C++ backend."""

import sys
import numpy as np

import pybdt._pybdt as _pybdt

def set_epsilon (eps):
    """Set the global epsilon value.

    :type   eps: float
    :param  eps: The value to use to trim purity from [0,1] to
        [epsilon,1-epsilon] when using purity for training or scoring
    """
    try:
        eps = float (eps)
    except:
        raise TypeError ('`eps` must be a float')
    if not 0 < eps < 1:
        return ValueError ('`eps` must be in (0,1)')
    _pybdt.set_epsilon (eps)

def get_epsilon ():
    """Get the global epsilon value.

    :return: The value used to trim purity from [0,1] to
        [epsilon,1-epsilon] when using purity for training or scoring
    """
    return _pybdt.get_epsilon ()


def _is_number (x):
    return      isinstance (x, int) \
            or  isinstance (x, float) \
            or  isinstance (x, np.number)

def wrapped (cpp_object):
    """Get a pure-Python wrapped instance.

    :type   cpp_object: Boost.Python.instance
    :param  cpp_object: A C++ pybdt class instance.

    :return: A pure-Python class instance.
    """
    import inspect
    cpp_class_name = cpp_object.__class__.__name__
    Class = globals ()[cpp_class_name]
    if cpp_object is None:
        return None
    argspec = inspect.getfullargspec (Class.__init__)
    n_args = len (argspec.args) if argspec.args is not None else 0
    n_defaults = len (argspec.defaults) if argspec.defaults is not None else 0
    n_no_default_args = n_args - n_defaults - 1
    if n_no_default_args == 1:
        n_None_args = 0
    else:
        n_None_args = n_no_default_args
    out = Class (*(n_None_args * [None]), _=cpp_object)
    return out

def unwrapped (py_object):
    """Get a pure-Python wrapped instance.

    :type   cpp_object: object
    :param  cpp_object: A pure-Python pybdt class instance.

    :return: A C++ pybdt class instance.
    """
    return py_object._


class DataSet :

    """A pybdt-friendly representation of a set of events."""

    # special

    def __init__ (self, data, subset='all', _=None):
        """
        :type   data: dict
        :param  data: The data, specified as a mapping from variable names to
            numpy arrays.  If there is a relavent livetime, it may be specified
            with a mapping item like this: ('livetime', ``T``).

        :type   subset: str
        :param  subset: Either 'all' or 'odd' or 'even'.
        """
        if _ is not None:
            self._ = _
            return
        if subset not in ('all', 'even', 'odd'):
            raise ValueError ("subset must be one of 'all' or 'odd' or 'even'")
        final_data = {}
        #for k, v in data.items ():
        for k in data:
            v = data[k]
            if isinstance (v, np.ndarray):
                try:
                    final_data[k] = np.asarray (v, dtype=float)
                except TypeError:
                    continue
            else:
                final_data[k] = v
        self._ = _pybdt.DataSet (final_data, subset)

    def __getstate__ (self):
        data = dict ((name, self[name]) for name in self.names)
        data['livetime'] = self.livetime
        return data

    def __setstate__ (self, data):
        self._ = _pybdt.DataSet (data)

    # operators

    def __getitem__ (self, name):
        """Get the column named ``name``."""
        return self._[name]

    def __len__ (self):
        """The number of events in the DataSet."""
        return len (self._)

    # properties

    @property
    def livetime (self):
        """The livetime of this DataSet (or -1 if never specified)."""
        return self._.livetime
    @livetime.setter
    def livetime (self, t):
        self._.livetime = t

    @property
    def n_events (self):
        """The number of rows in this DataSet."""
        return self._.n_events

    @property
    def n_features (self):
        """The number of columns in this DataSet."""
        return self._.n_features

    @property
    def names (self):
        """The names of the features stored by this DataSet."""
        return self._.names

    # methods

    def get_subset (self, idx):
        """Get a subset of this dataset.

        :type   idx: array of bools
        :param  idx: The subset of samples to keep.

        :return:    `pybdt.ml.Dataset`.
        """
        old_data = self.to_dict ()
        new_data = {}
        for k, v in old_data.items ():
            new_data[k] = v[idx]
        new_data['livetime'] = self.livetime
        if self.livetime > 0:
            new_data['livetime'] /= (1. * idx.sum () / len (self))
        return DataSet (new_data)


    def eval (self, expr, names={}):
        """Evaluate an expression in terms of variables in this DataSet.

        :type   expr: str
        :param  expr: The expression to evaluate.

        :type   names: dict
        :param  names: Names to be passed into eval.

        :return: numpy.ndarray --- one element per event

        When expr is evaluated, each variable stored in the dataset will be
        available.  If the dataset has a livetime set, 'livetime' will also be
        available.

        Other allowed identifiers are np (Numpy) and scipy, in addition to
        anything specified in the names parameter.
        """
        import scipy
        import numpy as np
        variables = {}
        for name in self.names:
            if expr.find (name) != -1:
                variables[name] = self[name]
        if self.livetime > 0:
            variables['livetime'] = self.livetime
        default_names = dict (np=np, scipy=scipy)
        default_names.update (names)
        result = eval (expr, variables, default_names)
        return result

    def to_dict (self):
        """Get a dictionary with all data from the DataSet."""
        return self._.to_dict ()


class Learner :

    """Train classification models."""

    def __init__ (self, _):
        self._ = _

    # methods

    def train (self, signal_dataset, background_dataset):
        """Train using the given DataSets."""
        _model = self._.train (
                unwrapped (signal_dataset),
                unwrapped (background_dataset))
        return wrapped (_model)

    def train_given_weights (self,
            signal_dataset, background_dataset,
            signal_weights, background_weights):
        """Train using the given DataSets and the given weights."""
        _model = self._.train_given_weights (
                unwrapped (signal_dataset),
                unwrapped (background_dataset),
                signal_weights, background_weights)
        return wrapped (_model)

class DTLearner (Learner):

    """Train single decision trees."""

    def __init__ (self, feature_names=[], weight_name='', bg_weight_name='',
            _=None):
        """Construct a DTLearner.

        :type   feature_names: list
        :param  feature_names: The names of the event features to use.

        :type   weight_name: str
        :param  weight_name: The name of the :class:`DataSet` column
            corresponding to the event weights ('' for no weighting).

        :type   bg_weight_name: str
        :param  bg_weight_name: The name of the :class:`DataSet` column
            corresponding to the background event weights ('' for no
            weighting).
        """
        if _ is not None:
            self._ = _
            return
        else:
            if not feature_names:
                raise ValueError ('must provide feature names')
        self._ = _pybdt.DTLearner (feature_names, weight_name, bg_weight_name)

    # properties

    @property
    def linear_cuts (self):
        """Space cuts linearly (default: True)."""
        return self._.linear_cuts
    @linear_cuts.setter
    def linear_cuts (self, linear_cuts):
        self._.linear_cuts = linear_cuts

    @property
    def max_depth (self):
        """The maximum depth to which to train each individual tree."""
        return self._.max_depth
    @max_depth.setter
    def max_depth (self, max_depth):
        self._.max_depth = int (max_depth)

    @property
    def min_split (self):
        """The minimum number of entries in a node which warrants further
        splitting."""
        return self._.min_split
    @min_split.setter
    def min_split (self, min_split):
        self._.min_split = int (min_split)

    @property
    def num_cuts (self):
        """The number of cuts to try at each potential split."""
        return self._.num_cuts
    @num_cuts.setter
    def num_cuts (self, num_cuts):
        self._.num_cuts = int (num_cuts)

    @property
    def num_random_variables (self):
        """The number of variables to consider using at each node.

        Set to 0 to use every variable at every node.
        """
        return self._.num_random_variables
    @num_random_variables.setter
    def num_random_variables (self, num_random_variables):
        self._.num_random_variables = int (num_random_variables)

    @property
    def separation_type (self):
        """The separation type to use (one of 'cross_entropy', 'gini', or
        'misclass_error': default is 'gini').

        .. warning:: As of this writing, only 'gini' is known to be
            well-tested.
        """
        return self._.separation_type
    @separation_type.setter
    def separation_type (self, sep_type):
        if sep_type not in ('gini', 'cross_entropy', 'misclass_error'):
            raise ValueError (
                'separation type "{0}" not supported'.format (sep_type))
        self._.separation_type = sep_type

    # methods

    def set_defaults (self):
        """Reset default DTLearner properties."""
        self._.set_defaults ()

class BDTLearner (Learner):

    """Train boosted decision trees."""

    def __init__ (self, feature_names=[], weight_name='', bg_weight_name='',
            _=None):
        """Construct a BDTLearner.

        :type   feature_names: list
        :param  feature_names: The names of the event features to use.

        :type   weight_name: str
        :param  weight_name: The name of the :class:`DataSet` column
            corresponding to the event weights ('' for no weighting).

        :type   bg_weight_name: str
        :param  bg_weight_name: The name of the :class:`DataSet` column
            corresponding to the background event weights ('' for no
            weighting).
        """
        if _ is not None:
            self._ = _
            return
        else:
            if not feature_names:
                raise ValueError ('must provide feature names')
        self._ = _pybdt.BDTLearner (feature_names, weight_name, bg_weight_name)

    # properties

    @property
    def after_pruners (self):
        """List of :class:`Pruner`'s which are applied after boosting."""
        return map (wrapped, self._.after_pruners)

    @property
    def before_pruners (self):
        """:class:`Pruner`'s which are applied before boosting."""
        return map (wrapped, self._.before_pruners)

    @property
    def beta (self):
        """The AdaBoost scaling factor."""
        return self._.beta
    @beta.setter
    def beta (self, beta):
        self._.beta = beta

    @property
    def dtlearner (self):
        """The :class:`DTLearner` used to train individual trees."""
        return wrapped (self._.dtlearner)

    @property
    def frac_random_events (self):
        """The fraction of events to use for training each tree.

        Set to 1.0 to use every event for every tree.
        """
        return self._.frac_random_events
    @frac_random_events.setter
    def frac_random_events (self, frac_random_events):
        self._.frac_random_events = frac_random_events

    @property
    def num_trees (self):
        """The number of individual decision trees to train."""
        return self._.num_trees
    @num_trees.setter
    def num_trees (self, n):
        self._.num_trees = n

    @property
    def use_purity (self):
        """Whether to use decision tree leaf purity information.

        If this option is set to True, purity information will be used during
        training as described in J. Zhu, H. Zou, S. Rosset, T. Hastie,
        "Multi-class AdaBoost", 2009.
        """
        return self._.use_purity
    @use_purity.setter
    def use_purity (self, use_purity):
        self._.use_purity = use_purity

    @property
    def quiet (self):
        """Whether to silence the training progress bar."""
        return self._.quiet
    @quiet.setter
    def quiet (self, quiet):
        self._.quiet = quiet

    # methods

    def add_after_pruner (self, pruner):
        """Add a :class:`Pruner` for after boosting."""
        self._.add_after_pruner (unwrapped (pruner))

    def add_before_pruner (self, pruner):
        """Add a :class:`Pruner` for before boosting."""
        self._.add_before_pruner (unwrapped (pruner))

    def clear_after_pruners (self):
        """Clear the set of :class:`Pruner`'s used after boosting."""
        self._.clear_after_pruners ()

    def clear_before_pruners (self):
        """Clear the set of :class:`Pruner`'s used before boosting."""
        self._.clear_before_pruners ()

    def set_defaults (self):
        """Reset default BDTLearner (and internal DTLearner) properties."""
        self._.set_defaults ()

class VineLearner (Learner):

    """Train VineModels."""

    def __init__ (self,
            vine_feature,
            vine_feature_min,
            vine_feature_max,
            vine_feature_width,
            vine_feature_step,
            learner,
            _=None):
        """Construct a VineLearner.

        :type   vine_feature: str
        :param  vine_feature: The feature along which the vine is
            constructed.

        :type   vine_feature_min: double
        :param  vine_feature_min: The lowest feature value to consider.

        :type   vine_feature_max: double
        :param  vine_feature_max: The highest feature value to consider.

        :type   vine_feature_width: double
        :param  vine_feature_width: The width of the vine bins.

        :param  vine_feature_step: double
        :param  vine_feature_step: The difference between the starts and
            ends of consecutive bins.

        """
        if _ is not None:
            self._ = _
            return
        self._ = _pybdt.VineLearner (
                vine_feature, vine_feature_min, vine_feature_max,
                vine_feature_width, vine_feature_step, unwrapped (learner))

    @property
    def quiet (self):
        """Whether to silence the training progress bar."""
        return self._.quiet
    @quiet.setter
    def quiet (self, quiet):
        self._.quiet = quiet

    @property
    def vine_feature (self):
        """Whether to silence the training progress bar."""
        return self._.vine_feature
    @vine_feature.setter
    def vine_feature (self, vine_feature):
        self._.vine_feature = vine_feature

    @property
    def vine_feature_min (self):
        """Whether to silence the training progress bar."""
        return self._.vine_feature_min
    @vine_feature_min.setter
    def vine_feature_min (self, vine_feature_min):
        self._.vine_feature_min = vine_feature_min

    @property
    def vine_feature_max (self):
        """Whether to silence the training progress bar."""
        return self._.vine_feature_max
    @vine_feature_max.setter
    def vine_feature_max (self, vine_feature_max):
        self._.vine_feature_max = vine_feature_max

    @property
    def vine_feature_width (self):
        """Whether to silence the training progress bar."""
        return self._.vine_feature_width
    @vine_feature_width.setter
    def vine_feature_width (self, vine_feature_width):
        self._.vine_feature_width = vine_feature_width

    @property
    def vine_feature_step (self):
        """Whether to silence the training progress bar."""
        return self._.vine_feature_step
    @vine_feature_step.setter
    def vine_feature_step (self, vine_feature_step):
        self._.vine_feature_step = vine_feature_step

    @property
    def learner (self):
        """The underlying learner used for each vine bin."""
        return wrapped (self._.learner)


class Pruner :

    """Prune :class:`DTModel`'s."""

    def __init__ (self, _):
        self._ = _

    # methods

    def prune (self, tree):
        """Prune a decision tree.

        :type   tree: :class:`DTModel`
        :param  tree: The decision tree to prune.

        """
        self._.prune (unwrapped (tree))

class SameLeafPruner (Pruner):

    """Prune trees where adjacent leaves yield the same class."""

    def __init__ (self, _=None):
        if _ is not None:
            self._ = _
            return
        self._ = _pybdt.SameLeafPruner ()

class CostComplexityPruner (Pruner):

    """Prune trees by eliminating nodes with the worst information-added to
    complexity-added ratio."""

    def __init__ (self, strength=None, _=None):
        if _ is not None:
            self._ = _
            return
        elif strength is None:
            raise ValueError ('must give prune strength argument')
        self._ = _pybdt.CostComplexityPruner (strength)

    # properties

    @property
    def strength (self):
        """The pruning strength.

        Once the pruning sequence is computed, this is the percentage
        (0-100) of the prune operations which are actually executed.
        """
        return self._.strength
    @strength.setter
    def strength (self, strength):
        self._.strength = strength

    # static methods

    @staticmethod
    def gain (node):
        """The weighted Gini separation gain of this node.

        :type   node: :class:`DTNode`.
        :param  node: The node.
        """
        return _pybdt.CostComplexityPruner.gain (unwrapped (node))

    @staticmethod
    def rho (node):
        """The relative cost of pruning this node.

        :type   node: :class:`DTNode`.
        :param  node: The node.
        """
        return _pybdt.CostComplexityPruner.rho (unwrapped (node))

class ErrorPruner (Pruner):

    """Prune trees by eliminating the nodes which least improve the estimated
    error.

    .. warning:: As of this writing, this pruning method is not yet
        well-tested and should be considered unsupported.
    """

    def __init__ (self, strength=None, _=None):
        if _ is not None:
            self._ = _
            return
        elif strength is None:
            raise ValueError ('must give strength factor argument')
        self._ = _pybdt.ErrorPruner (strength)

    # properties

    @property
    def strength (self):
        """The pruning strength.

        Once the pruning sequence is computed, this is the percentage
        (0-100) of the prune operations which are actually executed.
        """
        return self._.strength
    @strength.setter
    def strength (self, strength):
        self._.strength = strength

    # methods

    def subtree_error (self, node):
        """The expected error of the subtree below this node.

        :type   node: :class:`DTNode`.
        :param  node: The node.
        """
        return self._.subtree_error (unwrapped (node))

    def node_error (self, node):
        """The expected error of this node (affected by strength parameter).

        :type   node: :class:`DTNode`.
        :param  node: The node.
        """
        return self._.node_error (unwrapped (node))


class Model :

    """Classify events based on some past training by a Learner.

    Learners ultimately return Models upon training.  The Model can be used to
    classify events using the score() methods.
    """

    def __init__ (self, _):
        self._ = _

    # properties

    @property
    def feature_names (self):
        """The names of the event features used by this Model."""
        return self._.feature_names

    # methods

    def score (self, data, use_purity=False, quiet=False):
        """Obtain the score for a set of events.

        :type   data: :class:`DataSet` or dict
        :param  data: Either a DataSet or a DataSet initializer dict.

        :type   use_purity: bool
        :param  use_purity: Whether to use decision tree leaf purity
            information (as opposed to returning -1 or +1 for an individual
            decision tree).

        :type   quiet: bool
        :param  quiet: Whether to suppress the progress bar.

        This convenience method calls :meth:`Model.score_DataSet`,
        :meth:`Model.score_dict` or :meth:`Model.score_event` as
        appropriate.
        """
        kind = Model._data_kind (data)
        if kind == 'event':
            return self.score_event (data, use_purity=use_purity)
        elif kind == 'dict':
            return self.score_dict (data, use_purity, quiet)
        elif kind == 'ds':
            return self.score_DataSet (data, use_purity, quiet)

    def score_DataSet (self, ds, use_purity=False, quiet=False):
        """Obtain the score for a DataSet object.

        :type   ds: :class:`DataSet`
        :param  ds: The dataset.

        :type   use_purity: bool
        :param  use_purity: Whether to use decision tree leaf purity
            information (as opposed to returning -1 or +1 for an individual
            decision tree).

        :type   quiet: bool
        :param  quiet: Whether to suppress the progress bar.

        :return: A numpy.ndarray of per-event scores.
        """
        return self._.score_DataSet (unwrapped (ds), use_purity, quiet)

    def score_dict (self, data, use_purity=False, quiet=False):
        """Obtain the score for a set of events.

        :type   data: dict
        :param  data: A :class:`DataSet` initializer dict.

        :type   use_purity: bool
        :param  use_purity: Whether to use decision tree leaf purity
            information (as opposed to returning -1 or +1 for an individual
            decision tree).

        :type   quiet: bool
        :param  quiet: Whether to suppress the progress bar.

        :return: A numpy.ndarray of per-event scores.
        """
        return self._.score_DataSet (
                unwrapped (DataSet (data)), use_purity, quiet)

    def score_event (self, event, use_purity=False):
        """Obtain the score for a single event.

        :type   event: dict
        :param  event: A mapping from variable names to float values.

        :type   use_purity: bool
        :param  use_purity: Whether to use decision tree leaf purity
            information (as opposed to returning -1 or +1 for an individual
            decision tree).

        :return: A single float BDT score.
        """
        try:
            vals_list = [
                float (event[feature]) for feature in self.feature_names]
        except KeyError as ke:
            raise KeyError ('event missing cut variable "{0}"'.format (
                str (ke)[1:-1]))
        return self._.score_event (vals_list, use_purity)

    # static methods

    @staticmethod
    def _data_kind (data):
        """Check the kind of data being passed in.

        :type   data: :class:`DataSet` or dict
        :param  data: Either a DataSet or a DataSet initializer dict.

        :return: 'ds', 'dict', or 'event'
        """
        if isinstance (data, DataSet):
            return 'ds'
        elif isinstance (data, dict):
            many_events = False
            for v in data.values ():
                if not _is_number (v):
                    many_events = True
            if many_events:
                return 'dict'
            else:
                return 'event'
        else:
            raise TypeError (
                    'could not handle "data" argument of '
                    'type {0}'.format (str (type (data))))

    def _event_list_from_dict (self, event):
        """Convert a single-event dict into a list for C++.

        :type   event: dict
        :param  event: A mapping from variable names to float values.

        :return: list with the data members in proper order.
        """
        try:
            vals_list = [
                float (event[feature]) for feature in self.feature_names]
        except KeyError as ke:
            raise KeyError ('event missing cut variable "{0}"'.format (
                str (ke)[1:-1]))
        return vals_list

class DTNode :

    """Represent a node in a decision tree."""

    def __init__ (self,
                  w_sig, w_bg, n_sig, n_bg, sep_index,
                  sep_gain=None, feature_id=None, feature_val=None,
                  left=None, right=None,
                  _=None):
        """Construct a DTNode.

        :type   w_sig: float
        :param  w_sig: signal weight in this node

        :type   w_bg: float
        :param  w_bg: background weight in this node

        :type   n_sig: int
        :param  n_sig: signal counts in this node

        :type   n_bg: int
        :param  n_bg: background counts in this node

        :type   sep_index: float
        :param  sep_index: separation index at this node

        :type   sep_gain: float
        :param  sep_gain: separation gain at this node (splits only)

        :type   feature_id: int
        :param  feature_id: id of the feature that is cut on at this node
            (splits only)

        :param  feature_val: float
        :param  feature_val: value of the feature that is cut on at this node
            (splits only)

        :param  left: :class:`DTNode`
        :param  left: the node descending to the left from this node (splits
            only)

        :param  right: :class:`DTNode`
        :param  right: the node descending to the right from this node (splits
            only)

        """
        if _ is not None:
            self._ = _
            return

        # check for ambiguous split-or-leaf
        opts = sep_gain, feature_id, feature_val, left, right
        if 1 <= sum ([thing is None for thing in opts]) < len (opts):
            raise ValueError ('all parameters are required for split nodes')

        if left is None:
            self._ = _pybdt.DTNode (
                float (sep_index),
                float (w_sig),
                float (w_bg),
                int (n_sig),
                int (n_bg),
            )
        else:
            self._ = _pybdt.DTNode (
                float (sep_gain),
                float (sep_index),
                int (feature_id),
                float (feature_val),
                float (w_sig),
                float (w_bg),
                int (n_sig),
                int (n_bg),
                unwrapped (left), unwrapped (right),
            )


    # mutators

    def prune (self):
        """Prune the tree at this node.

        This method prunes the tree at this node. The node becomes a leaf. If
        the purity is greater than 50%, it is a signal leaf; otherwise it is a
        background leaf.
        """
        self._.prune ()

    # properties

    @property
    def feature_id (self):
        """The id of the feature for this cut.

        If this is a leaf, feature_id is +1 or -1 for signal or background,
        respectively.
        """
        return self._.feature_id

    @property
    def feature_name (self):
        """The name of the feature for this cut."""
        return self._.feature_name

    @property
    def feature_val (self):
        """The cut value for the feature specified by feature_id at this
        node."""
        return self._.feature_val

    @property
    def is_leaf (self):
        """Whether this node is a leaf."""
        return self._.is_leaf

    @property
    def left (self):
        """The node for feature < feature_val."""
        return wrapped (self._.left) if self._.left is not None else None

    @property
    def max_depth (self):
        """The maximum depth of the tree below this node."""
        return self._.max_depth

    @property
    def n_bg (self):
        """The number of training background events in this node."""
        return self._.n_bg

    @property
    def n_leaves (self):
        """The number of leaves below (and including) this node."""
        return self._.n_leaves

    @property
    def n_sig (self):
        """The number of training signal events in this node."""
        return self._.n_sig

    @property
    def n_total (self):
        """The number of training signal + background events in this node."""
        return self._.n_total

    @property
    def purity (self):
        """The purity of this node."""
        return self._.purity

    @property
    def right (self):
        """The node for feature >= feature_val."""
        return wrapped (self._.right) if self._.right is not None else None

    @property
    def sep_gain (self):
        """The separation gain from this node."""
        return self._.sep_gain

    @property
    def sep_index (self):
        """The separation index at this node."""
        return self._.sep_index

    @property
    def tree_size (self):
        """The size of the tree below (and including) this node."""
        return self._.tree_size

    @property
    def w_bg (self):
        """The sum of background weight in this node."""
        return self._.w_bg

    @property
    def w_sig (self):
        """The sum of signal weight in this node."""
        return self._.w_sig

    @property
    def w_total (self):
        """The sum of signal and background weight in this node."""
        return self._.w_total

class DTModel (Model):

    """Represent a decision tree."""

    def __init__ (self, feature_names=[], root=None, _=None):
        if _ is not None:
            self._ = _
            return
        if not feature_names:
            raise ValueError ('must provide feature names')
        if not root:
            raise ValueError ('must provide root node')
        self._ = _pybdt.DTModel (feature_names, unwrapped (root))

    # properties

    @property
    def root (self):
        """Get the root :class:`DTNode`."""
        return wrapped (self._.root)

    # methods

    def event_variable_importance (self, event, sep_weight=True):
        """Get a dictionary of variable importance values.

        :type   event: dict
        :param  event: A mapping from variable names to float values.

        :type   sep_weight: bool
        :param  sep_weight: Whether to weight nodes where a variable is used by
            separation gain achieved rather than weighting all nodes equally.

        :return: dict with variable name keys and float values from 0 to 1.
        """
        vals = self._event_list_from_dict (event)
        return self._.event_variable_importance (vals, sep_weight)

    def variable_importance (self, sep_weight=True):
        """Get a dictionary of variable importance values.

        :type   sep_weight: bool
        :param  sep_weight: Whether to weight nodes where a variable is used by
            separation gain achieved rather than weighting all nodes equally.

        :return: dict with variable name keys and float values from 0 to 1.
        """
        return self._.variable_importance (sep_weight)

class BDTModel (Model):

    """Represent a boosted decision tree classifier."""

    def __init__ (self, feature_names=[], dtmodels=[], alphas=[],
                  _=None):
        if _ is not None:
            self._ = _
            return

        if not feature_names:
            raise ValueError ('must provide feature names')
        dtmodels = list (dtmodels)
        alphas = map (float, alphas)
        if len (dtmodels) != len (alphas):
            raise ValueError (
                'must provide equal number of DTModels and alphas')
        self._ = _pybdt.BDTModel (feature_names,
                                  map (unwrapped, dtmodels),
                                  alphas)


    # operators

    def __len__ (self):
        """The number of DTModels in this BDTModel."""
        return self.n_dtmodels

    # properties

    @property
    def alphas (self):
        """The alphas, or weights, for each decision tree."""
        return np.array ([
            self._.get_alpha (m) for m in range (self.n_dtmodels)])

    @property
    def dtmodels (self):
        """The DTModels that make up this BDTModel."""
        return np.array ([
            wrapped (self._.get_dtmodel (m)) for m in range (self.n_dtmodels)])

    @property
    def n_dtmodels (self):
        """The number of DTModels in this BDTModel."""
        return self._.n_dtmodels

    # methods

    def get_subset_bdtmodel (self, n_i, n_f):
        """Get a BDTModel using DTModels number n_i thru n_f.

        :type   n_i: int
        :param  n_i: The number of the first DTModel to include.

        :type   n_f: int
        :param  n_f: The number of the last DTModel to include plus one.

        The parameters of this method follow the indexing convention of
        Python's builtin range(i,j).
        """
        return wrapped (self._.get_subset_bdtmodel (n_i, n_f))

    def get_subset_bdtmodel_list (self, dtmodel_indices):
        """Get a BDTModel using DTModels number n_i thru n_f.

        :type   dtmodel_indices: list of int
        :param  dtmodel_indices: The numbers of the DTModels to include.

        """
        if np.asarray (dtmodel_indices).dtype == bool:
            dtmodel_indices = np.where (dtmodel_indices)[0]
        return wrapped (self._.get_subset_bdtmodel_list (
            list(map (int, dtmodel_indices))))

    def get_trimmed_bdtmodel (self, threshold):
        """Get a BDTModel using only DTModels that differ enough from the
        preceeding one.

        :type   threshold: float
        :param  threshold: The minimum percent change in alpha values of
            consecutive trees required in order to keep a given tree

        .. warning:: This method may not be useful, and should be considered
            experimental.
        """
        return wrapped (self._.get_trimmed_bdtmodel (threshold))

    def event_variable_importance (self, event,
            sep_weight=True, tree_weight=True):
        """Get a dictionary of variable importance values.

        :type   event: dict
        :param  event: A mapping from variable names to float values.

        :type   sep_weight: bool
        :param  sep_weight: Whether to weight nodes where a variable is used by
            separation gain achieved rather than weighting all nodes equally.

        :type   tree_weight: bool
        :param  tree_weight: Whether to trees according to their performance on
            the training set rather than weighting all trees equally.

        :return: dict with variable name keys and float values from 0 to 1.
        """
        vals = self._event_list_from_dict (event)
        return self._.event_variable_importance (vals, sep_weight, tree_weight)

    def variable_importance (self, sep_weight=True, tree_weight=True):
        """Get a dictionary of variable importance values.

        :type   sep_weight: bool
        :param  sep_weight: Whether to weight nodes where a variable is used by
            separation gain achieved rather than weighting all nodes equally.

        :type   tree_weight: bool
        :param  tree_weight: Whether to trees according to their performance on
            the training set rather than weighting all trees equally.

        :return: dict with variable name keys and float values from 0 to 1.
        """
        return self._.variable_importance (sep_weight, tree_weight)


class VineModel (Model):

    """Represent a decision tree."""

    def __init__ (self, _):
        self._ = _


class MultiModel1D (Model):

    """A collection of BDTs, one for each bin along a single axis."""

    def __init__ (self, column, bins, bdts):
        """Construct a MultiModel1D.

        :type   column: str
        :param  column: The name of the variable by which the BDTs are binned.

        :type   bins: array-like
        :param  bins: The edges of the bins.

        :type   bdts: array-like
        :param  bdts: The BDTs; it is required that len (bins) = len (bdts) +
            1.
        """
        assert len (bins) == len (bdts) + 1
        self.column = column
        self.bins = np.array (bins)
        self.bdts = np.array (bdts)

    # methods

    def score_DataSet (self, ds):
        """Obtain the scores for a DataSet.

        :type   ds: :class:`DataSet`
        :param  ds: Thte dataset.
        """
        column_vals = ds[self.column]
        out = np.zeros (len (column_vals))
        for i in range (len (self.bins) - 1):
            v1, v2 = self.bins[i:i+2]
            idx = (v1 <= column_vals) * (column_vals < v2)
            out[idx] = self.bdts[i].score (ds)[idx]
        return out

    def score_dict (self, data):
        """Obtain the score for a set of events.

        :type   data: dict
        :param  data: A :class:`DataSet` initializer dict.

        :return: A numpy.ndarray of per-event scores.
        """
        return self.score_DataSet (DataSet (data))

    def score_event (self, event):
        """Obtain the score for an event.

        :type   event: dict
        :param  event: A mapping of BDT variable name -> value.
        """
        column_val = event[self.column]
        i_bin = np.sum (self.bins <= column_val) - 1
        assert 0 <= i_bin < len (self.bdts)
        bdt = self.bdts[i_bin]
        vals_list = [event[feature] for feature in bdt.feature_names]
        return bdt.score (vals_list)

    def get_cut (self, cut_values):
        """A :class:`MultiModel1DCut` for this MultiModel1D.

        :type   cut_values: array-like
        :param  cut_values: The per-bin cut values.
        """
        return MultiModel1DCut (self, cut_values)

class MultiModel1DCut :

    """A cut which takes one value per MultiModel1D bin."""

    def __init__ (self, multi_bdtmodel_1d, cut_values):
        """Construct a :class:`MultiModel1DCut`.

        :type   multi_bdtmodel_1d: :class:`MultiModel1D`
        :param  multi_bdtmodel_1d: The MultiModel1D.

        :type   cut_values: array-like
        :param  cut_values: The cut values, where len (cut_values) == len
            (multi_bdt_model_1d.bdts)
        """
        self.multi_bdtmodel_1d = multi_bdtmodel_1d
        self.cut_values = np.array (cut_values)
        assert len (cut_values) == len (self.multi_bdtmodel_1d.bdts)

    def decision (self, thing, scores=None):
        """Return the cut decision for thing, possibly given scores.

        :type   thing: :class:`DataSet` or dict, or bin column array
        :param  thing: If DataSet or dict, see
            :meth:`Model.score`; otherwise, this is an array of
            values by which the MultiModel1D is binned.

        :param  scores: array-like
        :param  scores: The score or scores for the event or events
            (score or scores are calculated if not given).
        """
        mbdtm = self.multi_bdtmodel_1d
        if scores is None:
            scores = mbdtm.score (thing)
        many_events = False
        if isinstance (thing, DataSet):
            many_events = True
            column_vals = thing[mbdtm.column]
        elif isinstance (thing, np.ndarray) or isinstance (thing, list):
            many_events = True
            column_vals = np.asarray (thing)
        elif isinstance (thing, dict):
            # any non-numbers?
            for k, v in thing.items ():
                if not _is_number (v):
                    many_events = True
                    column_vals = thing[mbdtm.column]
                    break
        if many_events:
            out = np.zeros (len (column_vals), dtype=bool)
            for i in range (len (mbdtm.bins) - 1):
                v1, v2 = mbdtm.bins[i:i+2]
                idx = (v1 <= column_vals) * (column_vals < v2)
                out[idx] = scores[idx] > self.cut_values[i]
            return out
        else:
            scores = mbdtm.score_event (thing)
            column_val = thing[mbdtm.column]
            i_bin = np.sum (mbdtm.bins <= column_val) - 1
            assert 0 <= i_bin < len (self.cut_values)
            return scores > self.cut_values[i_bin]

