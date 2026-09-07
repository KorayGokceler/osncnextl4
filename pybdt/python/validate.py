# validate.py


__doc__ = """Validation suite for PyBDT forests."""

import copy

import os, sys

import numpy as np

try:
    from scipy.interpolate import interp1d
    import scipy.special
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    print ('Matplotlib and/or scipy are missing, so validate.py cannot run.')
    sys.exit()

#from pybdt import BDTModel

from pybdt.util import load
from pybdt.histlite import Binner, Plotter, Style


class StorableObject :

    """Hold an object or a filename for a pickle of it."""

    def __init__ (self, arg):
        """Construct a StorableObject.

        :type   arg: str or something else
        :param  arg: Either the filename or the object itself.

        Upon construction, either the object or a filename for a pickle of the
        object may be given.  This allows for lazy storage and loading of the
        object later on.  Since a str argument is assumed to be a filename,
        this class cannot be used to store a str, but this should be a trivial
        limitation.
        """
        if isinstance (arg, StorableObject):
            self.__setstate__ (arg.__getstate__ ())
        elif isinstance (arg, str):
            filename = arg
            perm_obj = None
        else:
            filename = None
            perm_obj = arg
        self.__setstate__ (dict (filename=filename, perm_obj=perm_obj))

    def __getstate__ (self):
        out = dict (
                filename=self._filename,
                perm_obj=self._perm_obj)
        return out

    def __setstate__ (self, state):
        if state['filename'] is not None:
            filename = os.path.abspath (state['filename'])
        else:
            filename = None
        perm_obj = state['perm_obj']
        if perm_obj is not None:
            self._perm_obj = perm_obj
            self._obj = perm_obj
            self._filename = None
        elif filename is not None:
            self._perm_obj = None
            self._obj = None
            self._filename = filename
        else:
            raise ValueError ('either filename or obj must be given')

    def __call__ (self):
        """Get the storable object."""
        if self._obj is None:
            if self._perm_obj is not None:
                self._obj = self._perm_obj
            elif self._filename is not None:
                self._obj = load (self._filename)
        return self._obj


def get_weights (dataset, arg):
    """Get the weights for the given dataset.

    :type   dataset: :class:`DataSet`
    :param  dataset: The data.

    :type   arg: str, numpy.ndarray, or float
    :param  arg: The name of the weight column, the weights, the livetime, or
        the word 'livetime'.

    Note that the type of dataset is not strictly enforced.  It must be able to
    act as a mapping if the column name is given; it must support len() if the
    livetime is given.
    """
    if isinstance (arg, np.ndarray):
        return arg
    elif isinstance (arg, str):
        if arg == 'livetime':
            if dataset.livetime == -1:
                raise ValueError (
                    'cannot weight by livetime when livetime was not specified'
                )
            return np.ones (len (dataset)) / dataset.livetime
        else:
            return dataset[arg]
    elif isinstance (arg, float):
        return np.ones (len (dataset)) / arg
    else:
        raise TypeError ('arg should be str, numpy.ndarray, or float')

def mix_kwargs (*kwarg_dicts):
    """Combine multiple kwarg dicts, prefering later ones over earlier
    ones."""
    out = {}
    for kwarg_dict in kwarg_dicts:
        out.update (kwarg_dict)
    return out


class CDF :

    """A cumulative distribution function."""

    @staticmethod
    def _sort_arrays (a, weights):
        """Sort a and weights by a values."""
        import numpy as np
        return np.array (sorted (zip (a, weights))).T

    def __init__ (self, a, weights=None, bins=None):
        """Construct a CDF.

        Note that NaNs in ``a``, and their corresponding weights, will be
        ignored.
        """
        idx = np.isfinite (a)
        a = a[idx]
        weights = weights[idx]
        N = len (a)
        if weights is None:
            weights = np.ones (N)
        if bins is not None:
            H = Binner (int (bins)).hist (a, weights)
            cum_x = H.bin_centers
            cum_y = H.sum_normed.cumulative_right.values
        else:
            sort_x, sort_y = self._sort_arrays (a, weights)
            cum_x = sort_x[np.diff (sort_x)>0]
            cum_y = sort_y.cumsum()[np.diff(sort_x)>0] / sort_y.sum()
        cdf_interp = interp1d (cum_x, cum_y)
        def cdf_base (x):
            if x <= cum_x.min ():
                return 0.
            elif x >= cum_x.max ():
                return 1.
            else:
                return cdf_interp (x)

        self._cdf = np.vectorize (cdf_base)

    def __call__ (self, x):
        return self._cdf (x)

def kolmogorov_smirnov_probability (a1, w1, a2, w2, bins=1000):
    """Calculate the Kolmogorov Smirnov p-value for two distributions.

    :type   a1: numpy.ndarray
    :param  a1: The first distribution values.

    :type   w1: numpy.ndarray
    :param  w1: The first distribution weights.

    :type   a2: numpy.ndarray
    :param  a2: The second distribution values.

    :type   w2: numpy.ndarray
    :param  w2: The second distribution weights.

    :type   bins: int
    :param  bins: The number of bins to use.
    """
    cdf1 = CDF (a1, w1, bins=bins)
    cdf2 = CDF (a2, w2, bins=bins)
    a_both = np.r_[a1, a2]
    a_both = a_both[np.isfinite (a_both)]
    x = np.linspace (a_both.min (), a_both.max (), bins)
    D = np.max (np.abs (cdf1 (x) - cdf2 (x)))
    n_eff1 = np.sum (w1)**2 / np.sum (w1**2)
    n_eff2 = np.sum (w2)**2 / np.sum (w2**2)
    N_eff = np.sqrt (n_eff1 * n_eff2 / (n_eff1 + n_eff2))
    # old implementation didn't account for weights:
    #N_eff = np.sqrt (a1.size * a2.size / (a1.size + a2.size))
    d = D * N_eff
    p = scipy.special.kolmogorov (d)
    return p


class Validator :

    """Test and validate BDTs."""

    class Proxy :

        """A class to map data set keys directly to the some object, which may
        require an arbitrary extra dereferencing step.
        """

        def __init__ (self, validator, d, getfunc):
            self.validator = validator
            self.d = d
            self.getfunc = getfunc

        def __getitem__ (self, key):
            return self.getfunc (self.d[key])

    def __init__ (self, bdt):
        """Construct a Validator instance.

        :type   bdt: str or :class:`BDTModel`
        :param  bdt: The BDT model instance :class:`StorableObject` initializer.

        """
        self._bdt = StorableObject (bdt)
        self._datasets = {}
        self._labels = {}
        self._scores = {}
        self._pscores = {}
        self._weights = {}
        self._weight_labels = {}
        self._full_labels = {}
        self._add_to_mc = []
        self._use_as_data = []
        self._style = {}
        self.setup_total_mc ()

    def get_clone (self, bdt=None):
        """Construct a copy Validator.

        :type   bdt: str or :class:`BDTModel`
        :param  bdt: The BDT model instance :class:`StorableObject` initializer.

        """
        out = copy.deepcopy (self)
        if bdt is not None:
            out._bdt = StorableObject (bdt)
            for key, dataset in out._datasets.items ():
                if key in self._scores:
                    out._scores[key] = out.bdt.score (dataset ())
                if key in self._pscores:
                    out._pscores[key] = out.bdt.score (dataset (),
                            use_purity=True)
        return out

    # ----- properties -----

    @property
    def bdt (self):
        """The BDTModel for this Validator."""
        return self._bdt()

    @property
    def scores (self):
        """Mapping of keys to score arrays."""
        # return Validator.Proxy (self, self._scores, lambda s: s)
        return self._scores

    @property
    def pscores (self):
        """Mapping of keys to purity-based score arrays."""
        return self._pscores

    @property
    def weights (self):
        """Mapping of keys to mappings of weight keys to weight arrays."""
        # return Validator.Proxy (self, self._weights, lambda d: d)
        return self._weights

    @property
    def weight_label (self):
        """Mapping of keys to mappings of weight keys to weight labels."""
        # return Validator.Proxy (self, self._weight_labels, lambda d: d)
        return self._weight_labels

    @property
    def style (self):
        """A mapping of set_specs to :class:`histlight.Style` objects."""
        return self._style

    @property
    def label (self):
        """Mapping of keys to labels."""
        # return Validator.Proxy (self, self._labels, lambda l: l)
        return self._labels

    @property
    def data (self):
        """Mapping of keys to DataSets."""
        return Validator.Proxy (self, self._datasets, lambda d: d())

    @property
    def full_label (self):
        """Mapping of (key,wkey) to full labels."""
        return self._full_labels

    # ----- configuration -----

    def add_data (self, key, data, label='',
            scores=True, pscores=False):
        """Add a data set to this Validator.

        :type   key: str
        :param  key: The key for this data set.

        :type   data: str or :class:`DataSet`
        :param  data: The data set :class:`StorableObject` initializer.

        :type   label: str
        :param  label: A nice label for this data set in plots.
        """
        data = StorableObject (data)
        self._datasets[key] = data
        if scores:
            self._scores[key] = self.bdt.score (data())
        if pscores:
            self._pscores[key] = self.bdt.score (data(), use_purity=True)

        self._weights[key] = {}
        self._weight_labels[key] = {}

        if label:
            self._labels[key] = label
        else:
            self._labels[key] = key

    def add_weighting (self, arg, key, wkey='default', label='',
            add_to_mc=False,
            use_as_data=False,
            **style_kwargs
            ):
        """Add a weighting to a dataset.

        :type   arg: str, numpy.ndarray, or float
        :param  arg: The name of the weight column, the weights, or the
            livetime.

        :type   key: str
        :param  key: The key for the desired data set.

        :type   wkey: str
        :param  wkey: The key for this weighting of the data set.

        :type   label: str
        :param  label: A nice label for this weighting.

        :type   add_to_mc: bool
        :param  add_to_mc: Whether to include this dataset weighting in
            "total monte carlo" calculations.

        :type   use_as_data: bool
        :param  use_as_data: Whether to use this dataset weighting as the
            "data" sample in data/mc ratio calculations.

        :type   style_kwargs: dict
        :param  style_kwargs: Arguments to pass to
            the :class:`histlight.Style` constructor.

        """
        self._weights[key][wkey] = get_weights (self.data[key], arg)
        if label:
            self._weight_labels[key][wkey] = label
        else:
            self._weight_labels[key][wkey] = wkey
        if label == '' and wkey == 'default':
            self._full_labels[key,wkey] = self.label[key]
        else:
            self._full_labels[key,wkey] = '{0} {1}'.format (
                    self.label[key], self.weight_label[key][wkey])
        if use_as_data:
            self._use_as_data.append ((key,wkey))
        if add_to_mc:
            self._add_to_mc.append ((key,wkey))

        self._style[key,wkey] = Style (
                label=self.full_label[key,wkey], **style_kwargs)

    def clear_weightings (self):
        """Erase any stored weightings."""
        for key in self._datasets:
            self._weights[key] = {}
            self._weight_labels[key] = {}
        self._full_labels = {}
        self._add_to_mc = []
        self._use_as_data = []
        self._theme = {}

    def setup_total_mc (self, label='Total MC', **style_kwargs):
        """Setup total monte carlo plotting properties.

        :type   label: str
        :param  label: The label for total monte carlo lines.

        Any additional arguments are passed to the
        :class:`histlight.Style` constructor.

        """
        self._total_mc_label = label
        self._style['total_mc'] = Style (label=label, **style_kwargs)

    def load_all_data (self, dbg=False):
        """Load all data from disk into RAM."""
        import sys
        for k in sorted (self._datasets.keys ()):
            if dbg:
                print ('* {0} ...'.format (k), file=sys.stderr)
            self.data[k]

    # ----- calculations -----

    def eval (self, set_spec, expr, names={}):
        """Evaluate an expression in terms of variables in a dataset.

        :type   set_spec: str or tuple
        :param  set_spec: See :meth:`Validator.get_key_wkey`

        :type   expr: str
        :param  expr: The expression to evaluate.

        :type   names: dict
        :param  names: Names to be passed into eval.

        :return: The result of the expression evaluation.

        When expr is evaluated, each variable stored in the dataset will be
        available.  The variables 'scores', 'pscores' and 'weights' will also
        be available.  If the dataset has a livetime set, 'livetime' will also
        be available.

        Other allowed identifiers are np (Numpy) and scipy , in addition to
        anything specified in the names parameter.

        This method is implemented in terms of :meth:`DataSet.eval`.
        """
        key, wkey = self.get_key_wkey (set_spec)
        ds = self.data[key]
        all_names = dict (weights=self.weights[key][wkey])
        if key in self.scores:
            all_names['scores'] = self.scores[key]
        if key in self.pscores:
            all_names['pscores'] = self.pscores[key]
        all_names.update (names)
        result = ds.eval (expr, names=all_names)
        return result

    def get_key_wkey (self, set_spec):
        """Get the key and weighting key for a given data set spec.

        :type   set_spec: str or tuple
        :param  set_spec: (key,wkey), or just key (in which case,
            wkey=='default' is assumed)

        :return: A (key,wkey) tuple.
        """
        if isinstance (set_spec, tuple):
            assert len (set_spec) == 2
            key, wkey = set_spec
        else:
            key = set_spec
            wkey = 'default'
        return key, wkey

    def get_values_weights (self, set_spec, expr,
            cut=None,
            eval_names={}):
        """Evaluate an expression, and get weights and scores.

        :type   set_spec: str or tuple
        :param  set_spec: See :meth:`Validator.get_key_wkey`

        :type   expr: str
        :param  expr: An expression for :meth:`Validator.eval` which returns an
            numerical array.

        :type   cut: str
        :param  cut: An expression for :meth:`Validator.eval` which returns an
            array of bools, where True means "include this event".

        :type   eval_names: dict
        :param  eval_names: Names to be passed to :meth:`Validator.eval` for
            the main expression or the cut expression.

        :return: A ([expression result], weights, scores) tuple
        """
        key, wkey = self.get_key_wkey (set_spec)
        values = self.eval (set_spec, expr, eval_names)
        weights = self.weights[key][wkey]
        if cut is None:
            idx = np.ones (len (weights), dtype=bool)
        else:
            idx = self.eval (set_spec, cut, eval_names)
        return values[idx], weights[idx]

    def get_kolmogorov_smirnov_probability (self, set_spec_1, set_spec_2,
            expr='scores',
            bins=1000):
        """Calculate the Kolmogorov-Smirnov p value for two distributions.
        using :func:`kolmogorov_smirnov_probability`.

        :type   set_spec_1: str or tuple
        :param  set_spec_1: The first data and weighting.

        :type   set_spec_2: str or tuple
        :param  set_spec_2: The second data and weighting.

        :type   expr: str
        :param  expr: An expression for :meth:`Validator.eval` which returns an
            numerical array.

        :type   bins: int
        :param  bins: The number of bins to use.

        :return: The Kolmogorov-Smirnov p value.
        """
        key1, wkey1 = self.get_key_wkey (set_spec_1)
        key2, wkey2 = self.get_key_wkey (set_spec_2)
        a1, w1 = self.get_values_weights (set_spec_1, expr)
        a2, w2 = self.get_values_weights (set_spec_2, expr)
        return kolmogorov_smirnov_probability (a1, w1, a2, w2, bins=bins)

    def get_correlation (self, set_spec, expr1, expr2,
            cut=None,
            eval_names={}):
        """Get the correlation between two variables for a data set.

        :type   set_spec: str or tuple
        :param  set_spec: See :meth:`Validator.get_key_wkey`

        :type   expr1: str
        :param  expr1: The first variable or expression.

        :type   expr2: str
        :param  expr2: The second variable or expression.

        :type   cut: str
        :param  cut: An expression for :meth:`Validator.eval` which returns an
            array of bools, where True means "include this event".

        :type   eval_names: dict
        :param  eval_names: Names to be passed to :meth:`Validator.eval` for
            cut evaluation.

        """
        def weighted_mean (arr, weight):
            mean = np.sum (weight * arr) / np.sum (weight)
            return mean

        key, wkey = self.get_key_wkey (set_spec)
        a1 = self.data[key].eval (expr1)
        a2 = self.data[key].eval (expr2)
        w = self.weights[key][wkey]

        mask = np.isfinite (a1) * np.isfinite (a2) * np.isfinite (w)
        if cut is not None:
            mask *= self.eval (set_spec, cut, names=eval_names)
        a1 = a1[mask]
        a2 = a2[mask]
        w = w[mask]

        wmean1 = weighted_mean (a1, w)
        wmean2 = weighted_mean (a2, w)

        wcov12 = np.sum (w * (a1 - wmean1) * (a2 - wmean2)) / np.sum (w)
        wcov11 = np.sum (w * (a1 - wmean1) * (a1 - wmean1)) / np.sum (w)
        wcov22 = np.sum (w * (a2 - wmean2) * (a2 - wmean2)) / np.sum (w)
        wcorr = wcov12 / np.sqrt (wcov11 * wcov22)

        return wcorr

    def get_Hist (self, set_spec, expr,
            bins=100,
            range=None,
            normed=False,
            cut=None,
            eval_names={}):
        """Get a :class:`histlite.Hist` for a variable for a given data
        set and weighting.

        :type   set_spec: str or tuple
        :param  set_spec: See :meth:`Validator.get_key_wkey`

        :type   expr: str
        :param  expr: An expression for :meth:`Validator.eval` which returns an
            numerical array.

        :type   bins: int
        :param  bins: The number of bins to create [default: 100].

        :type   range: 2-tuple
        :param  range: The range over which to make the histogram [default:
            (min value, max value) found in all included data sets].

        :type   normed: bool
        :param  normed: Whether to normalize the y axis histograms.

        :type   cut: str
        :param  cut: An expression for :meth:`Validator.eval` which returns an
            array of bools, where True means "include this event".

        :type   eval_names: dict
        :param  eval_names: Names to be passed to :meth:`Validator.eval` for
            the main expression or the cut expression.

        :return: An instance of :class:`histlite.Hist`.
        """
        binner = Binner (bins, range)
        values, weights = self.get_values_weights (
                    set_spec, expr,
                    cut=cut, eval_names=eval_names)
        hist = binner.hist (values, weights)
        if normed:
            return hist.sum_normed
        else:
            return hist

    def get_range (self, set_specs, expr,
            cut=None,
            eval_names={}):
        """Get the range of values of variable (after transform) for given
        datasets and weightings.

        :type   set_specs: list
        :param  set_specs: The datasets and weightings (see
            :meth:`Validator.get_key_wkey`).

        :type   expr: str
        :param  expr: An expression for :meth:`Validator.eval` which returns an
            numerical array.

        :type   cut: str
        :param  cut: An expression for :meth:`Validator.eval` which returns an
            array of bools, where True means "include this event".

        :type   eval_names: dict
        :param  eval_names: Names to be passed to :meth:`Validator.eval`.

        :return: A (min_val, max_val) tuple.

        """
        min_val, max_val = np.inf, -np.inf
        for set_spec in set_specs:
            if set_spec == 'total_mc':
                continue
            values, weights = self.get_values_weights (
                    set_spec, expr,
                    cut=cut, eval_names=eval_names)
            if len (values) > 0:
                min_val = min (min_val, values.min ())
                max_val = max (max_val, values.max ())
        if min_val > max_val:
            raise ValueError ('range could not be set: no events found')
        return min_val, max_val

    # ----- plotting -----

    def plot_variable (self,
            axes,
            expr,
            kind,
            left_set_specs,
            right_set_specs=[],
            twin_axes=None,
            data_mc=False,
            **kwargs):
        """Create a BDT score distribution, rate plot, or efficiency plot.

        :type   axes: matplotlib.axes.Axes
        :param  axes: The Axes on which to draw the plot.

        :type   expr: str
        :param  expr: An expression for :meth:`Validator.eval` which returns an
            numerical array.

        :type   kind: str
        :param  kind: One of 'dist', 'rate' or 'eff'.

        :type   left_set_specs: list
        :param  left_set_specs: What to plot on the main y axis (see
            :meth:`Validator.get_key_wkey`).

        :type   right_set_specs: list
        :param  right_set_specs: What to plot on the secondary y axis (see
            :meth:`Validator.get_key_wkey`).

        :type   twin_axes: matplotlib.axes.Axes
        :param  twin_axes: The secondary-y axes, if already created with
            axes.twinx().

        :type   data_mc: bool
        :param  data_mc: Plot ratio of given curves to total_mc.

        If 'total_mc' is included in either left_set_specs or right_set_specs,
        then a total monte carlo line will be added.

        The following additional kwargs are allowed.

        :type   legend: bool or dict
        :param  legend: If True or non-empty dict, draw a legend. If dict, use
            as keyword arguments for matplotlib.axes.Axes.legend.

        :type   cut: str
        :param  cut: An expression for :meth:`Validator.eval` which returns an
            array of bools, where True means "include this event".

        :type   eval_names: dict
        :param  eval_names: Names to be passed to :meth:`Validator.eval`.

        :type   log: bool
        :param  log: Whether to use a log-y scale.

        :type   normed: bool
        :param  normed: Whether to normalize the y axis histograms.

        :type   left_log: bool
        :param  left_log: Whether to use a log-y scale on the main y axis.

        :type   left_normed: bool
        :param  left_normed: Whether to normalize the main y axis histograms.

        :type   right_log: bool
        :param  right_log: Whether to use a log-y scale on the secondary y
            axis.

        :type   right_normed: bool
        :param  right_normed: Whether to normalize the secondary y axis
            histograms.

        :type   dbg: bool
        :param  dbg: Whether to print debugging/logging information while
            plotting.
        """
        legend = kwargs.get ('legend', False)
        cut = kwargs.get ('cut', None)
        eval_names = kwargs.get ('eval_names', {})
        bins = kwargs.get ('bins', 100)
        range = kwargs.get ('range', None)
        log = kwargs.get ('log', False)
        normed = kwargs.get ('normed', False)
        left_log = kwargs.get ('left_log', log)
        left_normed = kwargs.get ('left_normed', normed)
        right_log = kwargs.get ('right_log', log)
        right_normed = kwargs.get ('right_normed', normed)
        dbg = kwargs.get ('dbg', False)

        if range is None:
            range = self.get_range (
                    left_set_specs + (right_set_specs or []),
                    expr, cut=cut, eval_names=eval_names)

        def prdbg (*args, **kwargs):
            if not dbg:
                return
            import sys
            kwargs['file'] = sys.stderr
            print (*args, **kwargs)

        plotter = Plotter (
                axes, twin_axes=twin_axes, log=log, twin_log=right_log)

        if data_mc:
            total_mc_h = None
            for mc_spec in self._add_to_mc:
                mc_h = self.get_Hist (mc_spec, expr,
                        bins=bins, range=range,
                        cut=cut, eval_names=eval_names)
                if total_mc_h is None:
                    total_mc_h = mc_h
                else:
                    total_mc_h += mc_h
            if left_normed:
                total_mc_h = total_mc_h.sum_normed
            if kind == 'dist':
                total_mc_line = total_mc_h
            elif kind == 'rate':
                total_mc_line = total_mc_h.cumulative_left
            elif kind == 'eff':
                total_mc_line = total_mc_h.cumulative_left / total_mc_h.sum
            else:
                raise ValueError ('kind {0} not understood'.format (kind))

        def handle_set_specs (twin, set_specs, log, normed):
            for set_spec in set_specs:
                prdbg ('- handling {0} ...'.format (set_spec))
                if set_spec == 'total_mc':
                    h = None
                    for mc_spec in self._add_to_mc:
                        mc_h = self.get_Hist (mc_spec, expr,
                                bins=bins, range=range,
                                cut=cut, eval_names=eval_names)
                        if h is None:
                            h = mc_h
                        else:
                            h += mc_h
                    st = self.style['total_mc']
                else:
                    key, wkey = self.get_key_wkey (set_spec)
                    h = self.get_Hist (set_spec, expr,
                            bins=bins, range=range,
                            cut=cut, eval_names=eval_names)
                    st = self.style[key,wkey]
                prdbg ('  - getting final line...')
                if normed:
                    h = h.sum_normed
                if kind == 'dist':
                    line = h
                elif kind == 'rate':
                    line = h.cumulative_left
                elif kind == 'eff':
                    line = h.cumulative_left / h.sum
                else:
                    raise ValueError ('kind {0} not understood'.format (kind))
                if data_mc:
                    line = line / total_mc_line
                prdbg ('  - adding painter...')
                if log and h.sum <= 0:
                    print ('warning: pybdt.validate: '
                            'omitting line for {0}'.format (set_spec),
                            file=sys.stderr)
                else:
                    plotter.add (line, style=st, twin=twin)

        prdbg ('- handling main y axis...')
        handle_set_specs (False, left_set_specs, left_log, left_normed)
        if right_set_specs:
            prdbg ('- handling secondary y axis...')
            handle_set_specs (True, right_set_specs, right_log, right_normed)

        plotter.finish (legend=legend)

        # handle default axis ranges
        axes.set_xlim (range)
        if not data_mc:
            if left_set_specs:
                if kind == 'eff':
                    axes.set_ylim (ymax=1)
                if not left_log:
                    axes.set_ylim (ymin=0)
            if right_set_specs:
                if kind == 'eff':
                    axes.set_ylim (ymax=1)
                if not right_log:
                    plotter.twin_axes.set_ylim (ymin=0)
        else:
            if left_set_specs:
                if not left_log:
                    axes.set_ylim (0, 4)
                else:
                    axes.set_ylim (1e-1, 1e2)
            if right_set_specs:
                if not right_log:
                    plotter.twin_axes.set_ylim (0, 4)
                else:
                    plotter.twin_axes.set_ylim (1e-1, 1e2)

        if data_mc:
            axes.axhline (1.0, ls='--', color='.3')

    def create_plot (self, expr, kind, left_set_specs, right_set_specs=[],
            fignum=None,
            **kwargs):
        """Create a BDT score distribution, rate plot, or efficiency plot.

        :type   expr: str
        :param  expr: An expression for :meth:`Validator.eval` which returns an
            numerical array.

        :type   kind: str
        :param  kind: One of 'dist', 'rate' or 'eff'.

        :type   left_set_specs: list
        :param  left_set_specs: What to plot on the main y axis (see
            :meth:`Validator.get_key_wkey`).

        :type   right_set_specs: list
        :param  right_set_specs: What to plot on the secondary y axis (see
            :meth:`Validator.get_key_wkey`).

        :type   fignum: int
        :param  fignum: If given, create a figure with the given number using
            matplotlib.pyplot; otherwise use matplotlib.figure.Figure.

        A new figure is created using :meth:`Validator.plot_variable`.  The
        following keyword arguments can be used to create dual linear/log
        plots.

        :type   dual: bool
        :param  dual: Make a dual figure with a linear y scale on the
            left and a log y scale on the right.

        :type   data_mc: bool
        :param  data_mc: Include data/mc ratio plot(s).

        :type   linear_kwargs: dict
        :param  linear_kwargs: If given, this dict of keyword arguments
            supercedes individually passed keyword arguments for the linear
            plot.

        :type   log_kwargs: dict
        :param  log_kwargs: If given, this dict of keyword arguments
            supercedes individually passed keyword arguments for the linear
            plot.

        The following keyword arguments determine the plot appearance.

        :type   title: str
        :param  title: The title of the plot.

        :type   xlabel: str
        :param  xlabel: The xaxis label (default: expr).

        :type   ylabel: str
        :param  ylabel: The xaxis label.

        :type   left_ylabel: str
        :param  left_ylabel: The main y axis label.

        :type   right_ylabel: str
        :param  right_ylabel: The secondary y axis label.

        :type   data_mc_ylabel: str
        :param  data_mc_ylabel: The data/mc ratio plot y axis label (default:
            "data/mc ratio").

        :type   grid: bool
        :param  grid: Whether to include grids (default: True)

        :type   margin_left: float
        :param  margin_left: Fraction of width to reserve as left margin.

        :type   margin_right: float
        :param  margin_right: Fraction of width to reserve as right margin.

        :type   margin_top: float
        :param  margin_top: Fraction of width to reserve as top margin.

        :type   margin_bottom: float
        :param  margin_bottom: Fraction of width to reserve as bottom margin.

        :type   aspect: float
        :param  aspect: Width / height ratio.

        All other keyword arguments are passed through to
        :meth:`Validator.plot_variable`.

        :return: A dict with string keys and values of the new
            :class:`matplotlib.figure.Figure` and each set of axes used.
            Depending on the above argument, some or all of the following keys
            will be available:

            ['fig', 'first_main_ax', 'twin_first_main_ax', 'first_dm_ax',
            'second_main_ax', 'twin_second_main_ax', 'second_dm_ax']

        """
        dual = kwargs.pop ('dual', False)
        data_mc = kwargs.pop ('data_mc', False)
        linear_kwargs = mix_kwargs (
                copy.deepcopy (kwargs),
                kwargs.pop ('linear_kwargs', {}))
        linear_kwargs['log'] = False
        log_kwargs = mix_kwargs (
                copy.deepcopy (kwargs),
                linear_kwargs.pop ('log_kwargs', {}))
        log_kwargs['log'] = True

        title = kwargs.pop ('title', None)
        data_mc_ylabel = kwargs.pop ('data_mc_ylabel', 'data/mc ratio')
        margin_left = kwargs.pop ('margin_left', .09)
        margin_right = kwargs.pop ('margin_right', .09)
        margin_top = kwargs.pop ('margin_top', .09)
        margin_bottom = kwargs.pop ('margin_bottom', .09)
        aspect = kwargs.pop ('aspect', 4/3.)
        grid = kwargs.pop ('grid', True)

        out = {}

        fig_height = 7
        fig_width = aspect * fig_height
        fig_kwargs = {}


        if fignum is not None:
            Figure = plt.figure
            fig_kwargs['num'] = fignum
        else:
            Figure = mpl.figure.Figure

        if dual:
            fig_kwargs['figsize'] = 2 * fig_width, fig_height
            ax_width = .5
            margin_left *= ax_width
            margin_right *= ax_width
            first_kwargs = linear_kwargs
            second_kwargs = log_kwargs
        else:
            fig_kwargs['figsize'] = fig_width, fig_height
            ax_width = 1.
            log = kwargs.get ('log', False)
            first_kwargs = log_kwargs if log else linear_kwargs

        fig = Figure (**fig_kwargs)
        if fignum is not None:
            fig.clf ()
        else:
            mpl.backends.backend_agg.FigureCanvasAgg (fig)
        out['fig'] = fig

        if data_mc:
            dm_ratio = 1./3
            dm_ax_height = dm_ratio - margin_bottom
            main_ax_height = 1 - dm_ratio - margin_top
            main_ax_bottom = dm_ratio
        else:
            main_ax_height = 1 - margin_top - margin_bottom
            main_ax_bottom = margin_bottom

        def set_xlabel (axes, kwargs):
            axes.set_xlabel (kwargs.get ('xlabel', expr))

        def set_ylabels (axes, twin_axes, kwargs):
            axes.set_ylabel (kwargs.get ('left_ylabel', ''))
            if right_set_specs:
                twin_axes.set_ylabel (kwargs.get ('right_ylabel', ''))

        # first main axes
        rect = [
            margin_left,
            main_ax_bottom,
            ax_width - margin_left - margin_right,
            main_ax_height]
        first_main_ax = fig.add_axes (rect)
        twin_first_main_ax = None
        if right_set_specs:
            twin_first_main_ax = first_main_ax.twinx ()
        self.plot_variable (first_main_ax, expr, kind,
                left_set_specs, right_set_specs, twin_first_main_ax,
                **first_kwargs)
        set_ylabels (first_main_ax, twin_first_main_ax, first_kwargs)
        if grid:
            first_main_ax.grid ()
        if twin_first_main_ax:
            out['first_main_ax'] = first_main_ax
            out['twin_first_main_ax'] = twin_first_main_ax
        else:
            out['first_main_ax'] = first_main_ax
        if data_mc:
            # first data/mc axes
            mpl.artist.setp (
                    first_main_ax.get_xaxis ().get_majorticklabels (),
                    visible=False)
            mpl.artist.setp (
                    first_main_ax.get_yaxis ().get_majorticklabels ()[:1],
                    visible=False)
            rect = [
                margin_left,
                margin_bottom,
                ax_width - margin_left - margin_right,
                dm_ax_height]
            first_dm_ax = fig.add_axes (rect)
            first_kwargs.update (dict (data_mc=True, legend=False))
            if 'range' not in first_kwargs:
                first_kwargs['range'] = first_main_ax.get_xlim ()
            self.plot_variable (first_dm_ax, expr, kind,
                    self._use_as_data,
                    **first_kwargs)
            mpl.artist.setp (
                    first_dm_ax.get_yaxis ().get_majorticklabels ()[-1:],
                    visible=False)
            set_xlabel (first_dm_ax, first_kwargs)
            first_dm_ax.set_ylabel (data_mc_ylabel)
            out['first_dm_ax'] = first_dm_ax
        else:
            set_xlabel (first_main_ax, first_kwargs)
        if dual:
            # second main axes
            rect = [
                ax_width + margin_left,
                main_ax_bottom,
                ax_width - margin_left - margin_right,
                main_ax_height]
            second_main_ax = fig.add_axes (rect)
            twin_second_main_ax = None
            if right_set_specs:
                twin_second_main_ax = second_main_ax.twinx ()
            self.plot_variable (second_main_ax, expr, kind,
                    left_set_specs, right_set_specs, twin_second_main_ax,
                    **second_kwargs)
            set_ylabels (second_main_ax, twin_second_main_ax, second_kwargs)
            if grid:
                second_main_ax.grid ()
            if twin_second_main_ax:
                out['second_main_ax'] = second_main_ax
                out['twin_second_main_ax'] = twin_second_main_ax
            else:
                out['second_main_ax'] = second_main_ax
            if data_mc:
                # seconddata/mc axes
                mpl.artist.setp (
                        second_main_ax.get_xaxis ().get_majorticklabels (),
                        visible=False)
                mpl.artist.setp (
                        second_main_ax.get_yaxis ().get_majorticklabels ()[:1],
                        visible=False)
                rect = [
                    ax_width + margin_left,
                    margin_bottom,
                    ax_width - margin_left - margin_right,
                    dm_ax_height]
                second_dm_ax = fig.add_axes (rect)
                second_kwargs.update (dict (data_mc=True, legend=False))
                if 'range' not in second_kwargs:
                    second_kwargs['range'] = second_main_ax.get_xlim ()
                self.plot_variable (second_dm_ax, expr, kind,
                        self._use_as_data,
                        **second_kwargs)
                mpl.artist.setp (
                        second_dm_ax.get_yaxis ().get_majorticklabels ()[-1:],
                        visible=False)
                set_xlabel (second_dm_ax, second_kwargs)
                second_dm_ax.set_ylabel (data_mc_ylabel)
                out['second_dm_ax'] = second_dm_ax
            else:
                set_xlabel (second_main_ax, second_kwargs)

        if title:
            fig.suptitle (title, size='large', weight='bold')

        return out

    def create_overtrain_check_plot (self,
            sig_train_set_spec, sig_test_set_spec,
            bg_train_set_spec, bg_test_set_spec,
            legend=dict (loc='best'),
            legend_side='right',
            fignum=None,
            expr='scores',
            **kwargs
            ):
        """Create an overtraining check plot.

        :type   sig_train_set_spec: str or tuple
        :param  sig_train_set_spec: The signal training set. (Each signal or
            background training or testing set is specified as with
            :meth:`Validator.get_key_wkey`)

        :type   sig_test_set_spec: str or tuple
        :param  sig_test_set_spec: The signal testing set.

        :type   bg_train_set_spec: str or tuple
        :param  bg_train_set_spec: The background training set.

        :type   bg_test_set_spec: str or tuple
        :param  bg_test_set_spec: The background testing set.

        :type   legend: bool or dict
        :param  legend: If True or non-empty dict, draw a legend. If dict, use
            as keyword arguments for matplotlib.axes.Axes.legend.

        :type   legend_side: str
        :param  legend_side: Either 'left' or 'right'; on which axes to draw
            the legend.

        :type   fignum: int
        :param  fignum: If given, create a figure with the given number using
            matplotlib.pyplot; otherwise use matplotlib.figure.Figure.

        :type   expr: str
        :param  expr: The expression to evaluate on each data set.

        The following additional keyword arguments are allowed:

        :type   title: str
        :param  title: The title of the plot.

        :type   xlabel: str
        :param  xlabel: The xaxis label.

        :type   ylabel: str
        :param  ylabel: The xaxis label.

        :type   left_ylabel: str
        :param  left_ylabel: The main y axis label.

        :type   right_ylabel: str
        :param  right_ylabel: The secondary y axis label.

        :type   margin_left: float
        :param  margin_left: Fraction of width to reserve as left margin.

        :type   margin_right: float
        :param  margin_right: Fraction of width to reserve as right margin.

        :type   margin_top: float
        :param  margin_top: Fraction of width to reserve as top margin.

        :type   margin_bottom: float
        :param  margin_bottom: Fraction of width to reserve as bottom margin.

        :type   bins: int
        :param  bins: Number of bins to use in histograms.


        :return: A dict with string keys and values of the new
            :class:`matplotlib.figure.Figure` and each set of axes used.
            Depending on the above argument, some or all of the following keys
            will be available:

            ['fig', 'first_main_ax', 'twin_first_main_ax', 'first_ratio_ax',
            'second_main_ax', 'twin_second_main_ax', 'second_ratio_ax']

        """
        out = {}

        title = kwargs.pop ('title', 'Overtraining Check')
        margin_left = kwargs.pop ('margin_left', .09)
        margin_right = kwargs.pop ('margin_right', .09)
        margin_top = kwargs.pop ('margin_top', .09)
        margin_bottom = kwargs.pop ('margin_bottom', .09)

        bins = kwargs.pop ('bins', 100)

        fig_height = 7
        fig_width = 4./3 * fig_height
        fig_kwargs = {}

        if legend_side == 'left':
            left_legend, right_legend = legend, {}
        elif legend_side == 'right':
            left_legend, right_legend = {}, legend
        else:
            raise ValueError ('legend_side must be either "left" or "right"')


        if fignum is not None:
            Figure = plt.figure
            fig_kwargs['num'] = fignum
        else:
            Figure = mpl.figure.Figure

        fig_kwargs['figsize'] = 2 * fig_width, fig_height
        ax_width = .5
        margin_left *= ax_width
        margin_right *= ax_width

        fig = Figure (**fig_kwargs)
        if fignum is not None:
            fig.clf ()
        else:
            mpl.backends.backend_agg.FigureCanvasAgg (fig)
        out['fig'] = fig

        ratio_ratio = 1./3
        ratio_ax_height = ratio_ratio - margin_bottom
        main_ax_height = 1 - ratio_ratio - margin_top
        main_ax_bottom = ratio_ratio

        all_set_specs = (sig_train_set_spec, sig_test_set_spec,
                bg_train_set_spec, bg_test_set_spec)

        if 'range' not in kwargs:
            range = self.get_range (all_set_specs, expr)
        else:
            range = kwargs['range']

        h_sig_train = self.get_Hist (sig_train_set_spec, expr,
                bins=bins, range=range)
        h_sig_test = self.get_Hist (sig_test_set_spec, expr,
                bins=bins, range=range)

        h_bg_train = self.get_Hist (bg_train_set_spec, expr,
                bins=bins, range=range)
        h_bg_test = self.get_Hist (bg_test_set_spec, expr,
                bins=bins, range=range)

        p_ks_sig = self.get_kolmogorov_smirnov_probability (
                sig_train_set_spec, sig_test_set_spec, expr)
        p_ks_bg = self.get_kolmogorov_smirnov_probability (
                bg_train_set_spec, bg_test_set_spec, expr)

        def add_hists (plotter):
            plotter.add (h_sig_train, twin=True,
                    style=Style (color='blue', label='Signal, Training'))
            plotter.add (h_sig_test, twin=True,
                    style=Style (
                        line=False, markers=True, marker='.', errorbars=True,
                        color='blue',
                        label='Signal, Testing\n(KS p = {0:.3e})'.format (
                            p_ks_sig)))
            plotter.add (h_bg_train,
                    style=Style (color='red', label='Background, Training'))
            plotter.add (h_bg_test,
                    style=Style (
                        line=False, markers=True, marker='.', errorbars=True,
                        color='red',
                        label='Background, Testing\n(KS p = {0:.3e})'.format (
                            p_ks_bg)))

        def add_ratios (plotter):
            plotter.add (h_bg_test / h_bg_train,
                    style=Style (
                        line=False, markers=True, marker='.', errorbars=True,
                        color='red'))
            plotter.add (h_sig_test / h_sig_train,
                    style=Style (
                        line=False, markers=True, marker='.', errorbars=True,
                        color='blue'))

        # first main axes
        rect = [
            margin_left,
            main_ax_bottom,
            ax_width - margin_left - margin_right,
            main_ax_height]
        first_main_ax = fig.add_axes (rect)
        twin_first_main_ax = first_main_ax.twinx ()
        plotter = Plotter (first_main_ax, twin_first_main_ax)
        add_hists (plotter)
        plotter.finish (legend=left_legend)
        first_main_ax.set_xlim (range)
        first_main_ax.set_ylim (ymin=0)
        twin_first_main_ax.set_ylim (ymin=0)
        first_main_ax.set_ylabel (kwargs.get ('left_ylabel', ''))
        twin_first_main_ax.set_ylabel (kwargs.get ('right_ylabel', ''))
        out['first_main_ax'] = first_main_ax
        out['twin_first_main_ax'] = twin_first_main_ax
        # first data/mc axes
        mpl.artist.setp (
                first_main_ax.get_xaxis ().get_majorticklabels (),
                visible=False)
        mpl.artist.setp (
                first_main_ax.get_yaxis ().get_majorticklabels ()[:1],
                visible=False)
        rect = [
            margin_left,
            margin_bottom,
            ax_width - margin_left - margin_right,
            ratio_ax_height]
        first_ratio_ax = fig.add_axes (rect)
        plotter = Plotter (first_ratio_ax)
        add_ratios (plotter)
        plotter.finish ()
        first_ratio_ax.set_xlabel (kwargs.get ('xlabel', 'BDT score'))
        first_ratio_ax.set_xlim (range)
        first_ratio_ax.set_ylim (0, 4)
        first_ratio_ax.axhline (1.0, ls='--', color='.3')
        mpl.artist.setp (
                first_ratio_ax.get_yaxis ().get_majorticklabels ()[-1:],
                visible=False)
        out['first_ratio_ax'] = first_ratio_ax
        # second main axes
        rect = [
            ax_width + margin_left,
            main_ax_bottom,
            ax_width - margin_left - margin_right,
            main_ax_height]
        second_main_ax = fig.add_axes (rect)
        twin_second_main_ax = second_main_ax.twinx ()
        plotter = Plotter (second_main_ax, twin_second_main_ax, log=True)
        add_hists (plotter)
        plotter.finish (legend=right_legend)
        second_main_ax.set_xlim (range)
        second_main_ax.set_ylabel (kwargs.get ('left_ylabel', ''))
        twin_second_main_ax.set_ylabel (kwargs.get ('right_ylabel', ''))
        out['second_main_ax'] = second_main_ax
        out['twin_second_main_ax'] = twin_second_main_ax
        # second data/mc axes
        mpl.artist.setp (
                second_main_ax.get_xaxis ().get_majorticklabels (),
                visible=False)
        mpl.artist.setp (
                second_main_ax.get_yaxis ().get_majorticklabels ()[:1],
                visible=False)
        rect = [
            ax_width + margin_left,
            margin_bottom,
            ax_width - margin_left - margin_right,
            ratio_ax_height]
        second_ratio_ax = fig.add_axes (rect)
        plotter = Plotter (second_ratio_ax, log=True)
        add_ratios (plotter)
        plotter.finish ()
        second_ratio_ax.set_xlabel (kwargs.get ('xlabel', 'BDT score'))
        second_ratio_ax.set_xlim (range)
        second_ratio_ax.set_ylim (1e-1, 1e2)
        second_ratio_ax.axhline (1.0, ls='--', color='.3')
        mpl.artist.setp (
                second_ratio_ax.get_yaxis ().get_majorticklabels ()[-1:],
                visible=False)
        out['second_ratio_ax'] = second_ratio_ax

        fig.suptitle (title, size='large', weight='bold')

        return out

    def create_correlation_matrix_plot (self,
            set_spec,
            exprs=None,
            fignum=None,
            cut=None,
            eval_names={},
            ):
        """Create a correlation matrix plot.

        :type   set_spec: str or tuple
        :param  set_spec: See :meth:`Validator.get_key_wkey`

        :type   exprs: list
        :param  exprs: List of expressions to put on the axes (default:
            self.bdt.feature_names)

        :type   fignum: int
        :param  fignum: If given, create a figure with the given number using
            matplotlib.pyplot; otherwise use matplotlib.figure.Figure.

        :type   cut: str
        :param  cut: An expression for :meth:`Validator.eval` which returns an
            array of bools, where True means "include this event".

        :type   eval_names: dict
        :param  eval_names: Names to be passed to :meth:`Validator.eval` for
            cut evaluation.

        :return: The new :class:`matplotlib.figure.Figure`.

        """
        if exprs is None:
            exprs = self.bdt.feature_names

        fig_height = 7
        fig_width = 4./3 * fig_height
        fig_kwargs = {}

        if fignum is not None:
            Figure = plt.figure
            fig_kwargs['num'] = fignum
        else:
            Figure = mpl.figure.Figure

        fig_kwargs['figsize'] = fig_width, fig_height

        fig = Figure (**fig_kwargs)
        if fignum is not None:
            fig.clf ()
        else:
            mpl.backends.backend_agg.FigureCanvasAgg (fig)

        ax = fig.add_subplot (111)

        # begin code based on Andreas's program

        corr_all = []
        keys = []
        for key1 in exprs:
            corr = []
            keys.append (key1)
            for key2 in exprs:
                corr.append (self.get_correlation (
                    set_spec, key1, key2, cut=cut, eval_names=eval_names))
            corr_all.append (corr)

        nbins = len (keys)
        arr = np.arange (nbins)
        histdata = np.histogram2d (arr, arr, bins=(nbins,nbins))
        entries = histdata[0]
        entries = corr_all
        im1 = ax.imshow (entries, interpolation='nearest', origin='lower',
                vmin=-1, vmax=1)
        ax.set_xticks (arr)
        ax.set_yticks (arr)
        xlabels = ax.set_xticklabels (keys, rotation=90, fontsize=10)
        ylabels = ax.set_yticklabels (keys, fontsize=10)

        def on_draw (event):
            ybboxes = []
            for label in ylabels:
                bbox = label.get_window_extent ()
                bboxi = bbox.inverse_transformed (fig.transFigure)
                ybboxes.append (bboxi)
            ybbox = mpl.transforms.Bbox.union (ybboxes)
            if fig.subplotpars.left < ybbox.width - 0.01:
                leftmargin = 1.1 * ybbox.width
                fig.subplots_adjust (left=leftmargin)
                fig.canvas.draw ()

            xbboxes = []
            for label in xlabels:
                bbox = label.get_window_extent ()
                bboxi = bbox.inverse_transformed (fig.transFigure)
                xbboxes.append (bboxi)
            xbbox = mpl.transforms.Bbox.union (xbboxes)
            if fig.subplotpars.bottom < xbbox.height - 0.01:
                bottommargin = 1.1 * xbbox.height
                fig.subplots_adjust (bottom=bottommargin)
                fig.canvas.draw ()
            return False

        fig.canvas.mpl_connect ('draw_event', on_draw)

        leftmargin = fig.subplotpars.left
        pos = ax.get_position ()
        l, b, w, h = pos.bounds
        cax = fig.add_axes ([leftmargin, b+h+0.04, l+w-leftmargin, 0.04])
        fig.colorbar (im1, cax=cax, orientation='horizontal')

        return fig

    def create_correlation_ratio_matrix_plot (self,
            set_spec1, set_spec2,
            exprs=None,
            fignum=None,
            clog=False,
            cut=None,
            eval_names={},
            ):
        """Create a correlation matrix plot.

        :type   set_spec1: str or tuple
        :param  set_spec1: See :meth:`Validator.get_key_wkey`

        :type   set_spec2: str or tuple
        :param  set_spec2: See :meth:`Validator.get_key_wkey`

        :type   exprs: list
        :param  exprs: List of expressions to put on the axes (default:
            self.bdt.feature_names)

        :type   fignum: int
        :param  fignum: If given, create a figure with the given number using
            matplotlib.pyplot; otherwise use matplotlib.figure.Figure.

        :type   clog: bool
        :param  clog: Whether to use a log color scale (absolute values of
            ratios will be shown)

        :type   cut: str
        :param  cut: An expression for :meth:`Validator.eval` which returns an
            array of bools, where True means "include this event".

        :type   eval_names: dict
        :param  eval_names: Names to be passed to :meth:`Validator.eval` for
            cut evaluation.

        :return: The new :class:`matplotlib.figure.Figure`.

        """
        if exprs is None:
            exprs = self.bdt.feature_names

        fig_height = 7
        fig_width = 4./3 * fig_height
        fig_kwargs = {}

        if fignum is not None:
            Figure = plt.figure
            fig_kwargs['num'] = fignum
        else:
            Figure = mpl.figure.Figure

        fig_kwargs['figsize'] = fig_width, fig_height

        fig = Figure (**fig_kwargs)
        if fignum is not None:
            fig.clf ()
        else:
            mpl.backends.backend_agg.FigureCanvasAgg (fig)

        ax = fig.add_subplot (111)

        # begin code based on Andreas's program

        corr_all1 = []
        corr_all2 = []
        keys = []
        for key1 in exprs:
            corr1 = []
            corr2 = []
            keys.append (key1)
            for key2 in exprs:
                corr1.append (self.get_correlation (
                    set_spec1, key1, key2, cut=cut, eval_names=eval_names))
                corr2.append (self.get_correlation (
                    set_spec2, key1, key2, cut=cut, eval_names=eval_names))
            corr_all1.append (corr1)
            corr_all2.append (corr2)
        corr_all1 = np.array (corr_all1)
        corr_all2 = np.array (corr_all2)
        ratios = corr_all1 / corr_all2
        entries = np.abs (ratios)
        vmin = np.min (entries)
        vmax = np.max (entries)
        if clog:
            vmax = max (vmax, 1 / vmin)
            vmin = 1 / vmax

        nbins = len (keys)
        arr = np.arange (nbins)
        histdata = np.histogram2d (arr, arr, bins=(nbins,nbins))
        if clog:
            kw = dict (norm=mpl.colors.LogNorm (vmin=vmin, vmax=vmax))
        else:
            kw = dict (vmin=vmin, vmax=vmax)
        im1 = ax.imshow (entries, interpolation='nearest', origin='lower',
                **kw)
        ax.set_xticks (arr)
        ax.set_yticks (arr)
        xlabels = ax.set_xticklabels (keys, rotation=90, fontsize=10)
        ylabels = ax.set_yticklabels (keys, fontsize=10)
        #ylabels = ax.set_yticklabels (
        #        ['{0} ({1})'.format (k,a)
        #            for (k,a) in izip (keys,arr)], fontsize=10)

        def on_draw (event):
            ybboxes = []
            for label in ylabels:
                bbox = label.get_window_extent ()
                bboxi = bbox.inverse_transformed (fig.transFigure)
                ybboxes.append (bboxi)
            ybbox = mpl.transforms.Bbox.union (ybboxes)
            if fig.subplotpars.left < ybbox.width - 0.01:
                leftmargin = 1.1 * ybbox.width
                fig.subplots_adjust (left=leftmargin)
                fig.canvas.draw ()

            xbboxes = []
            for label in xlabels:
                bbox = label.get_window_extent ()
                bboxi = bbox.inverse_transformed (fig.transFigure)
                xbboxes.append (bboxi)
            xbbox = mpl.transforms.Bbox.union (xbboxes)
            if fig.subplotpars.bottom < xbbox.height - 0.01:
                bottommargin = 1.1 * xbbox.height
                fig.subplots_adjust (bottom=bottommargin)
                fig.canvas.draw ()
            return False

        fig.canvas.mpl_connect ('draw_event', on_draw)

        leftmargin = fig.subplotpars.left
        pos = ax.get_position ()
        l, b, w, h = pos.bounds
        cax = fig.add_axes ([leftmargin, b+h+0.04, l+w-leftmargin, 0.04])
        fig.colorbar (im1, cax=cax, orientation='horizontal')

        for key1, ratio_row, x in zip (exprs, ratios, arr):
            for key2, ratio, y in zip (exprs, ratio_row, arr):
                if ratio < 0:
                    ax.plot (x, y, 'w_', zorder=10, ms=10)
                    ax.plot (x, y, 'k_', zorder=11, ms=7)

        return fig

    def create_variable_pair_plot (self,
            set_spec, exprx, expry,
            bins=100,
            range=None,
            fignum=None,
            clog=False,
            cut=None,
            eval_names={},
            ):
        """Create a variable-variable 2D histogram.

        :type   set_spec: str or tuple
        :param  set_spec: See :meth:`Validator.get_key_wkey`

        :type   exprx: str
        :param  exprx: Expression to put on the x axis

        :type   expry: str
        :param  expry: Expression to put on the x axis

        :type   bins: int
        :param  bins: The number of bins to create [default: 100].

        :type   range: tuple of tuples of floats
        :param  range: If given, the x range and y range in the form
            ((xmin,xmax), (ymin,ymax))

        :type   fignum: int
        :param  fignum: If given, create a figure with the given number using
            matplotlib.pyplot; otherwise use matplotlib.figure.Figure.

        :type   clog: bool
        :param  clog: Whether to use a log color scale (absolute values of
            ratios will be shown)

        :type   cut: str
        :param  cut: An expression for :meth:`Validator.eval` which returns an
            array of bools, where True means "include this event".

        :type   eval_names: dict
        :param  eval_names: Names to be passed to :meth:`Validator.eval` for
            cut evaluation.

        """
        fig_height = 7
        fig_width = 4./3 * fig_height
        fig_kwargs = {}

        if fignum is not None:
            Figure = plt.figure
            fig_kwargs['num'] = fignum
        else:
            Figure = mpl.figure.Figure

        fig_kwargs['figsize'] = fig_width, fig_height

        fig = Figure (**fig_kwargs)
        if fignum is not None:
            fig.clf ()
        else:
            mpl.backends.backend_agg.FigureCanvasAgg (fig)

        ax = fig.add_subplot (111)
        x, w = self.get_values_weights (set_spec, exprx,
                cut=cut, eval_names=eval_names)
        y, w = self.get_values_weights (set_spec, expry,
                cut=cut, eval_names=eval_names)
        h, bx, by = np.histogram2d (
                x, y, weights=w, bins=bins, range=range, normed=True)
        if clog:
            H = np.ma.array (np.log10 (h.T))
            H.mask = h.T == h.T.min ()
        else:
            H = h.T
        Bx, By = np.meshgrid (bx, by)
        pc = ax.pcolormesh (Bx, By, H)
        ax.set_xlabel (exprx)
        ax.set_ylabel (expry)
        ax.set_xlim (bx.min (), bx.max ())
        ax.set_ylim (by.min (), by.max ())

        return dict (fig=fig, ax=ax)


__all__ = ['Validator']
