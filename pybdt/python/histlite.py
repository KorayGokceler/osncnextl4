# histlight.py


__doc__  = """Calculate and plot histograms easily.

Numerous solutions are possible and already exist for generating and plotting
histograms with matplotlib. This module aims to provide the minimal interface
needed to add this useful functionality to a matplotlib environment.

"""



import copy
import matplotlib as mpl
import numpy as np


## 1D stuff ---------------------------


class Binner :

    """Tool to generate :class:`Hist` instances."""

    def __init__ (self, bins=50, range=None):
        """Initialize a Binner.
        
        :type   bins: int or numpy.ndarray
        :param  bins: The bins to be passed to numpy.histogram().

        :type   range: tuple
        :param  range: The range to be passed to numpy.histogram().

        """
        self._bins = bins
        self._range = range

        if range is not None and np.sum (np.isfinite (range)) != 2:
            raise ValueError ('Histogram range must be finite.')


    @property
    def bins (self):
        """The bins to be passed to numpy.histogram()."""
        return self._bins

    @property
    def range (self):
        """The range to be passed to numpy.histogram()."""
        return self._range

    @property
    def kwargs (self):
        """The numpy.histogram() keyword arguments."""
        return dict (bins=self.bins, range=self.range)

    def hist (self, array, weights=None):
        """Create a :class:`Hist`."""
        if weights is None:
            weights = np.ones (len (array))
        idx = np.isfinite (array) * np.isfinite (weights)

        range = (self.range,) if self.range else None

        if idx.sum () == 0:
            ignored, bins = np.histogram ([1], weights=[0], bins=self.bins,
                    range=self.range)
            return Hist (
                    bins, np.zeros (len (bins) - 1), np.zeros (len (bins) - 1))

        result = np.histogramdd (array[idx], weights=weights[idx],
                bins=self.bins, range=range)
        values, bins = result[0], result[1][0]
        result = np.histogramdd (array[idx], weights=weights[idx]**2,
                bins=self.bins, range=range)
        errors = np.sqrt (result[0])

        return Hist (bins, values, errors)


class Line :

    """Base class for binned lines such as histograms."""

    def __init__ (self, bins, values, errors=None):
        """Initialize a Line.

        :type   bins: numpy.ndarray
        :param  bins: The bin edges.

        :type   values: numpy.ndarray
        :param  values: The bin values.

        :type   errors: numpy.ndarray
        :param  errors: The per-bin errors.
        """
        self.bins = np.asarray (bins)
        self.values = np.asarray (values)
        if errors is None:
            self.errors = np.zeros (len (values))
        else:
            self.errors = np.asarray (errors)

    def bins_match (a, b):
        """Check whether two Lines have matching bins.

        :type   a: :class:`Line`
        :param  a: The first object.
        
        :type   b: :class:`Line`
        :param  b: The second object.

        :return: Whether the bins match (bool).
        """
        return np.sum ((a.bins - b.bins)**2) == 0

    def __add__ (a, b):
        if isinstance (a, Line) and isinstance (b, Line):
            assert (a.bins_match (b))
            values = 1.0 * a.values + b.values
            errors = np.sqrt (a.errors**2 + b.errors**2)
            return a.__class__ (a.bins, values, errors)
        else:
            return a.__class__ (a.bins, a.values + b, errors)

    def __sub__ (a, b):
        if a.__class__ is b.__class__:
            assert (a.bins_match (b))
            values = 1.0 * a.values - b.values
            errors = np.sqrt (a.errors**2 - b.errors**2)
            return a.__class__ (a.bins, values, errors)
        else:
            return a.__class__ (a.bins, values + b, errors)

    def __mul__ (a, b):
        if isinstance (b, Line):
            assert (a.bins_match (b))
            values = a.values * b.values
            errors = values * np.sqrt (
                    (a.errors / a.values)**2 + (b.errors / b.values)**2)
            return a.__class__ (a.bins, values, errors)
        else:
            return a.__class__ (a.bins, b * a.values, abs (b) * a.errors)

    def __rmul__ (self, scalar):
        return self * scalar

    def __div__ (a, b):
        if isinstance (b, Line):
            assert (a.bins_match (b))
            values = a.values / b.values
            errors = values * np.sqrt (
                    (a.errors / a.values)**2 + (b.errors / b.values)**2)
            return a.__class__ (a.bins, values, errors)
        else:
            b = 1.0 * b
            return a.__class__ (a.bins, a.values / b, a.errors / b)

    __truediv__ = __div__

    @property
    def bins (self):
        """The bin boundaries."""
        return self._bins
    @bins.setter
    def bins (self, bins):
        self._bins = copy.deepcopy (bins)

    @property
    def bin_centers (self):
        """The bin centers."""
        return (self.bins[:-1] + self.bins[1:]) / 2.

    @property
    def values (self):
        """The bin values, or counts."""
        return self._values
    @values.setter
    def values (self, values):
        self._values = copy.deepcopy (values)

    @property
    def errors (self):
        """The bin value errors."""
        return self._errors
    @errors.setter
    def errors (self, errors):
        self._errors = copy.deepcopy (errors)

    def at (self, x):
        """The value of the line at x (number or array)."""
        try:
            len (x)
            arr = True
        except:
            arr = False
            x = [x]
        i = np.digitize (np.asarray (x), self.bins) - 1
        if arr:
            return self.values[i]
        else:
            return float (self.values[i])


class Hist (Line):

    """A histogram."""

    def __init__ (self, bins, values, errors=None):
        """Initialize a Hist.

        All arguments are passed directly to the :class:`Line`
        constructor.

        """
        Line.__init__ (self, bins, values, errors)


    @property
    def sum (self):
        """The sum of the bin values."""
        return self.values.sum ()

    @property
    def integral (self):
        """The integral of the histogram."""
        return np.sum (self.values * (np.diff (self.bins)))

    @property
    def sum_normed (self):
        """A copy of this Hist normalized so the bin counts sum to 1."""
        return self / self.sum

    @property
    def integral_normed (self):
        """A copy of this Hist normalized so the integral is 1."""
        return self / self.integral

    @property
    def cumulative_right (self):
        """The cumulative histogram, adding to the right."""
        # TODO: include proper errors
        return Line (self.bins, self.values.cumsum ())

    @property
    def cumulative_left (self):
        """The cumulative histogram, adding to the left."""
        # TODO: include proper errors
        return Line (self.bins,
                self.values[::-1].cumsum ()[::-1])
        #return Line (self.bins, self.sum - self.values.cumsum ())

    def efficiency (self, base_hist):
        """Get an efficiency plot for this Hist divided by base_hist.

        :type   base_hist: :class:`Hist`
        :param  base_hist: The base histogram, of which this one should be a
            subset.

        This method differs from __div__ in the way that errors are propagated.

        """
        keep = self
        orig = base_hist
        rej = orig - keep

        eff = keep / orig
        nkeep = keep.values
        nrej = rej.values
        eff.errors = np.sqrt (
                (nrej / (nkeep+nrej)**2 * keep.errors)**2
                + (-nkeep / (nkeep+nrej)**2 * rej.errors)**2 )
        return eff

    def sample (self, n):
        """Draw n random samples from the histogram."""
        y = self.bins
        x = np.r_[0, self.values.cumsum ()] / self.sum
        # interpolate inverse CDF
        out = np.interp (np.random.random (n), x, y)
        if n == 1:
            return out[0]
        else:
            return out.reshape ((n,))

    def rebin (self, bins, tol=1e-4):
        """Produce coarser binning.
        
        Each bin in bins should be contained in the existing bins, and the
        endpoints should match.  Tolerance for bin agreement is given as an
        absolute error by `tol`.
        """

        bins = np.copy (np.sort (bins))
        for (i, b) in enumerate (bins):
            misses = np.abs (b - self.bins)
            j = np.argmin (misses)
            closest = np.min (misses)
            if closest > tol:
                raise ValueError (
                        '{0} is not among current bin edges'.format (b))
            bins[i] = self.bins[j]
        if bins[0] != self.bins[0]:
            raise ValueError (
                    'binning startpoint should match ({0} vs {1})'.format (
                        bins[0], self.bins[0]))
        if bins[-1] != self.bins[-1]:
            raise ValueError (
                    'binning endpoint should match ({0} vs {1})'.format (
                        bins[-1], self.bins[-1]))

        n_newbins = len (bins) - 1
        newbin_indices = np.digitize (self.bins, bins)[:-1] - 1
        values = np.array ([
            np.sum (self.values[newbin_indices == i])
            for i in range (n_newbins)
            ])
        if self.errors is not None:
            errors = np.array ([
                np.sqrt (np.sum (self.errors[newbin_indices == i]**2))
                for i in range (n_newbins)
                ])
        else:
            errors = None
        return Hist (bins, values, errors)
        


class Style :

    """Simple style object for Lines."""

    def __init__ (self,
            line=True,
            markers=False,
            errorbars=False,
            errorcaps=False,
            **kwargs
            ):
        """Initialize a Style.

        :type   line: bool
        :param  line: Whether to draw a line.

        :type   markers: bool
        :param  markers: Whether to draw point markers.

        :type   errorbars: bool
        :param  errorbars: Whether to draw errorbars.

        :type   errorcaps: bool
        :param  errorcaps: Whether to draw error bar caps.

        All other keyword args are saved to be passed to
        matplotlib.axes.Axes.errorbar(). Note that drawstyle should not be
        specified; if line == True, then drawstyle='steps-post' will be used.

        """
        self._kwargs = {}
        self.line = True
        self.markers = False
        self.errorbars = False
        self.errorcaps = False
        self.update (
                line=line, markers=markers,
                errorbars=errorbars, errorcaps=errorcaps,
                **kwargs)

    def update (self, **kwargs):
        """Update the keyword args with the given values."""
        self.line = kwargs.pop ('line', self.line)
        self.markers = kwargs.pop ('markers', self.markers)
        self.errorbars = kwargs.pop ('errorbars', self.errorbars)
        self.errorcaps = kwargs.pop ('errorcaps', self.errorcaps)
        self._kwargs.update (copy.deepcopy (kwargs))
        if 'marker' in kwargs \
                and kwargs['marker'] not in ('none', 'None', None):
            self.markers = True

    def copy (self, **kwargs):
        """Get a copy of this Style, updating the given keyword args.

        All arguments accepted by the :class:`Style` constructor may be given,
        including line, markers, errorbars, errorcaps, and arbitrary matplotlib
        arguments.
        
        """
        out = copy.deepcopy (self)
        out.update (**kwargs)
        return out

    @property
    def line (self):
        """Whether to draw a line."""
        return self._line
    @line.setter
    def line (self, line):
        self._line = line

    @property
    def markers (self):
        """Whether to draw point markers."""
        return self._markers
    @markers.setter
    def markers (self, markers):
        self._markers = markers
        kwargs = self._kwargs
        if self.markers:
            if 'marker' not in kwargs \
                    or kwargs['marker'] in ('none', 'None', None):
                self._kwargs['marker'] = 'o'
        else:
            self._kwargs['marker'] = 'None'

    @property
    def errorbars (self):
        """Whether to draw error bars."""
        return self._errorbars
    @errorbars.setter
    def errorbars (self, errorbars):
        self._errorbars = errorbars
        if not self.errorbars:
            self._kwargs['elinewidth'] = np.finfo (float).tiny
        else:
            self._kwargs.pop ('elinewidth', None)

    @property
    def errorcaps (self):
        """Whether to draw error bar caps."""
        return self._errorcaps
    @errorcaps.setter
    def errorcaps (self, errorcaps):
        self._errorcaps = errorcaps
        if not self.errorcaps:
            self._kwargs['capsize'] = 0
        else:
            self._kwargs.pop ('capsize', None)

    @property
    def kwargs (self):
        """Keyword args for matplotlib.axes.Axes.errorbar()."""
        return copy.deepcopy (self._kwargs)


class Plotter :

    """Tool for plotting :class:`Line` objects."""

    def __init__ (self, axes,
            twin_axes=None,
            log=False,
            twin_log=None,
            expx=False,
            errormin=None):
        """Initialize a Plotter.

        :type   axes: matplotlib Axes
        :param  axes: The main axes on which to plot.

        :type   twin_axes: matplotlib Axes
        :param  twin_axes: The secondary-y axes, if already created with
            axes.twinx().

        :type   log: bool
        :param  log: Whether to use a log y scale on the main axes.

        :type   twin_log: bool
        :param  twin_log: Whether to use a log y scale on the twin x axes.
            (If not given, then same as log argument)

        :type   expx: bool
        :param  expx: If true, convert :math:`x` -> :math:`10^x`

        :type   errormin: float
        :param  errormin: The minimum value for the lower edge of errorbars
            (useful when log=True)

        """
        self._axes = axes
        self._twin_axes = twin_axes
        self._log = log
        if twin_log is None:
            self._twin_log = self.log
        else:
            self._twin_log = twin_log
        self._lines = np.empty (0, dtype=Line)
        self._line_styles = np.empty (0, dtype=dict)
        self._line_axes = np.empty (0, dtype=str)
        self._expx = bool (expx)
        self.mpl_lines = np.empty (0, dtype=object)
        self.labels = np.empty (0, dtype=str)
        temp_fig = mpl.figure.Figure ()
        self._legline_axes = temp_fig.add_subplot (111)

    @property
    def axes (self):
        """The matplotlib Axes."""
        return self._axes

    @property
    def log (self):
        """Whether to use a log y scale on the main axes."""
        return self._log

    @property
    def twin_log (self):
        """Whether to use a log y scale on the twin axes."""
        return self._twin_log

    @property
    def expx (self):
        """If true, convert :math:`x` -> :math:`10^x`"""
        return self._expx

    @property
    def twin_axes (self):
        """The matplotlib twinx Axes."""
        return self._twin_axes

    @property
    def lines (self):
        """The list of :class:`Lines` for this Plotter."""
        return self._lines

    @property
    def line_styles (self):
        """The list of keyword argument dicts for this Plotter."""
        return self._line_styles

    @property
    def line_axes (self):
        """The list of axes specifications for this Plotter (elements are
        'main' or 'twin')."""
        return self._line_axes


    def add (self, line_to_add,
            twin=False,
            style=None,
            **kwargs
            ):
        """Add a Line.

        :type   line_to_add: :class:`Line`
        :param  line_to_add: The line.

        :type   twin: bool
        :param  twin: Whether to use the secondary axes (default: use main axes)

        :type   style: :class:`Style`
        :param  style: The style for this line.

        If additional keyword arguments are given, then this line is plotted
        with a Style containing these extra keyword arguments.

        """
        self._lines = np.append (self.lines, line_to_add)
        if twin:
            line_axes = 'twin'
        else:
            line_axes = 'main'
        self._line_axes = np.append (self.line_axes, 'twin')
        if style:
            style = style.copy (**kwargs)
        else:
            style = Style (**kwargs)
        self._line_styles = np.append (self.line_styles, style)

        self._do_plot (line_to_add, style, line_axes)


    def _do_plot (self, line, style, axes_name):
        """Do the plotting."""
        if axes_name == 'main':
            axes = self.axes
            log = self.log
        else:
            if self.twin_axes is None:
                self._twin_axes = self.axes.twinx ()
            axes = self.twin_axes
            log = self.twin_log
        axes = self.axes if axes_name == 'main' else self.twin_axes
        prev_ymin, prev_ymax = axes.get_ylim ()
        x = line.bin_centers
        if self.expx:
            x = 10**x
        y = line.values
        yerr = line.errors
        kwargs = copy.deepcopy (style.kwargs)
        label = kwargs.get ('label', '')
        if style.line and 'label' in kwargs:
            del kwargs['label']
        if not style.errorbars:
            yerr = np.zeros_like (line.errors)
            kwargs['capsize'] = 0
        for k in ['ls', 'drawstyle']:
            if k in kwargs:
                del kwargs[k]
        kwargs['linestyle'] = 'none'
        line_count = 0
        mpl_line, mpl_errorcaps, mpl_errorbars = axes.errorbar (
                x, y, yerr,
                **kwargs)
        line_count += 1
        if style.line:
            bins = line.bins
            if self.expx:
                bins = 10**bins
            bx = bins
            by= np.r_[y, y[-1]]
            zorder = kwargs.pop ('zorder', 0)
            zorder -= .01
            color = mpl_line.get_color ()
            drawstyle= 'steps-post'
            line_kwargs = dict (
                    drawstyle=drawstyle, color=color, zorder=zorder)
            def keep (key):
                if key in style.kwargs:
                    line_kwargs[key] = style.kwargs[key]

            keep ('lw'), keep ('linewidth')
            keep ('ls'), keep ('linestyle')
            keep ('alpha')
            line_kwargs['label'] = label
            axes.plot (bx, by, **line_kwargs)
            keep ('marker')
            # this last bit makes the line that's used in the legend.
            mpl_line = self._legline_axes.errorbar (
                    [0, 1], [0, 1], [.5, .5], **line_kwargs)[0]
            # the hack below isn't strictly necessary, and it might not work on
            # all versions of matplotlib.  in matplotlib-0.99.1.1, this causes
            # the default color cycle to be preserved.
            try:
                axes._get_lines.count -= (line_count)
            except:
                pass

        if log:
            axes.set_yscale ('log')
            if np.sum (y>0) > 0:
                # don't let errorbars make the scale crazy
                new_ymin, new_ymax = axes.get_ylim ()
                min_accepted_ymin = min (prev_ymin, .1 * np.min (y[y>0]))
                final_ymin = max (new_ymin, min_accepted_ymin)
                axes.set_ylim (ymin=final_ymin)

        self.mpl_lines = np.append (self.mpl_lines, mpl_line)
        self.labels = np.append (self.labels, label)
        
        if self.expx:
            self.axes.set_xscale ('log', nonposy='clip')
            if self.twin_axes is not None:
                self.twin_axes.set_xscale ('log')


    def legend (self, **kwargs):
        """Draw the legend.

        All keyword arguments are passed to axes.legend().
        """
        axes = self.twin_axes or self.axes
        self.mpl_legend = axes.legend (self.mpl_lines, self.labels, **kwargs)


    def finish (self, legend=None):
        """Deprecated since plotting was moved to Plotter._do_plot().

        """
        if legend is True:
            kwargs = {}
        else:
            kwargs = legend
        if legend:
            self.legend (**kwargs)


## 2D stuff ---------------------------


class Binner2D :

    """Tool to generate :class:`Hist2D` instances."""

    def __init__ (self, bins=50, range=None):
        """Initialize a Binner.

        :type   bins: same as numpy.histogram2d()
        :param  bins: The bins to be passed to numpy.histogram2d().

        :type   range: same as numpy.histogram2d()
        :param  range: The range to be passed to numpy.histogram2d()

        """
        self._bins = bins
        self._range = range

    @property
    def bins (self):
        """The bins to be passed to numpy.histogram2d()."""
        return self._bins

    @property
    def range (self):
        """The range to be passed to numpy.histogram2d()."""
        return self._range

    @property
    def kwargs (self):
        """The numpy.histogram2d() keyword arguments."""
        return dict (bins=self.bins, range=self.range)

    def hist (self, x, y, weights=None):
        """Create a :class:`Hist2D`."""
        if weights is None:
            weights = np.ones (len (x))
        if not (len (x) == len (y) == len (weights)):
            raise ValueError ('x, y and weights must have equal lengths')

        idx = np.isfinite (x) * np.isfinite (y) * np.isfinite (weights)

        if idx.sum () == 0:
            pass

        ix = x[idx]
        iy = y[idx]
        iweights = weights[idx]

        values, xbins, ybins = np.histogram2d (
                ix, iy, weights=iweights, **self.kwargs)
        esquared, xignored, yignored = np.histogram2d (
                ix, iy, weights=iweights**2, **self.kwargs)
        errors = np.sqrt (esquared)

        return Hist2D (xbins, ybins, values, errors)


class Surface2D :

    """Base class for binned surfaces such as 2D histograms."""

    def __init__ (self, xbins, ybins, values, errors=None):
        """Initialize a Surface2D.

        :type   xbins: numpy.ndarray
        :param  xbins: The x bin edges.

        :type   ybins: numpy.ndarray
        :param  ybins: The y bin edges.

        :type   values: numpy.ndarray
        :param  values: The bin values.

        :type   errors: numpy.ndarray
        :param  errors: The per-bin errors.
        """
        self.xbins = np.asarray (xbins)
        self.ybins = np.asarray (ybins)
        self.values = np.asarray (values)
        if errors is None:
            self.errors = None
        else:
            self.errors = np.asarray (errors)

    def bins_match (a, b):
        """Check whether two Surface2Ds have matching bins.

        :type   a: :class:`Surface2D`
        :param  a: The first object.
        
        :type   b: :class:`Surface2D`
        :param  b: The second object.

        :return: Whether the bins match (bool).
        """
        return 0 == (
                np.sum ((a.xbins - b.xbins)**2)
                + np.sum ((a.ybins - b.ybins)**2) )

    def __add__ (a, b):
        if isinstance (a, Surface2D) and isinstance (b, Surface2D):
            assert (a.bins_match (b))
            values = 1.0 * a.values + b.values
            if a.errors is not None and b.errors is not None:
                errors = np.sqrt (a.errors**2 + b.errors**2)
            else:
                errors = None
            return a.__class__ (a.xbins, a.ybins, values, errors)
        else:
            return a.__class__ (a.xbins, a.ybins, a.values + b, errors)

    def __sub__ (a, b):
        if a.__class__ is b.__class__:
            assert (a.bins_match (b))
            values = 1.0 * a.values - b.values
            if a.errors is not None and b.errors is not None:
                errors = np.sqrt (a.errors**2 - b.errors**2)
            else:
                errors = None
            return a.__class__ (a.xbins, a.ybins, values, errors)
        else:
            return a.__class__ (a.xbins, a.ybins, a.values + b, errors)

    def __mul__ (a, b):
        if isinstance (b, Surface2D):
            assert (a.bins_match (b))
            values = a.values * b.values
            if a.errors is not None and b.errors is not None:
                errors = values * np.sqrt (
                        (a.errors / a.values)**2 + (b.errors / b.values)**2)
            else:
                errors = None
            return Surface2D (a.xbins, a.ybins, values, errors)
        else:
            return a.__class__ (a.xbins, a.ybins,
                    b * a.values, abs (b) * a.errors)

    def __rmul__ (self, scalar):
        return self * scalar

    def __div__ (a, b):
        if isinstance (b, Surface2D):
            assert (a.bins_match (b))
            values = a.values / b.values
            if a.errors is not None and b.errors is not None:
                errors = values * np.sqrt (
                        (a.errors / a.values)**2 + (b.errors / b.values)**2)
            else:
                errors = None
            return Surface2D (a.xbins, a.ybins, values, errors)
        else:
            b = 1.0 * b
            return a.__class__ (a.xbins, a.ybins, a.values / b, a.errors / b)

    __truediv__ = __div__

    @property
    def xbins (self):
        """The x bin boundaries."""
        return self._xbins
    @xbins.setter
    def xbins (self, xbins):
        self._xbins = copy.deepcopy (xbins)

    @property
    def ybins (self):
        """The y bin boundaries."""
        return self._ybins
    @ybins.setter
    def ybins (self, ybins):
        self._ybins = copy.deepcopy (ybins)

    @property
    def values (self):
        """The bin values, or counts."""
        return self._values
    @values.setter
    def values (self, values):
        self._values = copy.deepcopy (values)

    @property
    def errors (self):
        """The bin value errors."""
        return self._errors
    @errors.setter
    def errors (self, errors):
        self._errors = copy.deepcopy (errors)


class Hist2D (Surface2D):

    """A 2D histogram."""

    def __init__ (self, xbins, ybins, values, errors=None):
        """Initialize a Hist2D.

        All arguments are passed directly to the :class:`Surface2D`
        constructor.
        
        """
        Surface2D.__init__ (self, xbins, ybins, values, errors)


    @property
    def sum (self):
        """The sum of the bin values."""
        return self.values.sum ()

    @property
    def integral (self):
        """The integral of the histogram.

        This function assumes equal bin spacing.
        """
        dx = self.xbins[1] - self.xbins[0]
        dy = self.ybins[1] - self.ybins[0]
        return self.sum * (dx * dy)

    @property
    def rowsums (self):
        """The sums of each row (adding along x)."""
        return self.values.sum (axis=0)

    @property
    def rowintegrals (self):
        """The sums of each row (adding along x)."""
        dx = self.xbins[1] - self.xbins[0]
        return self.rowsums * dx

    @property
    def colsums (self):
        """The sums of each column (adding along y)."""
        return self.values.sum (axis=1)

    @property
    def colintegrals (self):
        """The sums of each row (adding along x)."""
        dy = self.ybins[1] - self.ybins[0]
        return self.colsums * dy

    @property
    def sum_normed (self):
        """A copy of this Hist2D normalized so the bin counts sum to 1."""
        norm = self.sum
        return Hist2D (
                self.xbins, self.ybins,
                self.values / norm,
                self.errors / norm)

    @property
    def rowsums_normed (self):
        """A copy of this Hist2D normalized so each column sums to 1."""
        norms = self.rowsums
        values = self.values / norms
        if self.errors is not None:
            errors = self.errors / norms
        else:
            errors = None
        return Hist2D (
                self.xbins, self.ybins,
                values, errors)

    @property
    def colsums_normed (self):
        """A copy of this Hist2D normalized so each column sums to 1."""
        norms = self.colsums
        values = (self.values.T / norms).T
        if self.errors is not None:
            errors = (self.errors.T / norms).T
        else:
            errors = None
        return Hist2D (
                self.xbins, self.ybins,
                values, errors)

    @property
    def integral_normed (self):
        """A copy of this Hist2D normalized so the bin counts sum to 1."""
        norm = self.integral
        return Hist2D (
                self.xbins, self.ybins,
                self.values / norm,
                self.errors / norm)

    @property
    def rowintegrals_normed (self):
        """A copy of this Hist2D normalized so each row sums to 1."""
        norms = self.rowintegrals
        values = self.values / norms
        if self.errors is not None:
            errors = self.errors / norms
        else:
            errors = None
        return Hist2D (
                self.xbins, self.ybins,
                values, errors)

    @property
    def colintegrals_normed (self):
        """A copy of this Hist2D normalized so each column sums to 1."""
        norms = self.colintegrals
        values = (self.values.T / norms).T
        if self.errors is not None:
            errors = (self.errors.T / norms).T
        else:
            errors = None
        return Hist2D (
                self.xbins, self.ybins,
                values, errors)

    @property
    def cumulative_right (self):
        return self.rowcumulative_right.colcumulative_right

    @property
    def cumulative_left (self):
        return self.rowcumulative_left.colcumulative_left

    @property
    def rowcumulative_right (self):
        return Hist2D (self.xbins, self.ybins, self.values.cumsum (axis=0))

    @property
    def colcumulative_right (self):
        return Hist2D (self.xbins, self.ybins, self.values.cumsum (axis=1))

    @property
    def rowcumulative_left (self):
        return Hist2D (self.xbins, self.ybins, 
                self.values[::-1,::-1].cumsum (axis=0)[::-1,::-1])

    @property
    def colcumulative_left (self):
        return Hist2D (self.xbins, self.ybins, 
                self.values[::-1,::-1].cumsum (axis=1).T[::-1,::-1].T)

    @property
    def xhist (self):
        """Flatten rows to get a 1D Hist."""
        bins = self.xbins
        values = self.colsums
        if self.errors is not None:
            errors = np.sqrt (np.sum (self.errors**2, axis=1))
        else:
            errors = None
        return Hist (bins, values, errors)

    @property
    def xhists (self):
        """Give rows as an array of 1D Hists."""
        if self.errors is None:
            all_errors = np.array (len (self.values) * [None])
        else:
            all_errors = self.errors
        hists = np.array ([Hist (self.xbins, values, errors)
                for (values, errors)
                in zip (self.values.T, all_errors.T)])
        return hists

    @property
    def yhist (self):
        """Flatten columns to get a 1D Hist."""
        bins = self.ybins
        values = self.rowsums
        if self.errors is not None:
            errors = np.sqrt (np.sum (self.errors**2, axis=0))
        else:
            errors = None
        return Hist (bins, values, errors)
        
    @property
    def yhists (self):
        """Give columns as an array of 1D Hists."""
        if self.errors is None:
            all_errors = np.array (len (self.values) * [None])
        else:
            all_errors = self.errors
        hists = np.array ([Hist (self.ybins, values, errors)
                for (values, errors)
                in zip (self.values, all_errors)])
        return hists


    def efficiency (self, base_hist):
        """Get an efficiency plot for this Hist divided by base_hist.

        :type   base_hist: :class:`Hist`
        :param  base_hist: The base histogram, of which this one should be a
            subset.

        This method differs from __div__ in the way that errors are propagated.

        """
        keep = self
        orig = base_hist
        rej = orig - keep

        eff = keep / orig
        nkeep = keep.values
        nrej = rej.values
        eff.errors = np.sqrt (
                (nrej / (nkeep+nrej)**2 * keep.errors)**2
                + (-nkeep / (nkeep+nrej)**2 * rej.errors)**2 )
        return eff


def function_to_surface (x, y, func, hist=False):
    """Return a Surface2D evaluating ``func`` on the ``x``, ``y`` grid.

    :type   x: array-like
    :param  x: Evenly spaced x values.

    :param  y: array-like
    :param  y: Evenly spaced y values.

    :type   func: function
    :param  func: Function to evaluate at each x,y point.

    :type   hist: bool
    :param  hist: If true, return a Hist2D instead of a Surface2D.

    :return: A Surface2D or Hist2D where the bin values are the function values
        at each point.

    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    xbins = np.r_[x - dx/2., x[-1] + dx/2.]
    ybins = np.r_[y - dy/2., y[-1] + dy/2.]
    values = np.vectorize (func) (*np.meshgrid (x, y)).T
    if hist:
        return Hist2D (xbins, ybins, values)
    else:
        return Surface2D (xbins, ybins, values)


def plot_surface (axes, surface,
        log=False,
        expx=False,
        expy=False,
        cbar=False,
        **kwargs
        ):
    """Plot a :class:`Surface2D`.

    :type   axes: matplotlib Axes
    :param  axes: The main axes on which to plot.

    :type   surface: :class:`Surface2D`
    :param  surface: The histogram to plot.

    :type   log: bool
    :param  log: Whether to use a log color scale

    :type   expx: bool
    :param  expx: If true, convert :math:`x` -> :math:`10^x`

    :type   expy: bool
    :param  expy: If true, convert :math:`y` -> :math:`10^y`

    :type   cbar: bool
    :param  cbar: If true, draw colorbar.

    :type   zmin: float
    :param  zmin: Minimum value to plot with color; bins below the minimum
        value will be white.

    Other keyword arguments are passed to axes.pcolormesh().

    :return If cbar, a dict containing a matplotlib.collection.QuadMesh and a
    matplotlib.colorbar.Colorbar as values; otherwise, just the QuadMesh.

    """
    plotvalues = surface.values.T
    zmin = kwargs.pop ('zmin', None)
    if log:
        if zmin is None:
            zmin = 0
        H = np.ma.array (plotvalues)
    else:
        H = np.ma.array (plotvalues)
    H.mask += ~np.isfinite (plotvalues)
    if zmin is not None:
        H.mask += (plotvalues <= zmin)
    if expx:
        xbins = 10**surface.xbins
        axes.set_xscale ('log')
    else:
        xbins = surface.xbins
    if expy:
        ybins = 10**surface.ybins
        axes.set_yscale ('log')
    else:
        ybins = surface.ybins
    Bx, By = np.meshgrid (xbins, ybins)
    if log:
        vmin = kwargs.pop ('vmin', None)
        vmax = kwargs.pop ('vmax', None)
        kwargs['norm'] = mpl.colors.LogNorm (vmin, vmax)
    pc = axes.pcolormesh (Bx, By, H, **kwargs)
    axes.set_xlim (xbins.min (), xbins.max ())
    axes.set_ylim (ybins.min (), ybins.max ())
    if cbar:
        if log:
            cb = axes.figure.colorbar (pc,
                    format=mpl.ticker.LogFormatterMathtext ())
        else:
            cb = axes.figure.colorbar (pc)
        return dict (colormesh=pc, colorbar=cb)
    else:
        return pc


def plot_surface_lines (axes, surface, varname,
        horizontal='x',
        log=False,
        expx=False,
        fmt='.3f',
        cmap='jet'):

    if surface.errors is None:
        all_errors = np.array (len (surface.values) * [None])
    else:
        all_errors = surface.errors
    if horizontal == 'x':
        bins = surface.ybins
        bin_edges = np.array (list(zip (bins[:-1], bins[1:])))
        hists = [Hist (surface.xbins, values, errors)
                for (values, errors)
                in zip (surface.values.T, all_errors.T)]
    elif horizontal == 'y':
        bins = surface.xbins
        bin_edges = np.array (list(zip (bins[:-1], bins[1:])))
        hists = [Hist (surface.ybins, values, errors)
                for (values, errors)
                in zip (surface.values, all_errors)]
    plotter = Plotter (axes, log=log, expx=expx)
    cm = mpl.cm.get_cmap (cmap)
    n_hists = len (hists)
    for n, (hist, bin_edge) in enumerate (zip (hists, bin_edges)):
        plotter.add (hist, color=cm ((n + 1) / (n_hists + 1)),
                label='{0} < {1} < {2}'.format (
                    format (bin_edge[0], fmt),
                    varname,
                    format (bin_edge[1], fmt)))
    return plotter
