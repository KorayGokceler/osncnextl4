.. _man_validator_usage:

Validator Usage
===============

Once a :class:`pybdt.validate.Validator` is configured, it is easy to
generate most of the classification-related plots needed for an analysis.
On this page, we go over these plotting capabilities.


Summary Plots
-------------

:class:`pybdt.validate.Validator` includes a powerful plotting method,
:class:`pybdt.validate.Validator.create_plot()`, which can be used for
generating a summary of the overall classifier performance.  Typically the
first plot of interest is the score distribution for the testing and
training samples.  In the :ref:`ABC example <man_example>`, this is done
like so (slightly edited for brevity)::

    lines = ['test_data', 'test_sig', 'bg', 'total_mc']
    objs = v.create_plot ('scores', 'dist',
        lines,
        bins=100, range=(-1, .5),
        dual=True,
        data_mc=True,
        xlabel='BDT score',
        left_ylabel='Hz per bin',
        title='BDT score distributions',
        linear_kwargs=dict (
            legend=dict (loc='best')
        ),
    )
    objs['second_main_ax'].set_ylim (ymin=1e-6)
    objs['fig'].savefig ('output/dist_vs_bdt.png')

Let's unpack this example.  First, we specify the datasets we want to plots
as ``lines``.  Then we create the plot.  The first argument is an expression
that can be evaluated by :attr:`pybdt.validate.Validator.eval()` -- in this
case, the per-event classifier scores.  The second argument is either
``'dist'`` (a simple histogram), ``'rate'`` (a histogram summed
cumulatively-to-the-left), or ``'eff'`` (like ``'rate'``, but divided by the
leftmost point to give an efficiency curve).  The ``bins`` and ``range``
arguments specify the histogram properties.  ``dual=True`` means that a two
panel plot will be produced with a linear vertical scale on the left and a
log vertical scale on the right (by default, a single linear-vertical panel
will be produced, but if ``log=True`` is set, a single log-vertical panel
will be produced).  ``data_mc=True`` means that data/MC ratio plots will be
included under the main panels.  ``xlabel`` specifies the horizontal axis
label for both panels.  ``left_ylabel`` specifies the left-vertical axis
label for both main panels (we'll revisit the right-vertical axis soon).
``title`` gives an overall figure title.  ``linear_kwargs`` is used to pass
extra arguments specifically to the left (linear-vertical) plot -- in this
case, a legend location.

The returned value, ``objs``, is a dict with the following keys::

    ['fig', 'first_main_ax', 'second_main_ax', 'first_dm_ax', 'second_dm_ax']

The first is the matplotlib figure itself.  The next two are the main
matplotlib axes.  The last two are the data/MC ratio plot matplotlib axes.
In the example, we use these to tweak the vertical axis range and then to
save the figure, but any matplotlib customizations are available here.

In the example script pybdt/resources/examples/validate_sample_bdt.py, the
BDT score distribution is plotted this way; the overall event rate and the
cut efficiency as a function of BDT score cut are also plotted in a similar
way, changing little more than the second argument to
``create_plot()``.


Overtraining Check
------------------

The simplest way to check for overtraining is to compare the classifier
performance for the samples used for training against independent testing
samples.  In the :ref:`ABC example <man_example>`, this plot is made like
so::

    objs = v.create_overtrain_check_plot (
            'train_sig', 'test_sig',
            'train_data', 'test_data',
            left_ylabel='relative abundance (background)',
            right_ylabel='relative abundance (signal)',
            legend_side='left',
            legend=dict (loc='upper left'),
            )

Here, the first two arguments are the training and testing signal sample
specifications.  The next two are the training and testing background sample
specifications.  Then the left and right vertical axis labels are given.
Finally, in the last two keyword arguments we request that the legend be
placed on the left (linear) panel, and that the legend be placed in the
upper left of that panel.

The resulting plot shows the training and testing, signal and background
distributions (four distributions total) on a linear vertical scale
(left panel) and log vertical scale (right panel).  It also shows the
testing / training ratio below these main panels.  Finally, in the legend,
it shows the the Kolmogorov-Smirnov p-value when the testing and training
datasets are compared.  A small p-value suggests that the distributions
differ significantly.

A more rigorous overtraining test would repeat this process for multiple
(possibly overlapping) testing/training dataset splits.  However, for
IceCube, we typically are satisfied if performance is consistent for a
single testing/training split.


Other Plotting Features
-----------------------

In :ref:`Dataset weighting <man_validator_setup>`, we discussed the
possibility of configuring multiple weightings for a single ensemble of
events (as is commonly used for, e.g., IceCube neutrino simulation).  To
facilitate the use of alternative weightings, there are two ways to specify
dataset+weighting combinations for plotting.  In the ABC example, only the
simplest is needed: give the dataset identifier, and the
``'default'`` weighting will be used.  If a non-default weighting is
desired, it can be specified as a tuple: ``(dataset, weighting)``.  Here is
an example from a real analysis::

    
    x = v.create_plot ('scores', 'dist',
        ['nugen', 'corsika', 'total_mc', 'test_exp', ],
        [('test_nugen_wr', 'E2')],
        bins=bins,
        range=(-1,1),
        dual=True,
        xlabel='BDT score',
        left_ylabel='Hz per bin',
        right_ylabel='relative abundance (signal)',
        data_mc=True,
        log_kwargs=dict (legend=dict (loc='lower left')),
        title='BDT Score Distribution',
        )

The salient featurehere is the fourth positional argument to
``create_plot()``; this gives one or more dataset+weighting specification,
or ``set_spec``, for a testing sample of well-reconstructed
neutrino-generator events weighted to an :math:`E^{-2}` spectrum.
Additional dataset specifications in this argument are plotted against the
right-vertical axis.  In this analysis, the overall normalization of the
training and testing signal samples was arbitrary, so we use the
``right_ylabel`` keyword argument to give an appropriate right-vertical axis
label.


:attr:`pybdt.validate.Validator.create_plot()` can also be used to create
other variable distributions.  For example, in the ABC example, the ``a``
distribution after a BDT score cut of ``score > cut_level`` can be obtained
like so::

    objs = v.create_plot ('a', 'dist',                             
        lines,                                                      
        bins=50,                                                    
        dual=True,                                                  
        data_mc=True,                                               
        left_ylabel='Hz per bin',                                   
        xlabel=name,                                                
        log_kwargs=dict (                                           
            legend=dict (loc='best')                                
            ),                                                      
        title='{0} | bdt score > {1:.3f}'.format (name, cut_level), 
        cut='scores > {0}'.format (cut_level)                       
        )                                                           

Here, the ``cut`` argument is used to specify a cut that should be applied
to every dataset prior to generating the plot.  This mechanism allows the
creation of similar plots for several or all parameters at multiple cut
levels with limited code repetition.

Finally, the Validator can produce variable correlation matrix plots.  For
example, a color-coded correlation matrix plot for the training signal
sample can be obtained simply with::

    fig = v.create_correlation_matrix_plot ('train_sig')
    fig.savefig ('output/correlation_matrix-signal.png')


See pybdt/resources/examples/validate_sample_bdt.py for more example code;
see :class:`pybdt.validate.Validate` for other Validator capabilities.  For
a long, but possibly instructive, real-world example, see the
the `ml_plot() <http://code.icecube.wisc.edu/projects/icecube/browser/IceCube/sandbox/richman/grb/scripts/trunk/ic79/ic79_north_numu.py#L598>`_
function from the IC79 northern :math:`\nu_\mu` analysis.
