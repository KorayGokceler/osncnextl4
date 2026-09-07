.. _common_pitfalls:

Common Pitfalls
===============

Some issues that do not arrise in the :ref:`man_example` are common in
IceCube work.  Here are some I have been asked about a few times.


Outliers
--------

Variables with extreme outlier values for some events are usually not useful
for training.  This is because at each node, variables are histogrammed, and
extreme outliers will cause the bulk of the distribution for both signal and
background to share a single bin.  There are two ways to ameliorate this
problem:

-   Apply a (usually very loose) preselection cut that restricts values to a
    reasonable range.
-   Tell the training script to use nonlinear binning so that cuts within
    the reasonable range are automatically tested.  This option,
    ``--nonlinear-cuts`` or ``-L``, creates bins of equal total
    signal+background statistics rather than equal spacing in the variable.

The former requires slightly deeper understanding of your variables, which
is usually a good thing.  Often extreme outliers should be excluded for good
physics reasons.  Also, nonlinear binning is slower.  Nevertheless,
nonlinear binning may give slightly better results just because it accounts
better for the dynamic range of each variable.


Energy Proxies
--------------

Sometimes, an energy related variable seems like it should be useful, at
least in some parts of the parameter space, and yet variable importance
measures show that it is rarely or never used.  Like in the case of extreme
outliers, this can be addressed by using ``--nonlinear-cuts``.  However, a
simpler and physically well-motivated approach is to store a new variable
like ``log_energy``.  After all, in IceCube we almost always plot energy on
a log axis.  By taking the logarithm (typically ``log10`` but any log will
work) to pybdt, you give it access to the information in the same way it
makes most sense to us.  The dynamic range is clearly visible, and
reasonable cuts can be tested.


X server unavailable
--------------------

While using :py:mod:`pybdt.validate` on a machine with no X server, you may stumble
onto a cryptic Traceback that ends with something like ``_tkinter.TclError:
couldn't connect to display``.  This is a sign that something (possibly
:py:mod:`pybdt.validate`) imported ``matplotlib`` without specifying an X-less
backend.  Then, when you try to make a plot, matplotlib tries to open a
window, and it crashes.  The simple solution is to find the entry point to
your code and then make sure that the first two lines are ::

    import matplotlib
    matplotlib.use ('Agg')

Any subsequent matplotlib import will then be forced to use the X-less
backend, and the crash should be resolved.
