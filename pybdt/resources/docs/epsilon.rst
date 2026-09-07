.. _epsilon:

Global Epsilon Reference
========================

The SAMME.R algorithm [1] uses leaf purity information to calculate
per-event scores.  The procedure takes the geometric mean over the purity of
each leaf node hit by an event.  Leaves with signal or background purity of
zero or very nearly zero can cause the entire geometric mean to go to zero,
guaranteeing a final score of -1 (maximally background-like).  As a result,
the purity must be clipped to the open interval (0, 1) by enforcing a
floating point offset from exactly 0 or 1.  By default, this value is the
standard C++ ``std::numeric_limits<double>::epsilon()``.  However, it may be
useful to modify this threshold globally, particularly for compatibility
with classifiers imported from older versions of scikit-learn.  The
following functions facilitate tuning this threshold.

[1] J. Zhu, H. Zou, S. Rosset, T. Hastie. “Multi-class AdaBoost”, 2009.

    **Functions:**

    .. autofunction:: pybdt.ml.set_epsilon
       :noindex:
    .. autofunction:: pybdt.ml.get_epsilon
       :noindex:
