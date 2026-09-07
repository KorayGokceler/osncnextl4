.. _man_inspection:

Tree and Forest Inspection
==========================

One of the design goals of pybdt is to make it relatively straightforward
for users to inspect the classifier.  On this page we discuss some features
that make this easy.


Direct Acces
------------

The most basic way to learn about a trained classifier is to check its
properties in `IPython <http://ipython.org/>`_.  If a :class:`BDTModel` is
loaded as ``bdt``, then the number of constituent :class:`DTModel` s is
``bdt.n_dtmodels`` (an ``int``).  The per-tree relative weights are given by
``bdt.alphas`` (numpy array with dtype ``float``).  The :class:`DTModel` s
themselves are stored in ``bdt.dtmodels`` (numpy array with dtype
``object``).  The cut parameters used by the classifier are
``bdt.feature_names`` (list of ``string`` s).

Now, suppose we store the first tree as ``dt = bdt.dtmodels[0]``.  The root
:class:`DTNode` is ``node = dt.root``; you can jump from a node to its
children with :attr:`DTNode.left` and :attr:`DTNode.left` (until you reach a
leaf node, for which both children are ``None``).  Split nodes apply a cut
of the form ``node.feature_name`` < ``node.feature_val``; passing events
descend left and failing events descend right.  Other details are also
available -- see the :class:`DTNode` reference.


Visualization
-------------

pybdt includes a module :mod:`pybdt.viz` for easily visualizing trees.
For a text-only printout, we can say::

    from pybdt import viz
    print viz.dtmodel_to_text (dt)

For a graphical visualization, we can instead say::

    pic = viz.dtmodel_to_graphviz (dt)
    pic.write_png (filename + '.png')
    pic.write_pdf (filename + '.pdf')

and so forth -- many ``write_*`` methods are provided by graphviz.  This
feature requires the pydot interface to graphviz to be installed.  On
ubuntu, it's provided by the ``python-pydot`` package.


Variable Importance
-------------------

The features described above make it reasonably easy to dig into the details
of individual trees, and the plotting facilities
(:doc:`man_validator_usage`) offer a number of options for quantifying the
overall classifier performance.  However, one question remains to be
answered: to what extent does each feature participate in the classifier?
This can be addressed with the following calls::

    print viz.variable_importance_to_text (bdt.variable_importance (True, True)) #1 
    print viz.variable_importance_to_text (bdt.variable_importance (False, False)) #2 
    print viz.variable_importance_to_text (bdt.variable_importance (True, False)) #3 
    print viz.variable_importance_to_text (bdt.variable_importance (False, True)) #4 

:meth:`pybdt.ml.BDTModel.variable_importance` produces a ``dict`` with
string keys (the variable names) and floating point values (the relative
importance, between 0 and 1).  :func:`pybdt.viz.variable_importance_to_text`
converts this ``dict`` to a string consisting of an easily-readable table,
sorted by descending variable importance.

``variable_importance()`` measures the importance of variables by counting
the split nodes using each feature.  The first agrument tells whether the
count should be weighted by the separation gain (``node.sep_gain``) achieved
by each split.  The second argument tells whether the count should be
weighted by the overall weight (``alpha``) of the tree in which each split
occurs.

Here is the variable importance for the classifier in the :ref:`ABC example
<man_example>` -- first with both weightings enabled, and then with both
disabled::

    print viz.variable_importance_to_text (bdt.variable_importance (True, True))
    1. b : 0.556785
    2. c : 0.259359
    3. a : 0.183856

    print viz.variable_importance_to_text (bdt.variable_importance (False, False))
    1. c : 0.387212
    2. b : 0.349272
    3. a : 0.263517
        
In general, enabling weighting causes splits in earlier trees and splits
closer to the roots of trees to be counted more strongly, relative to when
weighting is disabled.  Thus the weighted variable importance can be
interpreted as a measure that is biased in favor of those variables
responsible for the classification of the bulk of events.  The unweighted
variable importance is, by comparison, more fair to variables used in later
trees or closer to leaf nodes; it can be interpreted as a measure of which
variables are responsible for classifying the trickiest events.  It is less
clear how the ``(True, False)`` and ``(False, True)`` variable importance
weightings should be interpreted, but the former will have more of a bias
towards the roots of trees and the latter will have more of a bias towards
earlier trees in the forest.
