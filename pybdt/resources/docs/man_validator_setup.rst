.. _man_validator_setup:

Validator Setup
===============

Working with machine learning can be broadly divided into 3 phases:
training, testing, and application. On this page we begin to discuss testing
with pybdt.

Testing and application are related in that the performance of a classifier
is first tested by applying it to event ensembles of known classes.  Scores
are found not only for the training samples but also for separate testing
samples which were set aside prior to training.  We can confirm that
overtraining has been sufficiently suppressed by comparing the performance
on the training and testing samples.  The overall quality of the classifier
can be quantified in terms of signal to noise ratio after applying a cut on
the classifier output.  If the training background sample was actually a
background-dominated experimental dataset, we can also evaluate the data/MC
agreement in the testing samples.

pybdt provides a class, :class:`pybdt.validate.Validator`, for performing
some of the most common classifier tests.  The recommended usage is to write
two scripts or subscripts (if it is your style to write scripts with
subcommands): one for setting up the validator and another for using it to
generate plots.  This division of tasks is useful because per-event scores
are calculated and stored during the setup step.  Once this (relatively)
expensive step is complete, plotting step can be tuned to your preferred
style.


Validator Construction
----------------------

Most possibilities for :class:`pybdt.validate.Validator` setup can be found
in the example script in pybdt/resources/examples/setup_sample_validator.py.
Validator setup consists broadly of three steps:  1) construct a Validator
for some classifier; 2) add DataSets to the Validator; 3) specify weighting
schemes and their plotting styles.

In order to reduce data copying, the Validator includes a mostly transparent
abstraction layer that allows the classifier and DataSets to be referenced
by filename rather than stored directly as Validator member data.  This
method is usually most convenient, with the caveat that no straightforward
interface is provided for updating the filename references if the classifier
or DataSets move; if you must reorganize your file tree, it will probably be
simplest to recreate the Validator from scratch.

If some ``bdt`` is already loaded in memory, then a Validator can be
constructed as::

    from pybdt.validate import Validator
    v = Validator (bdt)

However, since the classifier is typically already stored on disk -- say,
with a filename ``bdt_filename`` -- the Validator can simply refer to this
file::

    v = Validator (bdt_filename)

Next we tell the Validator about the DataSets we are interested in, e.g.::

    v.add_data ('bg', 'datasets/bg.ds', 'Background sim')
    v.add_data ('train_sig', 'datasets/train_sig.ds', 'Training signal sim')
    v.add_data ('train_data', 'datasets/train_bg.ds', 'Training data')
    v.add_data ('test_sig', 'datasets/test_sig.ds', 'Testing signal sim')
    v.add_data ('test_data', 'datasets/test_bg.ds', 'Testing data')

and so on.  In each case, we provide a DataSet identifier, the file path,
and a default label for plotting (more on this "default" later).  Note that
in a real analysis, it's best to specify an absolute rather than relative
file path.

Each :attr:`pybdt.validate.Validator.add_data()` call causes the Validator
to calculate and store per-event scores for later use.


DataSet weighting
-----------------

Once the DataSets are added to the Validator, the allowed weightings can be
specified.  For example, in the :ref:`ABC example <man_example>`, the signal
training sample is weighted like so::

    v.add_weighting ('weight', 'train_sig', color='cyan')

The first argument is the column of ``train_sig`` that contains the weights.
The desired plotting style can be specified in keyword arguments (see
:class:`pybdt.histlite.Style`).

For an experimental DataSet, the weight is simply 1/livetime; this case can
be handled, e.g., like::

    v.add_weighting ('livetime', 'train_data',
        line=False, markers=True, marker='.', color='.5', errorbars=True)

Note that ``livetime`` uses the :attr:`pybdt.ml.DataSet.livetime` property
to achieve the desired behavior rather than finding a column of per-event
weights.

We often want to designate an experimental testing DataSet as "data" and one
or more MC DataSets as "MC" so that data/MC ratio plots can later be
generated automatically.  The following weighting specifications from the
ABC example make use of this functionality::

    v.add_weighting ('livetime', 'test_data',
        line=False, markers=True, marker='.', color='black', errorbars=True,
        use_as_data=True)
    v.add_weighting ('weight', 'test_sig', color='blue', add_to_mc=True)
    v.add_weighting ('livetime', 'bg', color='purple', add_to_mc=True)
    v.setup_total_mc (color='green',)

In this case, ``test_data`` will be treated as the "data" sample.  the sum
of ``test_sig`` and ``bg`` will be treated as the total MC.  Finally, the
total MC will be plotted as a green line.

In the ABC example, the classifier is trained to find a signal sample that
is also present as a small fraction of the "background data".  This is
analogous to training a classifier to identify atmospheric muon neutrinos in
an IceCube dataset.  However, it is possible to specify that the signal
sample has some other weighting, e.g. an :math:`E^{-2}` spectrum.  Such a
weighting can be added to the Validator as follows::

    v.add_weighting ('weight_E2', 'test_sig', 'E2',
        color='red', linewidth=2)

Here, the weights are drawn from the ``test_sig`` column ``weight_E2``.
Note the third positional argument, ``'E2'``.  This is the identifier for
this spectral weighting, for this sample.  When this third argument is left
out, the identifier is automatically set to ``'default'``.

Once the Validator is configured, it can be saved for later usage and
reusage, e.g.::

    from pybdt.util import save
    save (v, 'sample.validator')


For a real-world example, see
the the `ml_score() and ml_validator()
<http://code.icecube.wisc.edu/projects/icecube/browser/IceCube/sandbox/richman/grb/scripts/trunk/ic79/ic79_north_numu.py#L524>`_
function from the IC79 northern :math:`\nu_\mu` analysis.
