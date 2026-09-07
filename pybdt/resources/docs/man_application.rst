.. _man_application:

Classifier Application
======================

Once a classifier has been trained and validated to the your liking, you are
ready to use it to classify events of otherwise unknown class.  This can be
done for individual samples or for whole ensembles at a time.  Here we go
over methods of classifier application provided by pybdt.


Scoring with IceTray
--------------------

For IceCube data, it may be convenient to store a classifier score as an
I3Double in each Physics frame.  This can be done relatively simply with
:class:`icecube.pybdtmodule.PyBDTModule`.  For the :ref:`ABC example
<man_example>`, you can calculate and store the scores like so::

    from icecube.icetray import I3Tray
    from icecube import dataio
    from icecube.pybdtmodule import PyBDTModule

    tray = I3Tray()
    [...]

    def varsfunc (frame):
        a = frame['A'].value
        b = frame['B'].value
        c = frame['C'].value
        out = dict (a=a, b=b, c=c)
        return out

    tray.AddModule (PyBDTModule, 'bdt',
        BDTFilename='path/to/sample.bdt',
        varsfunc=varsfunc,
        OutputName='Score'
    )
    [...]

It is up to the user to provide ``varsfunc()``, which extracts the relevant
features from the frame for use by :class:`icecube.pybdtmodule.PyBDTModule`.
An alternative implemmentation would inspect each frame directly for
I3Double's corresponding to each required feature, but that approach would
require unnecessary pollution of the frames when features are nested deep in
frame objects.  Consider possible features ``zenith = SplineMPE.dir.zenith``
or ``plogl = SPEFitFitParams.logl / (SPEFitFitParams.ndof + 1.5)``.  The use
of ``varsfunc()`` simplifies scoring when training features are derived
from, but not exactly corresponding to, variables produced by the rest of
the processing chain.


Scoring without IceTray
-----------------------

In other contexts, it is useful to calculate scores outside of the IceTray
framework.  This can be done for individual events or ensembles at a time.
An individual event can be scored like so::

    score = bdt.score_event ({'a': 2.7, 'b': -0.31, 'c': 116})

The ``dict`` argument can of course be calculated by any method so long as
the keys match the features used by the classifier and the values are real
numbers.

Suppose one has an ensemble of events with numpy arrays ``a``, ``b`` and
``c`` holding one value per event.  Then the per-event scores can be
calculated like so::

    scores = bdt.score_dict (dict (a=a, b=b, c=c))

If a :class:`pybdt.ml.DataSet` object ``data`` is already available, the
per-event scores can be calculated like so::

    scores = bdt.score_DataSet (data)

In these last two exaples, the resulting ``scores`` are a numpy array with
``dtype=float``.  

A convienience method :attr:`pybdt.ml.BDTModel.score()` is provided which
automatically calls the correct one of the above methods, given the input.
Thus the following calls all work as expected::

    score = bdt.score ({'a': 2.7, 'b': -0.31, 'c': 116})
    scores = bdt.score (dict (a=a, b=b, c=c))
    scores = bdt.score (data)

Finally, note that the leaf purity :math:`p` can be used to scale the
contribution of each tree in the model (see :ref:`man_dt_intro`).  This is
especially common for forests in which the trees are differentiated only by
randomization but not by boosting.  To enable this type of scoring, change
the above calls to::

    score = bdt.score ({'a': 2.7, 'b': -0.31, 'c': 116}, use_purity=True)
    scores = bdt.score (dict (a=a, b=b, c=c), use_purity=True)
    scores = bdt.score (data, use_purity=True)
