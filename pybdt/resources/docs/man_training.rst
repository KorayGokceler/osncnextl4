.. _man_training:

DataSets and Training
=====================

Working with machine learning can be broadly divided into 3 phases:
training, testing, and application. On this page we discuss training
with pybdt.


DataSet objects
---------------

pybdt uses :class:`pybdt.ml.DataSet` objects to access your data.
Generating a DataSet object is very straightforward. The data should
be placed into `numpy <https://numpy.org/>`_ arrays with one
element per event. Then, you use a python dict to construct the
DataSet. For example,::

    # first, load variables a, b, and c, and event weights, from
    # wherever they are stored
    from pybdt import ml
    sim_data = ml.DataSet (dict (a=a, b=b, c=c, weight=weight))

Data columns can be accessed using the ``__getitem__`` operator,
i.e.::
    
    a = sim_data['a']
    weight = sim_data['weight']

It can be convenient to save a livetime for a DataSet, which may be
passed in as a single ``float`` like so: ::

    ...
    livetime = 3.15e7
    data = ml.DataSet (dict (a=a, b=b, c=c, livetime=livetime))

The livetime may be read or changed using the
:attr:`pybdt.ml.DataSet.livetime` property.


Training
--------

The simplest way to train is to use the included
``pybdt/resources/scripts/train.py`` script. Let's have a look at the
help output. ::

    Usage: train.py {options} [comma-sep'd variables] \
             [training signal] [training background] [bdt filename]

    Options:
      -h, --help            show this help message and exit

      Decision Tree options:
        -d N, --depth=N     make trees N levels deep
        -L, --nonlinear-cuts
                            use nonlinear cut spacing
        -c N, --num-cuts=N  try N cuts per var per node
        -s N, --min-split=N
                            do not split if a node contains fewer than N events
        -p STRENGTH, --prune-strength=STRENGTH
                            use STRENGTH prune strength
        -v N, --num-random-variables=N
                            use N randomly selected variables at each node (0 to
                            use every var at every node)

      Forest options:
        -t N, --num-trees=N
                            use N trees
        -b BETA, --beta=BETA
                            use BETA boost parameter
        -e FRAC, --frac-random-events=FRAC
                            use FRAC randomly selected fraction of events in each
                            tree (1, the default, to use every event in every
                            tree)


      Data options:
        --sig-weight=COLNAME
                            the name of the variable in which the signal weights
                            are stored
        --bg-weight=COLNAME
                            the name of the variable in which the bg weights are
                            stored


An example of the usage of this script can be found in
pybdt/resources/examples/train_sample_bdt.sh.  For a real-world example, see
the the `ml_train()
<http://code.icecube.wisc.edu/projects/icecube/browser/IceCube/sandbox/richman/grb/scripts/trunk/ic79/ic79_north_numu.py#L467>`_
function from the IC79 northern :math:`\nu_\mu` analysis.
