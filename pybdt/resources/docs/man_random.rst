.. _man_random_intro:

Introduction to decision tree randomization
===========================================

Decision tree classifiers are powerful because they find the best cut
for each region in the many dimensional parameter space. BDT
classifiers are an improvement over single decision trees because they
can provide good classification for events in the tails of the
variable distributions without becoming overtrained on fluctuations in
those distributions.

A more recent innovation is to generate so-called *Random Forests*.
These classifiers, like BDT classifiers, also make use of a forest of
decision trees to provide a score for events. However, rather than
using boosting to differentiate the individual trees, an element of
randomness is introduced.

Typically, one uses either boosting with no randomization, or
randomization with no boosting (boost strength of 0). However, there
is no technical reason why these techniques cannot be combined, and so
the implementation in pybdt allows the user to use both if desired.

pybdt allows the following two types of randomization.

    cut variable randomization

        The user provides an integer ``num_random_variables``, which
        must be less than the total number of variables being used.
        Then during training, *at each node*, only
        ``num_random_variables`` variables are randomly selected to be
        considered for choosing a cut.

    training event randomization

        The user provides a fraction ``frac_random_events`` between
        0.0 and 1.0.  Then during training, *for each tree*, only
        ``frac_random_events`` fraction of the full training sample is used
        for training.


In the ABC example, training event randomization is used to reduce training
sample overtraining. By using different events to train each tree, we avoid
tuning to fluctuations in the training sample.
