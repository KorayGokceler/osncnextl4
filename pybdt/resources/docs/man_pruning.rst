.. _man_pruning_intro:

Introduction to pruning
=======================

Splits in a decision tree are not all equally important. Nodes close
to the root node do most of the classification; deeper nodes help
classify the less typical events. Deeper nodes are also more likely to
be :ref:`overtrained <man_overtraining_intro>` on fluctuations in the
training sample.

*Pruning* a decision tree consists of identifying splits that are less
important by some metric, and converting those split nodes into leaf
nodes. Pruning can be a useful tool to reduce or eliminate
overtraining.


Same leaf pruning
-----------------

*Same leaf pruning* is a very simple method. The tree is searched for
split nodes for which each child node is a leaf. If both child nodes
are the signal leaves or both are background leaves, then the tree is
pruned at that split (the split becomes a leaf node). This is repeated
until no more splits with same leaf children remain.


Cost complexity pruning
-----------------------

*Cost complexity pruning* identifies subtrees that add more leaves but
not much separation. The algorithm defines a pruning sequence, where
the first step prunes the most expendable subtree, and the last step
would leave only the root node.

In pybdt, the user provides a pruning strength parameter, on a scale
from 0 to 100, which specifies the percentage of the pruning sequence
to actually execute. The pruning sequence is calculated by pruning a
copy of the tree until the root is reached, noting the node at which
each prune operation takes place in the copy. Afterwards, the desired
percentage of the prune sequence is executed on the original tree.

At each step, the next node to prune is calculated as follows.

-   For each split node, calculate the weighted separation gain:

        :math:`g = W \cdot p \cdot (1 - p)`.

-   Then calculate the pruning cost:

        :math:`\rho = \frac{g - g_L - g_R}{n_\mathrm{leaves} - 1}`,

    where :math:`L` and :math:`R` refer to the left and right leaves,
    and :math:`n_\mathrm{leaves}` refers to the total number of leaves
    in the subtree below this split node.

-   Prune at the split node with the smallest :math:`\rho`.


