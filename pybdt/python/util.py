# util.py

__doc__ = """Some general utility functions used by pybdt."""

import pickle
import errno
import os

import numpy as np

from pybdt import ml

def save (obj, filename):
    """Save obj directly to filename.

    :type   obj: object
    :param  obj: The object to save.

    :type   filename: str
    :param  filename: The filename.

    """
    with open (filename, 'wb') as f:
        pickle.dump (obj, f, -1)

def load (filename):
    """Load object directly from filename.

    :type   filename: str
    :param  filename: The filename containing the pickled object.

    """
    with open (filename, 'rb') as f:
        return pickle.load (f)

def mkdir (path):
    """Like os.makedirs, except it is not an error if path already exists.

    :type   path: str
    :param  path: The path to create.
    """
    try:
        os.makedirs (path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return path

def load_sklearn (filename, feature_names, signal_id, key=None):
    """Load a classifier saved using ``sklearn.externals.joblib.dump()``.

    :type   filename: str
    :param  filename: The filename containing the saved classifier.

    :type   feature_names: list of str
    :param  feature_names: The names of the event features to use.

    :type   signal_id: int
    :param  signal_id: The signal class id: either 0 or 1.

    :type   key: str
    :param  key: The key pointing to the classifier.  If not given, the
        classifier must be the only object in the file.
    """
    # get the sklearn BDT from disk
    try:
        # lazy import: the whole point is that sklearn might not be installed
        try:
            import joblib
        except:
            from sklearn.externals import joblib
        contents = joblib.load (filename)
    except:
        try:
            contents = load (filename)
        except:
            raise TypeError ('file must be saved with pickle or joblib')
    if key is None:
        if len (contents) != 1:
            raise ValueError (
                '`key` must be provided when file does not contain exactly '
                'one object')
        else:
            key = contents.keys()[0]
    sbdt = contents[key]
    return sklearn_to_BDTModel (sbdt, feature_names, signal_id)

def sklearn_to_BDTModel (sbdt, feature_names, signal_id):
    """Convert a scikit-learn classifier.

    :type   sbdt: sklearn classifier
    :param  sbdt: The scikit-learn classifier.

    :type   feature_names: list of str
    :param  feature_names: The names of the event features to use.

    :type   signal_id: int
    :param  signal_id: The signal class id: either 0 or 1.
    """

    if signal_id not in (0, 1):
        raise ValueError ('`signal_id` must be either 0 or 1')

    # recursively convert tree nodes to DTNode
    def sknode_to_DTNode (estimator, node_id):
        tree = estimator.tree_
        # get child IDs
        left_id = tree.children_left[node_id]
        right_id = tree.children_right[node_id]
        n_total = tree.n_node_samples[node_id]

        # get separation function
        seps = {
            'gini': (lambda p: p * (1 - p)),
            'entropy': (lambda p: -p * np.log (p)),
        }
        criterion = estimator.get_params()['criterion']
        sep = seps[criterion]

        # is this is a leaf?
        if left_id == -1:
            # weights given directly
            weights = tree.value[node_id][0]
            if signal_id == 0:
                w_sig, w_bg = weights
            else:
                w_bg, w_sig = weights
            purity = w_sig / (w_bg + w_sig)
            # resort to estimating counts (metadata not used for scoring)
            n_sig = int (purity * n_total)
            n_bg = n_total - n_sig
            sep_index = sep (purity)
            return ml.DTNode (w_sig, w_bg, n_sig, n_bg, sep_index)

        # if not a leaf, then a split
        feature_id = tree.feature[node_id]
        feature_val = tree.threshold[node_id]
        # scipy uses <= rather than < convention
        feature_val = np.nextafter (
            feature_val, feature_val + np.abs (.1 * feature_val))

        # get left and right subtrees
        left = sknode_to_DTNode (estimator, left_id)
        right = sknode_to_DTNode (estimator, right_id)

        # read their properties
        w_sig = left.w_sig + right.w_sig
        w_bg = left.w_bg + right.w_bg
        n_sig = left.n_sig + right.n_sig
        n_bg = left.n_bg + right.n_bg

        w_total = w_bg + w_sig
        purity = w_sig / w_total
        sep_index = sep (purity)
        sep_gain = sep_index * w_total \
            - left.w_total * left.sep_index \
            - right.w_total * right.sep_index

        return ml.DTNode (w_sig, w_bg, n_sig, n_bg, sep_index,
                        sep_gain, feature_id, feature_val,
                        left, right)


    # convert one tree to DTModel
    def sktree_to_DTModel (estimator):
        root = sknode_to_DTNode (estimator, 0)
        return ml.DTModel (feature_names, root)


    # get DTModels and their weights
    dtmodels = [sktree_to_DTModel (estimator)
                for estimator in sbdt.estimators_]
    alphas = sbdt.estimator_weights_
    return ml.BDTModel (feature_names, dtmodels, alphas)
