# viz.py


__doc__ = """Visualization functions for PyBDT forests."""


import numpy as np


def float_to_scientific_notation (x, sigfigs=None, pretty=True, html=False):
    """Create a string representation of `x` in scientific notation.

    :type   x: float
    :param  x: The number.

    :type   pretty: bool
    :param  pretty: Whether to pretty-print scientific notation, or simply use
        'e' to separate the significand and the exponent.

    :type   html: bool
    :param  html: If true, use the sup tag to write the super script.

    :return: A string of the form '`a` x10^ `b`' or '`a` e `b`'.
    """
    exponent = int (np.floor (np.log10 (abs (x))))
    significand = x / 10**exponent
    if pretty:
        if html:
            sep = '&times;10<sup>'
            last = '</sup>'
        else:
            sep = 'x10^'
            last = ''
    else:
        sep = 'e'
    if sigfigs is None:
        fmt = '{{0}}{0}{{1}}{1}'.format (sep, last)
    else:
        fmt = '{{0:.{0}f}}{1}{{1}}{2}'.format (sigfigs - 1, sep, last)
    return fmt.format (significand, exponent)


def dtmodel_to_graphviz (model,
        title='',
        emphasis='cuts',
        leaf_shape='ellipse',
        split_shape='box',
        sigfigs=3,
        float_pretty=True,):
    """Plot a :class:`pybdt.DTModel` visualization using graphviz (requires
    `pydot <https://github.com/pydot/pydot>`_).

    :type   model: :class:`pybdt.DTModel`
    :param  model: The DTModel to visualize.

    :type   title: str
    :param  title: Title for the DTModel visualization.

    :type   emphasis: str
    :param  emphasis: Either 'cuts' or 'events'; what style of labeling to use.

    :type   leaf_shape: str
    :param  leaf_shape: The shape for leaf nodes. (see
        `graphviz shapes <http://www.graphviz.org/doc/info/shapes.html>`_)

    :type   split_shape: str
    :param  split_shape: The shape for split nodes. (see
        `graphviz shapes <http://www.graphviz.org/doc/info/shapes.html>`_)

    :type   sigfigs: int
    :param  sigfigs: If given, the number of significant digits to use when
        printing floats.

    :type   float_pretty: bool
    :param  float_pretty: Whether to pretty-print scientific notation, or
        simply use 'e' to separate the significand and the exponent.

    """
    import pydot
    node_dict = {}

    if emphasis not in ('cuts', 'events'):
        raise ValueError ('emphasis must be in ("cuts", "events")')

    def node_label (node):
        lines = []
        if emphasis == 'events':
            lines.append ('"Wsig (Nsig): {0} ({1})'.format (
                float_to_scientific_notation (
                    node.w_sig, sigfigs=sigfigs, pretty=float_pretty),
                node.n_sig))
            lines.append ('Wbg (Nbg): {0} ({1})'.format (
                float_to_scientific_notation (
                    node.w_bg, sigfigs=sigfigs, pretty=float_pretty),
                node.n_bg))
            lines.append ('Purity: {0:.3f}"'.format (node.purity))
        else:
            p = format (max (node.purity, 1 - node.purity), '.3f')
            if node.is_leaf:
                if node.purity > 0.5:
                    lines.append ('Signal leaf')
                    lines.append ('p = {0}'.format (p))
                else:
                    lines.append ('Background leaf')
                    lines.append ('1 - p = {0}'.format (p))
            else:
                feature_name = model.feature_names[node.feature_id]
                feature_val = float_to_scientific_notation (
                        node.feature_val, sigfigs=sigfigs, pretty=float_pretty)
                lines.append ('Cut on \\"{0}\\" at {1}'.format (
                    feature_name, feature_val))
                if node.purity > 0.5:
                    lines.append ('p = {0}'.format (p))
                else:
                    lines.append ('1 - p = {0}'.format (p))
        return '"' + '\\n'.join (lines) + '"'

    def edge_label (node1, node2, is_left):
        """Label for node connection: is_left is bool."""
        if emphasis == 'events':
            label = '{0} {1} {2}'.format (
                model.feature_names[node1.feature_id],
                '<' if is_left else '>=',
                float_to_scientific_notation (
                    node1.feature_val, sigfigs=sigfigs, pretty=float_pretty) )
        elif emphasis == 'cuts':
            label = '<' if is_left else '>='
        return label

    def node_id (dt_node):
        """Unique id for dt_node.

        We can't use DTNodes as dict keys because hash(node) will be different
        each time.
        """
        return (dt_node.sep_gain, dt_node.w_sig, dt_node.w_bg)

    def create_graph_node (dt_node):
        """Create a new graph node for the dt_node."""
        n = str (len (node_dict))
        label = node_label (dt_node)
        if dt_node.is_leaf:
            return pydot.Node (n, label=label, shape=leaf_shape)
        else:
            return pydot.Node (n, label=label, shape=split_shape)

    def get_graph_node (dt_node):
        """Return cached copy of graph node or create new one if it doesn't
        exist."""
        if node_id (dt_node) in node_dict:
            graph_node = node_dict[node_id (dt_node)]
        else:
            graph_node = create_graph_node (dt_node)
            node_dict[node_id (dt_node)] = graph_node
        return graph_node

    def create_nodes_and_edge (dt_node1, dt_node2):
        graph_node1 = get_graph_node (dt_node1)
        graph_node2 = get_graph_node (dt_node2)
        edge = pydot.Edge (graph_node1, graph_node2)
        return graph_node1, graph_node2, edge

    def plot_dt0 (node):
        """Append the node, siblings (if present), and corresponding edges,
        recursing for siblings (if present)."""
        if node.is_leaf:
            return
        # handle left side
        graph_node, graph_node_left, edge = create_nodes_and_edge (
                node, node.left)
        edge.set_label (edge_label (node, node.left, True))
        graph.add_node (graph_node_left)
        graph.add_edge (edge)
        plot_dt0 (node.left)
        # handle right side
        graph_node, graph_node_right, edge = create_nodes_and_edge (
                node, node.right)
        edge.set_label (edge_label (node, node.right, False))
        graph.add_node (graph_node_right)
        graph.add_edge (edge)
        plot_dt0 (node.right)

    graph = pydot.Dot (graph_type='graph', charset='utf8')
    root_graph_node = get_graph_node (model.root)
    graph.add_node (root_graph_node)
    plot_dt0 (model.root)
    if title:
        graph.set_label (title)
    return graph


def dtmodel_to_text (model, indent=0, tab_width=2,
        split_func=None, leaf_func=None):
    """Print a :class:`pybdt.DTModel` visualization to a string.

    :type   model: :class:`pybdt.DTModel`
    :param  model: The DTModel to print.

    :type   indent: int
    :param  indent: The number of spaces to indent the entire visualization.

    :type   tab_width: int
    :param  tab_width: The amount to increase indentation at each tree level.

    :type   split_func: str function (:class:`DTNode`)
    :param  split_func: Custom function giving extra information for split
        DTNodes.

    :type   leaf_func: str function (:class:`DTNode`)
    :param  leaf_func: Custom function giving extra information for leaf
        DTNodes.

    :return: A string with no newline at the end.
    
    """
    lines = []
    def print_dtnode (node, level=0, side=''):
        spaces = (tab_width * level + indent) * ' '
        side_str = side + ' ' if side else ''
        leader = '{0}({1}) {2}'.format (spaces, level, side_str)
        if node.is_leaf:
            purity = node.purity
            is_signal = purity >= 0.5
            kind = 'S' if is_signal else 'B'
            info = 'p = {0:.6f}'.format (purity)
            if leaf_func is not None:
                extra = ' ({0})'.format (leaf_func (node))
            else:
                extra = ''
            lines.append ('{0}{1} leaf: {2}{3}'.format (
                leader, kind, info, extra))
        else:
            feature = model.feature_names[node.feature_id]
            value = node.feature_val
            info = 'split on {0} at {1:.3e}'.format (feature, value)
            if split_func is not None:
                extra = ' ({0})'.format (split_func (node))
            else:
                extra = ''
            lines.append ('{0}{1}{2}'.format (
                leader, info, extra))
            print_dtnode (node.left, level + 1, '<')
            print_dtnode (node.right, level + 1, '>')

    print_dtnode (model.root)
    return '\n'.join (lines)


def variable_importance_to_text (var_imp_dict, indent=0):
    """Get a table of variable importance values.

    :type   var_imp_dict: dict
    :param  var_imp_dict: One of
        :meth:`BDTModel.weighted_variable_importance`,
        :meth:`BDTModel.unweighted_variable_importance`,
        :meth:`BDTModel.weighted_variable_importance_n_use`, or
        :meth:`BDTModel.unweighted_variable_importance_n_use`.

    :return: A str containing the formatted table.

    """
    out = []
    max_len_number = np.max ([len (str (i))
        for i in range (len (var_imp_dict))])
    max_len_key = np.max ([len (k)
        for k in var_imp_dict.keys ()])
    fmt = '{{0}}{{1:>{0}}}. {{2:{1}}}: {{3:8.6f}}'.format (
            max_len_number, max_len_key + 1)
    for n, (k, v) in enumerate (sorted (
        iter(var_imp_dict.items ()), key=lambda a: -a[1])):
        out.append (fmt.format (indent * ' ', n+1, k, v))
    return '\n'.join (out)

