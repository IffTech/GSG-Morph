# G/SG Morph - The Graph/Subgraph Isomorphism Library for Quantum Annealers.
# Copyright (C) 2021 If and Only If (Iff) Technologies

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from pyqubo import Array
import networkx as nx
import warnings
from .utils import IncompatibleGraphError


def has_edge(graph, edge_tuple):
    """An implementation of the e_ij constant seen in the QUBO
    formulas from Calude et al.'s paper
    (https://doi.org/10.1016/j.tcs.2017.04.016). Returns a 1 if an
    edge is present in a graph, 0 otherwise.

    Args:
        graph (networkx.classes.graph.Graph):
            Undirected NetworkX graph
        edge_tuple ((any hashable type, any hashable type)):
            A tuple containing two nodes of the above graph.

    Returns:
        (int):
            1 if an edge is present in the graph, 0 otherwise.
    """
    if graph.has_edge(*edge_tuple):
        return 1
    return 0


def graph_isomorphism(graph_to_embed, target_graph):
    """Subgraph Isomorphism QUBO generator. Given a graph to embed
    (graph_to_embed) onto a target graph (target_graph), a PyQUBO
    expression is returned along with a dictionary that allows for
    translation from the QUBO variables to the potential node mapping
    combinations.

    Args:
        graph_to_embed (networkx.classes.graph.Graph):
            An undirected graph to be mapped to another graph.
        target_graph (networkx.classes.graph.Graph):
            An undirected graph that the graph_to_embed (see above)
            is to be mapped onto.

    Raises:
        IncompatibleGraphError:
            A graph isomorphism can only exist if the number of nodes
            and the number of edges in the graph are identical. If
            graphs are given that fail to satisfy this criteria the
            exception is raised and the QUBO is not generated.

    Returns:
        (pyqubo.Add, sample_translation_dict (dict of str: (networkx
        node (any hashable object), networkx node)):
            A tuple containing a PyQUBO expression and a dictionary for
            translating from the variable names used in the QUBO to the
            potential node mappings (represented as tuples).
            The dictionary should be passed to `translate_sample()` to
            generate the proper node-to-node mapping.
    """

    # Ensure that the requirements
    # |V1| = |V2| (graphs must have the same number of nodes/vertices)
    # and |E1| = |E2| (graphs must have the same number of edges)
    # holds
    if (graph_to_embed.number_of_nodes()
            != target_graph.number_of_nodes()):
        raise IncompatibleGraphError("The number of vertices "
                                     "do not match!")

    if (graph_to_embed.number_of_edges()
            != target_graph.number_of_edges()):
        raise IncompatibleGraphError("The number of edges "
                                     "do not match!")

    # target_graph_dict and graph_to_embed dict are used to
    # translate between the integer indices found in Calude et al.'s
    # paper and any hashable type used in the NetworkX graph.
    target_graph_dict = {i: node for i, node
                         in enumerate(target_graph.nodes())}
    graph_to_embed_dict = dict()

    # Allows for translation from the QUBO variable names
    # which follow an `x[i][j]` format to that of the actual
    # potential node mappings (stored in a tuple)
    sample_translation_dict = dict()

    for i, graph_to_embed_node in enumerate(graph_to_embed.nodes()):
        for j, target_graph_node in enumerate(target_graph.nodes()):
            sample_translation_dict['x[{0}][{1}]'.format(i, j)] = \
                (graph_to_embed_node, target_graph_node)
        graph_to_embed_dict[graph_to_embed_node] = i

    num_nodes = graph_to_embed.number_of_nodes()

    # Generate a vector of binary variables
    bin_vec = Array.create('x', shape=(num_nodes, num_nodes),
                           vartype='BINARY')

    # Part of the QUBO that enforces bijectivity.
    # In Calude et al.'s paper, the function (H(x)) has two seperate
    # summation terms and they are represented here as "h1"
    # being the first term and "h2" being the second.

    h1 = 0
    for i in range(num_nodes):
        h1_iter_sum = 0
        for i_prime in range(num_nodes):
            h1_iter_sum += bin_vec[(i, i_prime)]
        h1 += (1 - h1_iter_sum)**2

    h2 = 0
    for i_prime in range(num_nodes):
        h2_iter_sum = 0
        for i in range(num_nodes):
            h2_iter_sum += bin_vec[(i, i_prime)]
        h2 += (1 - h2_iter_sum)**2

    h = h1 + h2

    # Part of the QUBO responsible for ensuring edge invariance
    p = 0
    for edge in graph_to_embed.edges:
        i = graph_to_embed_dict[edge[0]]
        j = graph_to_embed_dict[edge[1]]
        outer_sum = 0
        for i_prime in range(num_nodes):
            inner_sum = 0
            for j_prime in range(num_nodes):
                inner_sum += bin_vec[j, j_prime] \
                          * (1 - has_edge(target_graph,
                                          (target_graph_dict[i_prime],
                                           target_graph_dict[j_prime])))
            outer_sum += bin_vec[i, i_prime] * inner_sum
        p += outer_sum

    return h + p, sample_translation_dict


def subgraph_isomorphism(graph_to_embed, target_graph, induced=False):
    """Subgraph Isomorphism QUBO generator. Given a graph to embed
    (graph_to_embed) onto a target graph (target_graph), a PyQUBO
    expression is returned along with a dictionary that allows for
    translation from the QUBO variables to the potential node mapping
    combinations.

    Args:
        graph_to_embed (networkx.classes.graph.Graph):
            An undirected graph to be mapped to another graph.
        target_graph (networkx.classes.graph.Graph):
            An undirected graph that the graph_to_embed (see above)
            is to be mapped onto.
        induced (bool):
            By default set to False, but if set to True will add
            additional constraints to produce the Induced Subgraph
            Isomorphism QUBO

    Raises:
        IncompatibleGraphError:
            A graph isomorphism can only exist if the number of nodes
            and the number of edges in the graph are identical. If
            graphs are given that fail to satisfy this criteria the
            exception is raised and the QUBO is not generated.

    Returns:
        (pyqubo.Add, sample_translation_dict (dict of str: (networkx
        node (any hashable object), networkx node)):
            A tuple containing a PyQUBO expression and a dictionary for
            translating from the variable names used in the QUBO to the
            potential node mappings (represented as tuples).
            The dictionary should be passed to `translate_sample()` to
            generate the proper node-to-node mapping.
    """

    n1 = graph_to_embed.number_of_nodes()
    n2 = target_graph.number_of_nodes()

    # Ensure that the requirements
    # |V1| <= |V2| (graph to embed must have number of nodes less than
    # or equal to target graph)
    # and |E1| <= |E2| (graph to embed must have number of edges
    # less than or equal to target graph)
    # holds
    if(n1 > n2):
        raise IncompatibleGraphError("The graph to embed has more nodes"
                                     " than the target graph!")

    if(graph_to_embed.number_of_edges()
       > target_graph.number_of_edges()):
        raise IncompatibleGraphError("The graph to embed has more edges"
                                     " than the target graph!")

    if((n1 == n2) and (graph_to_embed.number_of_edges() ==
                       target_graph.number_of_edges())):
        warnings.warn("The graphs provided have an equal number of "
                      "edges and nodes and are better suited for usage "
                      "with graph_isomorphism() where slack variables "
                      "are not introduced to the QUBO",
                      UserWarning, stacklevel=2)

    # target_graph_dict and graph_to_embed dict are used to
    # translate between the integer indices found in Calude et al.'s
    # paper and any hashable type used in the NetworkX graph.
    target_graph_dict = {i: node for i, node
                         in enumerate(target_graph.nodes())}

    sample_translation_dict = dict()
    graph_to_embed_dict = dict()

    for i, graph_to_embed_node in enumerate(graph_to_embed.nodes()):
        for j, target_graph_node in enumerate(target_graph.nodes()):
            sample_translation_dict['x[{0}][{1}]'.format(i, j)] = \
                (graph_to_embed_node, target_graph_node)
        graph_to_embed_dict[graph_to_embed_node] = i

    # Generate a vector of binary variables (similar to the
    # graph isomorphism QUBO problem) along with a number of slack
    # binary varialbes equal to the number of nodes in
    # the target graph
    bin_vec = Array.create('x', shape=(n1, n2), vartype='BINARY')
    slack_bin_vec = Array.create('y', shape=n2, vartype='BINARY')

    # Part of the QUBO that enforces injectivity.
    # In Calude et al.'s paper, the function (H(z)) is split into
    # two terms. The first term only takes the variables in the main
    # binary vector while the second term takes additional slack
    # variables.
    hx = 0
    for i in range(n1):
        hx_iter_sum = 0
        for i_prime in range(n2):
            hx_iter_sum += bin_vec[i, i_prime]
        hx_iter_sum = (1 - hx_iter_sum)**2
        hx += hx_iter_sum

    hy = 0
    for i_prime in range(n2):
        hy_iter_sum = 0
        for i in range(n1):
            hy_iter_sum += (bin_vec[i, i_prime]
                            - slack_bin_vec[i_prime])
        hy_iter_sum = (1 - hy_iter_sum)**2
        hy += hy_iter_sum

    # Part of the QUBO that ensures the preservation of edges
    p = 0
    for edge in graph_to_embed.edges:
        i = graph_to_embed_dict[edge[0]]
        j = graph_to_embed_dict[edge[1]]
        outer_sum = 0
        for i_prime in range(n2):
            inner_sum = 0
            for j_prime in range(n2):
                inner_sum += bin_vec[j, j_prime] \
                          * (1 - has_edge(target_graph,
                                          (target_graph_dict[i_prime],
                                           target_graph_dict[j_prime])))
            outer_sum += bin_vec[i, i_prime] * inner_sum
        p += outer_sum

    # If the "induced" argument is set to 'True', additional
    # constraints are added such that the QUBO is edge invariant as well
    if induced:
        n = 0
        for non_edge in nx.non_edges(graph_to_embed):
            i = graph_to_embed_dict[non_edge[0]]
            j = graph_to_embed_dict[non_edge[1]]
            outer_sum = 0
            for i_prime in range(n2):
                inner_sum = 0
                for j_prime in range(n2):
                    inner_sum += bin_vec[j, j_prime] \
                              * has_edge(target_graph,
                                         (target_graph_dict[i_prime],
                                          target_graph_dict[j_prime]))
                outer_sum += bin_vec[i, i_prime] * inner_sum
            n += outer_sum

        return hx + hy + p + n, sample_translation_dict

    return hx + hy + p, sample_translation_dict


def translate_sample(sample, sample_translation_dict):
    """ Takes a sample already translated from PyQUBO and a translation
    dictionary returned from either `subgraph_isomorphism()` or
    `graph_isomorphism()` to generate a dictionary that maps nodes
    from the graph to embed to the target graph in the above two
    functions.

    Args:
        sample (pyqubo.DecodedSample):
            A sample from an annealer, that has already undergone
            translation via PyQUBO back into the original QUBO
            variable names
        sample_translation_dict (dict of str: (networkx node
        (any hashable object), networkx node)):
            Dictionary that maps QUBO variable names to potential
            node mappings

    Returns:
        (dict of networkx node (any hashable type): networkx node):
            A mapping from the nodes in the graph to be embedded to the
            target graph
    """

    # For each variable in the QUBO that was set to a "1" from the
    # PyQUBO-translated annealing sample, a valid node mapping exists
    # and the sample_translation_dict is used to generate a
    # dictionary that contains all valid node mappings for the graph/
    # subgraph isomorphism
    node_translation_dict = {}
    for key in sample.sample.keys():
        if sample.sample[key] == 1:
            target_node, embed_node = sample_translation_dict[key]
            node_translation_dict[target_node] = embed_node

    return node_translation_dict
