import networkx as nx
import warnings
from itertools import product
from collections import defaultdict


class IncompatibleGraphError(Exception):
    """Exception used when graphs are given to generate a QUBO that
    fail to satisfy or violate the conditions needed to generate the
    QUBO.

    Args:
        message (str):
            String telling the user why the graphs given cannot be used
            to generate a certain QUBO.
    """

    def __init__(self, message):
        self.message = message


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
    """Graph Isomorphism QUBO generator. Given a graph to embed
    (graph_to_embed) onto a target graph (target_graph), a QUBO
    is returned along with a dictionary that allows for translation from
    the QUBO variables to the potential node mapping combinations.

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
        (defaultdict(int), { int: (networkx
        node (any hashable object), networkx node) }:
            A tuple containing a QUBO represented by a defaultdict with
            `int()` as the `default_factory` attribute and a dictionary
            for translating from the indices used in the QUBO to the
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

    # get the "n" used for indexing
    num_nodes = graph_to_embed.number_of_nodes()

    # target_graph_dict and graph_to_embed dict are used to
    # translate between the integer indices found in Calude et al.'s
    # paper and any hashable type used in the NetworkX graph.
    target_graph_dict = {i: node for i, node
                         in enumerate(target_graph.nodes())}
    graph_to_embed_dict = {node: i for i, node
                           in enumerate(graph_to_embed.nodes())}

    # Allow for samples to be translated
    # back to prospective node mappings (must be fed by user to
    # `translate_sample()` along with sample from `dimod.SampleSet`)
    sample_translation_dict = {i: v for i, v in
                               enumerate(product(graph_to_embed.nodes(),
                                                 target_graph.nodes()))}

    # translate summation indices to individual QUBO indices
    # `product` usage inspired by: https://arxiv.org/pdf/2009.00140.pdf
    # code repository: https://github.com/bdury/QUBO-for-Qubit-Allocation
    td = {v: k for k, v in enumerate(product(range(num_nodes),
                                             range(num_nodes)))}

    # initialize empty QUBO
    Q = defaultdict(int)

    # Ensure mapping function is bijective
    for i in range(num_nodes):
        for i_p in range(num_nodes):
            Q[td[i, i_p], td[i, i_p]] += -2
            for j, j_p in zip(range(num_nodes), range(num_nodes)):
                if j != i:
                    if td[i, i_p] <= td[j, i_p]:
                        Q[td[i, i_p], td[j, i_p]] += 1
                    else:
                        Q[td[j, i_p], td[i, i_p]] += 1
                if j_p != i_p:
                    if td[i, i_p] <= td[i, j_p]:
                        Q[td[i, i_p], td[i, j_p]] += 1
                    else:
                        Q[td[i, j_p], td[i, i_p]] += 1

    # Ensure edge invariance
    for edge in graph_to_embed.edges():
        i, j = graph_to_embed_dict[edge[0]], graph_to_embed_dict[edge[1]]
        for i_p in range(num_nodes):
            for j_p in range(num_nodes):
                if not target_graph.has_edge(target_graph_dict[i_p],
                                             target_graph_dict[j_p]):
                    if td[i, i_p] <= td[j, j_p]:
                        Q[td[i, i_p], td[j, j_p]] += 1
                    else:
                        Q[td[j, j_p], td[i, i_p]] += 1

    return Q, sample_translation_dict


def subgraph_isomorphism(graph_to_embed, target_graph, induced=False):
    """Subgraph Isomorphism QUBO generator. Given a graph to embed
    (graph_to_embed) onto a target graph (target_graph), a QUBO
    is returned along with a dictionary that allows for translation from
    the QUBO variables to the potential node mapping combinations.

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
        (defaultdict(int), { int: (networkx
        node (any hashable object), networkx node) }:
            A tuple containing a QUBO represented by a defaultdict with
            `int()` as the `default_factory` attribute and a dictionary
            for translating from the indices used in the QUBO to the
            potential node mappings (represented as tuples).
            The dictionary should be passed to `translate_sample()` to
            generate the proper node-to-node mapping.
    """

    # Get number of nodes from each graph
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
    graph_to_embed_dict = {node: i for i, node
                           in enumerate(graph_to_embed.nodes())}

    # translate answers back to prospective node mappings
    sample_translation_dict = {i: v for i, v in
                               enumerate(product(graph_to_embed.nodes(),
                                                 target_graph.nodes()))}

    # translate summation indices to individual QUBO indices
    td = {v: k for k, v in enumerate(product(range(n1),
                                             range(n2)))}
    td.update({i: i + len(td) for i in range(n2)})

    # initialize empty QUBO
    Q = defaultdict(int)

    # Part of the QUBO that enforces the domain
    for i_p in range(n2):
        Q[td[i_p], td[i_p]] += -1
        for i in range(n1):
            Q[td[i, i_p], td[i, i_p]] += -2
            if td[i, i_p] <= td[i_p]:
                Q[td[i, i_p], td[i_p]] += 2
            else:
                Q[td[i_p], td[i, i_p]] += 2
            for j in range(n1):
                if i != j:
                    if td[i, i_p] <= td[j, i_p]:
                        Q[td[i, i_p], td[j, i_p]] += 1
                    else:
                        Q[td[j, i_p], td[i, i_p]] += 1
            for j_p in range(n2):
                if i_p != j_p:
                    if td[i, i_p] <= td[i, j_p]:
                        Q[td[i, i_p], td[i, j_p]] += 1
                    else:
                        Q[td[i, j_p], td[i, i_p]] += 1

    # Part of the QUBO that ensures edge preservation
    # Calude et al. note "the function ... is not necessarily 
    # edge-invariant: it has only to be “edge-preserving"

    for edge in graph_to_embed.edges():
        i, j = graph_to_embed_dict[edge[0]], graph_to_embed_dict[edge[1]]
        for i_p in range(n2):
            for j_p in range(n2):
                if not target_graph.has_edge(target_graph_dict[i_p],
                                             target_graph_dict[j_p]):
                    if td[i, i_p] <= td[j, j_p]:
                        Q[td[i, i_p], td[j, j_p]] += 1
                    else:
                        Q[td[j, j_p], td[i, i_p]] += 1

    # If the induced option is chosen, the node mapping
    # must be invariant as well and additional penalties added
    if induced:
        for non_edge in nx.non_edges(graph_to_embed):
            i, j = (graph_to_embed_dict[non_edge[0]],
                    graph_to_embed_dict[non_edge[1]])
            for i_p in range(n2):
                for j_p in range(n2):
                    if target_graph.has_edge(target_graph_dict[i_p],
                                             target_graph_dict[j_p]):
                        if td[i, i_p] <= td[j, j_p]:
                            Q[td[i, i_p], td[j, j_p]] += 1
                        else:
                            Q[td[j, j_p], td[i, i_p]] += 1

    return Q, sample_translation_dict


def translate_sample(sample, sample_translation_dict):
    """ Takes a sample from an annealing run and a translation
    dictionary returned from either `subgraph_isomorphism()` or
    `graph_isomorphism()` to generate a dictionary that maps nodes
    from the graph to embed to the target graph in the above two
    functions.

    Args:
        sample (pyqubo.DecodedSample):
            A sample from an annealer, that has already undergone
            translation via PyQUBO back into the original QUBO
            variable names
        sample_translation_dict {int: (networkx
        node (any hashable object), networkx node)}:
            Dictionary that maps QUBO variable names to potential
            node mappings

    Returns:
        { networkx node (any hashable type): networkx node }:
            A mapping from the nodes in the graph to be embedded to the
            target graph
    """

    # For each variable in the QUBO that was set to a "1" from the
    # annealing sample, a valid node mapping exists
    # and the sample_translation_dict is used to generate a
    # dictionary that contains all valid node mappings for the graph/
    # subgraph/induced subgraph isomorphism
    node_translation_dict = {}
    for key in sample.sample.keys():
        if key in sample_translation_dict and sample.sample[key] == 1:
            target_node, embed_node = sample_translation_dict[key]
            node_translation_dict[target_node] = embed_node

    return node_translation_dict
