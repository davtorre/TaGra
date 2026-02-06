"""
Community detection and analysis.

This module provides functions for detecting communities in graphs
and computing community-related metrics.
"""

from datetime import datetime
from typing import Dict, Any, Optional, Union, List

import networkx as nx
import numpy as np


def detect_communities(
    graph: Union[nx.Graph, 'TaGraGraph'],
    method: str = 'girvan_newman',
    verbose: bool = True
) -> List[List[int]]:
    """
    Detect communities in the graph.

    Parameters
    ----------
    graph : nx.Graph or TaGraGraph
        Graph to analyze
    method : str, default='girvan_newman'
        Community detection method:
        - 'girvan_newman': Edge betweenness based
        - 'louvain': Louvain algorithm (requires python-louvain)
        - 'label_propagation': Label propagation
    verbose : bool, default=True
        Print progress messages

    Returns
    -------
    List[List[int]]
        List of communities, each as a list of node IDs

    Examples
    --------
    >>> communities = detect_communities(G)
    >>> print(f"Found {len(communities)} communities")
    """
    # Handle TaGraGraph
    if hasattr(graph, 'to_networkx'):
        G = graph.to_networkx()
    else:
        G = graph

    if verbose:
        print(f"{datetime.now()}: Detecting communities using {method}...")

    try:
        if method == 'girvan_newman':
            communities_generator = nx.algorithms.community.girvan_newman(G)
            top_level = next(communities_generator)
            communities = [list(c) for c in sorted(top_level, key=len, reverse=True)]

        elif method == 'louvain':
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(G)
                # Convert partition dict to list of communities
                comm_dict = {}
                for node, comm_id in partition.items():
                    if comm_id not in comm_dict:
                        comm_dict[comm_id] = []
                    comm_dict[comm_id].append(node)
                communities = sorted(comm_dict.values(), key=len, reverse=True)
            except ImportError:
                raise ImportError("python-louvain package required for Louvain method")

        elif method == 'label_propagation':
            communities_gen = nx.algorithms.community.label_propagation_communities(G)
            communities = [list(c) for c in sorted(communities_gen, key=len, reverse=True)]

        else:
            raise ValueError(f"Unknown community detection method: {method}")

        if verbose:
            print(f"{datetime.now()}: Found {len(communities)} communities")

        return communities

    except Exception as e:
        if verbose:
            print(f"{datetime.now()}: Community detection failed: {str(e)}")
        return []


def compute_modularity(
    graph: Union[nx.Graph, 'TaGraGraph'],
    communities: List[List[int]],
    verbose: bool = True
) -> float:
    """
    Compute modularity score for a partition.

    Modularity measures the quality of a community partition,
    comparing the density of edges within communities to what
    would be expected by chance.

    Parameters
    ----------
    graph : nx.Graph or TaGraGraph
        Graph
    communities : List[List[int]]
        Community partition
    verbose : bool, default=True
        Print progress messages

    Returns
    -------
    float
        Modularity score (-0.5 to 1.0)

    Examples
    --------
    >>> modularity = compute_modularity(G, communities)
    >>> print(f"Modularity: {modularity:.4f}")
    """
    if hasattr(graph, 'to_networkx'):
        G = graph.to_networkx()
    else:
        G = graph

    if not communities or len(communities) <= 1:
        return 0.0

    if verbose:
        print(f"{datetime.now()}: Computing modularity...")

    try:
        modularity = nx.algorithms.community.modularity(G, communities)
        if verbose:
            print(f"{datetime.now()}: Modularity: {modularity:.4f}")
        return modularity
    except Exception as e:
        if verbose:
            print(f"{datetime.now()}: Modularity computation failed: {str(e)}")
        return 0.0


def compute_community_stats(
    graph: Union[nx.Graph, 'TaGraGraph'],
    communities: List[List[int]],
    target_attribute: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute statistics about detected communities.

    Parameters
    ----------
    graph : nx.Graph or TaGraGraph
        Graph
    communities : List[List[int]]
        Detected communities
    target_attribute : str, optional
        Attribute for composition analysis

    Returns
    -------
    Dict[str, Any]
        Statistics including:
        - n_communities: Number of communities
        - sizes: List of community sizes
        - size_mean: Average community size
        - size_std: Standard deviation of sizes
        - composition: (if target_attribute) Dict of attribute counts per community
    """
    if hasattr(graph, 'to_networkx'):
        G = graph.to_networkx()
    else:
        G = graph

    stats = {
        'n_communities': len(communities),
        'sizes': [len(c) for c in communities],
    }

    if stats['sizes']:
        stats['size_mean'] = np.mean(stats['sizes'])
        stats['size_std'] = np.std(stats['sizes'])
        stats['size_min'] = min(stats['sizes'])
        stats['size_max'] = max(stats['sizes'])
    else:
        stats['size_mean'] = 0
        stats['size_std'] = 0
        stats['size_min'] = 0
        stats['size_max'] = 0

    # Composition analysis
    if target_attribute is not None:
        NONE_STR = 'None'
        unique_labels = set(G.nodes[n].get(target_attribute, NONE_STR) for n in G.nodes)

        composition = {}
        for i, community in enumerate(communities):
            composition[i] = {label: 0 for label in unique_labels}
            for node in community:
                label = G.nodes[node].get(target_attribute, NONE_STR)
                composition[i][label] += 1

        stats['composition'] = composition

    return stats


def measure_mixing_matrix(
    graph: Union[nx.Graph, 'TaGraGraph'],
    communities: Dict[Any, List[int]]
) -> Dict[tuple, int]:
    """
    Measure the mixing matrix between communities.

    Counts edges between pairs of communities.

    Parameters
    ----------
    graph : nx.Graph or TaGraGraph
        Graph
    communities : Dict[Any, List[int]]
        Dictionary mapping community ID to list of nodes

    Returns
    -------
    Dict[tuple, int]
        Dictionary mapping (comm_i, comm_j) to edge count

    Raises
    ------
    ValueError
        If communities don't cover all graph nodes
    """
    if hasattr(graph, 'to_networkx'):
        G = graph.to_networkx()
    else:
        G = graph

    # Initialize edge counts
    community_edge_count = {
        (i, j): 0 for i in communities.keys() for j in communities.keys()
    }

    # Create node to community mapping
    node_to_community = {}
    for community, nodes in communities.items():
        for node in nodes:
            node_to_community[node] = community

    # Verify all nodes are covered
    graph_nodes = set(G.nodes())
    community_nodes = set(node_to_community.keys())
    if graph_nodes != community_nodes:
        missing_in_communities = graph_nodes - community_nodes
        missing_in_graph = community_nodes - graph_nodes
        raise ValueError(
            f"Node mismatch. In graph not in communities: {missing_in_communities}. "
            f"In communities not in graph: {missing_in_graph}"
        )

    # Count edges between communities
    for source, target in G.edges():
        c_source = node_to_community[source]
        c_target = node_to_community[target]
        if not G.is_directed():
            community_edge_count[c_source, c_target] += 1
            community_edge_count[c_target, c_source] += 1
        else:
            community_edge_count[c_source, c_target] += 1

    return community_edge_count
