"""
Graph metrics computation.

This module provides functions for computing various graph metrics
including density, clustering, and connectivity.
"""

from datetime import datetime
from typing import Dict, Any, Optional, Union

import networkx as nx
import numpy as np


def compute_metrics(
    graph: Union[nx.Graph, 'TaGraGraph'],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute basic graph metrics.

    Parameters
    ----------
    graph : nx.Graph or TaGraGraph
        Graph to analyze
    verbose : bool, default=True
        Print progress messages

    Returns
    -------
    Dict[str, Any]
        Dictionary containing computed metrics:
        - nodes: Number of nodes
        - edges: Number of edges
        - density: Graph density
        - avg_clustering: Average clustering coefficient
        - connected_components: Number of connected components
        - largest_component_size: Size of largest component
        - assortativity: Degree assortativity coefficient

    Examples
    --------
    >>> metrics = compute_metrics(G)
    >>> print(f"Density: {metrics['density']:.4f}")
    """
    # Handle TaGraGraph
    if hasattr(graph, 'to_networkx'):
        G = graph.to_networkx()
    else:
        G = graph

    metrics = {}

    if verbose:
        print(f"{datetime.now()}: Computing basic graph metrics...")

    # Basic counts
    metrics['nodes'] = G.number_of_nodes()
    metrics['edges'] = G.number_of_edges()

    # Density
    metrics['density'] = nx.density(G)

    # Clustering
    if verbose:
        print(f"{datetime.now()}: Computing clustering coefficient...")
    metrics['avg_clustering'] = nx.average_clustering(G)

    # Connected components
    if verbose:
        print(f"{datetime.now()}: Analyzing connected components...")
    components = list(nx.connected_components(G))
    metrics['connected_components'] = len(components)
    metrics['largest_component_size'] = len(max(components, key=len)) if components else 0

    # Assortativity
    try:
        if verbose:
            print(f"{datetime.now()}: Computing degree assortativity...")
        metrics['assortativity'] = nx.degree_assortativity_coefficient(G)
    except Exception as e:
        if verbose:
            print(f"{datetime.now()}: Could not compute assortativity: {str(e)}")
        metrics['assortativity'] = None

    return metrics


def compute_degree_stats(
    graph: Union[nx.Graph, 'TaGraGraph']
) -> Dict[str, float]:
    """
    Compute degree distribution statistics.

    Parameters
    ----------
    graph : nx.Graph or TaGraGraph
        Graph to analyze

    Returns
    -------
    Dict[str, float]
        Statistics: min, max, mean, median, std
    """
    if hasattr(graph, 'to_networkx'):
        G = graph.to_networkx()
    else:
        G = graph

    degrees = [d for _, d in G.degree()]

    if not degrees:
        return {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0}

    return {
        'min': min(degrees),
        'max': max(degrees),
        'mean': np.mean(degrees),
        'median': np.median(degrees),
        'std': np.std(degrees)
    }


def compute_centrality_stats(
    graph: Union[nx.Graph, 'TaGraGraph'],
    centrality_type: str = 'degree',
    verbose: bool = True
) -> Dict[str, float]:
    """
    Compute centrality measure statistics.

    Parameters
    ----------
    graph : nx.Graph or TaGraGraph
        Graph to analyze
    centrality_type : str, default='degree'
        Type of centrality: 'degree', 'betweenness', 'closeness', 'eigenvector'
    verbose : bool, default=True
        Print progress messages

    Returns
    -------
    Dict[str, float]
        Statistics for the centrality measure
    """
    if hasattr(graph, 'to_networkx'):
        G = graph.to_networkx()
    else:
        G = graph

    if verbose:
        print(f"{datetime.now()}: Computing {centrality_type} centrality...")

    if centrality_type == 'degree':
        centrality = nx.degree_centrality(G)
    elif centrality_type == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif centrality_type == 'closeness':
        centrality = nx.closeness_centrality(G)
    elif centrality_type == 'eigenvector':
        try:
            centrality = nx.eigenvector_centrality(G, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            if verbose:
                print(f"{datetime.now()}: Eigenvector centrality did not converge")
            return {'min': None, 'max': None, 'mean': None, 'median': None, 'std': None}
    else:
        raise ValueError(f"Unknown centrality type: {centrality_type}")

    values = list(centrality.values())

    return {
        'min': min(values),
        'max': max(values),
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values)
    }
