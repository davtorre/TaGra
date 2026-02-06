"""
TaGra Analysis Module.

This module provides comprehensive graph analysis functionality including
basic metrics, neighborhood analysis, community detection, and reporting.

Examples
--------
Basic usage with the convenience function:

>>> from tagra.analysis import analyze
>>> metrics = analyze(graph, target_attribute='label')

Using individual functions:

>>> from tagra.analysis import compute_metrics, detect_communities
>>> metrics = compute_metrics(G)
>>> communities = detect_communities(G)

Computing homophily:

>>> from tagra.analysis import compute_homophily
>>> stats = compute_homophily(G, 'label')
>>> print(f"Homophily: {stats['homophily_score']:.4f}")
"""

import os
import pickle
from datetime import datetime
from typing import Dict, Any, Optional, Union, List

import networkx as nx
import numpy as np
import pandas as pd

from .metrics import compute_metrics, compute_degree_stats, compute_centrality_stats
from .neighborhood import (
    analyze_neighborhoods,
    compute_neighborhood_probabilities,
    compute_homophily,
    compute_chi_square
)
from .community import (
    detect_communities,
    compute_modularity,
    compute_community_stats,
    measure_mixing_matrix
)
from .report import generate_report, format_metrics_dict


def analyze(
    graph: Union[nx.Graph, str, 'TaGraGraph'],
    target_attributes: Optional[Union[str, List[str]]] = None,
    verbose: bool = True,
    pos: Optional[np.ndarray] = None,
    output_directory: Optional[str] = None,
    neigh_prob_filename: Optional[str] = None,
    degree_distribution_filename: Optional[str] = None,
    prob_heatmap_filename: Optional[str] = None,
    community_filename: Optional[str] = None,
    graph_visualization_filename: Optional[str] = None,
    network_metrics_filename: Optional[str] = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive graph analysis.

    This is the main entry point for graph analysis in TaGra. It computes
    various metrics, performs community detection, and optionally generates
    visualizations.

    Parameters
    ----------
    graph : nx.Graph, str, or TaGraGraph
        Graph to analyze (or path to pickled graph)
    target_attributes : str or List[str], optional
        Target attribute(s) for neighborhood analysis
    verbose : bool, default=True
        Print progress messages
    pos : np.ndarray, optional
        Node positions for visualization
    output_directory : str, optional
        Directory for output files
    neigh_prob_filename : str, optional
        Filename for neighborhood probabilities
    degree_distribution_filename : str, optional
        Filename for degree distribution plot
    prob_heatmap_filename : str, optional
        Filename for probability heatmap
    community_filename : str, optional
        Filename for community composition plot
    graph_visualization_filename : str, optional
        Filename for graph visualization
    network_metrics_filename : str, optional
        Filename for metrics report
    overwrite : bool, default=False
        Overwrite existing files

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all computed metrics

    Examples
    --------
    >>> metrics = analyze(G, target_attributes='label')
    >>> print(f"Homophily: {metrics['homophily_score']:.4f}")
    """
    # Handle file path input
    if isinstance(graph, str):
        if verbose:
            print(f"{datetime.now()}: Loading graph from file: {graph}")
        with open(graph, 'rb') as f:
            G = pickle.load(f)
    elif hasattr(graph, 'to_networkx'):
        G = graph.to_networkx()
    else:
        G = graph

    # Handle target attributes
    if target_attributes is not None and isinstance(target_attributes, list) and len(target_attributes) > 0:
        target_attributes = str(tuple(target_attributes))

    # Setup output paths
    time_str = datetime.now().strftime('%Y%m%d%H%M')
    if output_directory is None:
        output_directory = './'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    paths = _configure_paths(
        output_directory, time_str, overwrite,
        neigh_prob_filename, degree_distribution_filename,
        prob_heatmap_filename, community_filename,
        graph_visualization_filename, network_metrics_filename
    )

    if verbose:
        print(f"{datetime.now()}: Starting graph analysis...")
        print(f"{datetime.now()}: Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Compute basic metrics
    metrics = compute_metrics(G, verbose=verbose)

    # Community detection
    try:
        communities = detect_communities(G, verbose=verbose)
        metrics['community_count'] = len(communities)
        if len(communities) > 1:
            metrics['modularity'] = compute_modularity(G, communities, verbose=verbose)
        else:
            metrics['modularity'] = 0
    except Exception as e:
        if verbose:
            print(f"{datetime.now()}: Community detection failed: {str(e)}")
        communities = []
        metrics['community_count'] = 0
        metrics['modularity'] = None

    # Target attribute analysis
    if target_attributes is not None:
        if verbose:
            print(f"{datetime.now()}: Analyzing target attribute: {target_attributes}")

        try:
            df_neigh = analyze_neighborhoods(G, target_attributes, verbose=verbose)

            # Chi-square test
            chi_results = compute_chi_square(df_neigh, target_attributes, verbose=verbose)
            metrics.update(chi_results)

            # Homophily
            homophily_results = compute_homophily(G, target_attributes, verbose=verbose)
            metrics.update(homophily_results)

            # Compute and save neighborhood probabilities
            probabilities = compute_neighborhood_probabilities(df_neigh, target_attributes)

            if verbose:
                print(f"{datetime.now()}: Neighborhood probability results:")
                for (i, j), prob in probabilities.items():
                    print(f"P({j}|{i}) = {prob:.4f}")

            if paths['neigh_prob_filename']:
                with open(paths['neigh_prob_filename'], 'w') as fp:
                    for (i, j), prob in probabilities.items():
                        fp.write(f"P({j}|{i}) = {prob:.4f}\n")
                if verbose:
                    print(f"{datetime.now()}: Saved to {paths['neigh_prob_filename']}")

            # Generate visualizations (import here to avoid circular imports)
            if paths['prob_heatmap_filename']:
                try:
                    from ..visualization import heat_map_prob
                    heat_map_prob(probabilities, df_neigh, target_attributes,
                                 paths['prob_heatmap_filename'], verbose)
                except ImportError:
                    pass

        except Exception as e:
            if verbose:
                print(f"{datetime.now()}: Target analysis failed: {str(e)}")
            metrics['chi2_stat'] = None
            metrics['chi2_p_value'] = None
            metrics['homophily_score'] = None
            metrics['homophily_p_value'] = None
            metrics['homophily_z_score'] = None

    # Generate other visualizations
    if paths['degree_distribution_filename']:
        try:
            from ..visualization import plot_distribution
            degree_data = {
                'data': [d for _, d in G.degree()],
                'title': 'Degree distribution',
                'xlabel': 'Degree',
                'ylabel': 'Number of Nodes'
            }
            plot_distribution(degree_data, paths['degree_distribution_filename'], verbose)
        except ImportError:
            pass

    if paths['community_filename'] and communities:
        try:
            from ..visualization import plot_community_composition
            plot_community_composition(G, target_attributes, communities,
                                      paths['community_filename'], verbose)
        except ImportError:
            pass

    if paths['graph_visualization_filename']:
        try:
            from ..visualization import matplotlib_graph_visualization
            matplotlib_graph_visualization(G, target_attributes,
                                          paths['graph_visualization_filename'],
                                          verbose, pos=pos)
        except ImportError:
            pass

    # Generate report
    if paths['network_metrics_filename']:
        report = generate_report(metrics, target_attributes, communities,
                                paths['network_metrics_filename'], verbose)
    elif verbose:
        report = generate_report(metrics, target_attributes, communities)
        print(report)

    if verbose:
        print(f"{datetime.now()}: Graph analysis complete.")

    return metrics


def _configure_paths(
    output_directory: str,
    time_str: str,
    overwrite: bool,
    neigh_prob_filename: Optional[str],
    degree_distribution_filename: Optional[str],
    prob_heatmap_filename: Optional[str],
    community_filename: Optional[str],
    graph_visualization_filename: Optional[str],
    network_metrics_filename: Optional[str]
) -> Dict[str, Optional[str]]:
    """Configure output paths with timestamps."""
    paths = {}
    filenames = [
        ('neigh_prob_filename', neigh_prob_filename),
        ('degree_distribution_filename', degree_distribution_filename),
        ('prob_heatmap_filename', prob_heatmap_filename),
        ('community_filename', community_filename),
        ('graph_visualization_filename', graph_visualization_filename),
        ('network_metrics_filename', network_metrics_filename)
    ]

    for key, value in filenames:
        if value is not None:
            if not overwrite:
                base, ext = os.path.splitext(value)
                value = f"{base}_{time_str}{ext}"
            paths[key] = os.path.join(output_directory, value)
        else:
            paths[key] = None

    return paths


__all__ = [
    'analyze',
    'compute_metrics',
    'compute_degree_stats',
    'compute_centrality_stats',
    'analyze_neighborhoods',
    'compute_neighborhood_probabilities',
    'compute_homophily',
    'compute_chi_square',
    'detect_communities',
    'compute_modularity',
    'compute_community_stats',
    'measure_mixing_matrix',
    'generate_report',
    'format_metrics_dict'
]
