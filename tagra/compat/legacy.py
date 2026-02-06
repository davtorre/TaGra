"""
Legacy compatibility wrappers.

This module provides wrapper functions that maintain backward compatibility
with the old TaGra API while delegating to the new modular implementation.
"""

import warnings
from typing import Optional, Union, List, Tuple, Any, Dict
import numpy as np
import pandas as pd
import networkx as nx


def _deprecation_warning(old_name: str, new_name: str, module: str = None):
    """Issue a deprecation warning."""
    if module:
        msg = (f"'{old_name}' is deprecated and will be removed in version 1.0. "
               f"Use 'from tagra.{module} import {new_name}' instead.")
    else:
        msg = f"'{old_name}' is deprecated and will be removed in version 1.0. Use '{new_name}' instead."
    warnings.warn(msg, DeprecationWarning, stacklevel=3)


# Legacy preprocessing function
def preprocess_dataframe(
    input_dataframe=None,
    output_directory="results/",
    preprocessed_filename=None,
    inferred_columns_filename=None,
    numeric_columns=[],
    categorical_columns=[],
    target_columns=[],
    unknown_column_action='infer',
    ignore_columns=[],
    numeric_threshold=0.05,
    numeric_scaling='standard',
    categorical_encoding='one-hot',
    nan_action='infer',
    nan_threshold=0.5,
    verbose=True,
    manifold_method='UMAP',
    manifold_dim=2,
    overwrite=False
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Legacy preprocessing function.

    .. deprecated:: 0.3.0
        Use `from tagra.preprocessing import preprocess` instead.
    """
    _deprecation_warning('preprocess_dataframe', 'preprocess', 'preprocessing')

    from ..preprocessing import preprocess
    return preprocess(
        input_dataframe=input_dataframe,
        output_directory=output_directory,
        preprocessed_filename=preprocessed_filename,
        inferred_columns_filename=inferred_columns_filename,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        target_columns=target_columns,
        unknown_column_action=unknown_column_action,
        ignore_columns=ignore_columns,
        numeric_threshold=numeric_threshold,
        numeric_scaling=numeric_scaling,
        categorical_encoding=categorical_encoding,
        nan_action=nan_action,
        nan_threshold=nan_threshold,
        verbose=verbose,
        manifold_method=manifold_method,
        manifold_dim=manifold_dim,
        overwrite=overwrite
    )


# Legacy graph creation function
def create_graph(
    input_dataframe: Optional[Union[str, pd.DataFrame]] = None,
    preprocessed_dataframe: Optional[Union[str, pd.DataFrame]] = None,
    inferred_columns_filename: Optional[str] = None,
    numeric_columns: Optional[List[str]] = None,
    output_directory: Optional[str] = None,
    graph_filename: Optional[str] = None,
    method: str = "knn",
    k: int = 5,
    distance_threshold: Optional[float] = None,
    similarity_threshold: Optional[float] = None,
    verbose: bool = True,
    overwrite: bool = False,
) -> nx.Graph:
    """
    Legacy graph creation function.

    .. deprecated:: 0.3.0
        Use `from tagra.construction import build_graph` instead.
    """
    _deprecation_warning('create_graph', 'build_graph', 'construction')

    from ..construction import build_graph
    result = build_graph(
        input_dataframe=input_dataframe,
        preprocessed_dataframe=preprocessed_dataframe,
        inferred_columns_filename=inferred_columns_filename,
        numeric_columns=numeric_columns,
        output_directory=output_directory,
        graph_filename=graph_filename,
        method=method,
        k=k,
        distance_threshold=distance_threshold,
        similarity_threshold=similarity_threshold,
        verbose=verbose,
        overwrite=overwrite
    )
    # Return the underlying NetworkX graph for backward compatibility
    return result.to_networkx()


# Legacy analysis function
def analyze_graph(
    graph: Union[nx.Graph, str],
    target_attributes=None,
    verbose=True,
    pos=None,
    output_directory=None,
    neigh_prob_filename=None,
    degree_distribution_filename=None,
    prob_heatmap_filename=None,
    community_filename=None,
    graph_visualization_filename=None,
    network_metrics_filename=None,
    overwrite=False
) -> Dict[str, Any]:
    """
    Legacy analysis function.

    .. deprecated:: 0.3.0
        Use `from tagra.analysis import analyze` instead.
    """
    _deprecation_warning('analyze_graph', 'analyze', 'analysis')

    from ..analysis import analyze
    return analyze(
        graph=graph,
        target_attributes=target_attributes,
        verbose=verbose,
        pos=pos,
        output_directory=output_directory,
        neigh_prob_filename=neigh_prob_filename,
        degree_distribution_filename=degree_distribution_filename,
        prob_heatmap_filename=prob_heatmap_filename,
        community_filename=community_filename,
        graph_visualization_filename=graph_visualization_filename,
        network_metrics_filename=network_metrics_filename,
        overwrite=overwrite
    )


# Legacy utility functions
def analyze_neighborhood_attributes(graph, target_attribute, return_probs=False):
    """
    Legacy neighborhood analysis function.

    .. deprecated:: 0.3.0
        Use `from tagra.analysis import analyze_neighborhoods` instead.
    """
    _deprecation_warning('analyze_neighborhood_attributes', 'analyze_neighborhoods', 'analysis')

    from ..analysis import analyze_neighborhoods
    return analyze_neighborhoods(graph, target_attribute, return_probs)


def print_neighbors_prob(df_neigh, label_col):
    """
    Legacy probability computation function.

    .. deprecated:: 0.3.0
        Use `from tagra.analysis import compute_neighborhood_probabilities` instead.
    """
    _deprecation_warning('print_neighbors_prob', 'compute_neighborhood_probabilities', 'analysis')

    from ..analysis import compute_neighborhood_probabilities
    return compute_neighborhood_probabilities(df_neigh, label_col)


def heat_map_prob(probabilities, df_neigh, label_col, prob_heatmap_path, verbose):
    """
    Legacy heatmap function.

    .. deprecated:: 0.3.0
        Use `from tagra.visualization import heat_map_prob` instead.
    """
    _deprecation_warning('heat_map_prob', 'heat_map_prob', 'visualization')

    from ..visualization import heat_map_prob as new_heat_map_prob
    return new_heat_map_prob(probabilities, df_neigh, label_col, prob_heatmap_path, verbose)


def plot_distribution(data_dict, outpath, verbose, bins=None, double_log=True):
    """
    Legacy distribution plot function.

    .. deprecated:: 0.3.0
        Use `from tagra.visualization import plot_distribution` instead.
    """
    _deprecation_warning('plot_distribution', 'plot_distribution', 'visualization')

    from ..visualization import plot_distribution as new_plot_distribution
    return new_plot_distribution(data_dict, outpath, verbose, bins, double_log)


def plot_community_composition(G, attribute_name, communities, outpath, verbose, palette='seismic'):
    """
    Legacy community composition plot function.

    .. deprecated:: 0.3.0
        Use `from tagra.visualization import plot_community_composition` instead.
    """
    _deprecation_warning('plot_community_composition', 'plot_community_composition', 'visualization')

    from ..visualization import plot_community_composition as new_plot
    return new_plot(G, attribute_name, communities, outpath, verbose, palette)


def matplotlib_graph_visualization(G, attribute, outpath, verbose, palette='seismic', pos=None):
    """
    Legacy graph visualization function.

    .. deprecated:: 0.3.0
        Use `from tagra.visualization import matplotlib_graph_visualization` instead.
    """
    _deprecation_warning('matplotlib_graph_visualization', 'matplotlib_graph_visualization', 'visualization')

    from ..visualization import matplotlib_graph_visualization as new_viz
    return new_viz(G, attribute, outpath, verbose, palette, pos)


def measure_mixing_matrix(G, communities):
    """
    Legacy mixing matrix function.

    .. deprecated:: 0.3.0
        Use `from tagra.analysis import measure_mixing_matrix` instead.
    """
    _deprecation_warning('measure_mixing_matrix', 'measure_mixing_matrix', 'analysis')

    from ..analysis import measure_mixing_matrix as new_measure
    return new_measure(G, communities)
