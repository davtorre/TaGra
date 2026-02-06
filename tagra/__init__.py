# TaGra - Table to Graph
# Generate graphs from tabular data through manifold learning

"""
TaGra: Table to Graph

A library for converting tabular data to graphs using manifold learning
and analyzing the resulting graph structures.

Quick Start
-----------
>>> import tagra
>>>
>>> # New API (recommended)
>>> from tagra.preprocessing import preprocess
>>> from tagra.construction import build_graph
>>> from tagra.analysis import analyze
>>>
>>> df_processed, positions = preprocess(df, target_columns=['label'])
>>> graph = build_graph(df, df_processed, method='knn', k=5)
>>> metrics = analyze(graph, target_attributes='label')

>>> # Convenience function
>>> graph = tagra.from_dataframe(df, target='label', method='knn', k=5)

Legacy API (deprecated)
-----------------------
>>> from tagra import preprocess_dataframe, create_graph, analyze_graph
>>> # These still work but will show deprecation warnings

Modules
-------
- tagra.core: Core types and TaGraGraph class
- tagra.preprocessing: Data preprocessing pipeline
- tagra.construction: Graph construction methods
- tagra.analysis: Graph analysis and metrics
- tagra.visualization: Plotting and visualization
- tagra.ml: Machine learning on graphs
- tagra.io: File I/O utilities
"""

__version__ = "0.3.0"

# Core types
from tagra.core import (
    TaGraGraph,
    GraphMetadata,
    ScalingMethod,
    EncodingMethod,
    ConstructionMethod,
    ManifoldMethod
)

# Exceptions
from tagra.exceptions import (
    TaGraError,
    ConfigurationError,
    PreprocessingError,
    GraphConstructionError,
    AnalysisError
)

# Configuration
from tagra.config import load_config, save_config

# Legacy API (with deprecation warnings)
# These delegate to the new implementation
from tagra.compat import (
    preprocess_dataframe,
    create_graph,
    analyze_graph
)

# Optional Cytoscape visualization (requires ipycytoscape)
try:
    from tagra.cytoscape_vis import (
        CytoscapeVisualizer,
        visualize_cytoscape,
        cytoscape_graph_visualization
    )
    CYTOSCAPE_AVAILABLE = True
except ImportError:
    CYTOSCAPE_AVAILABLE = False
    CytoscapeVisualizer = None
    visualize_cytoscape = None
    cytoscape_graph_visualization = None


# New convenience functions
def from_dataframe(
    df,
    target=None,
    method='knn',
    k=5,
    distance_threshold=None,
    similarity_threshold=None,
    scaling='standard',
    encoding='one-hot',
    manifold_method='UMAP',
    manifold_dim=2,
    verbose=True,
    **kwargs
):
    """
    Create a TaGraGraph from a pandas DataFrame.

    This is the recommended entry point for the new TaGra API.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target : str or list, optional
        Target column(s) for analysis
    method : str, default='knn'
        Graph construction method: 'knn', 'distance', 'similarity'
    k : int, default=5
        Number of neighbors for KNN
    distance_threshold : float, optional
        Threshold for distance method
    similarity_threshold : float, optional
        Threshold for similarity method
    scaling : str, default='standard'
        Numeric scaling: 'standard', 'minmax'
    encoding : str, default='one-hot'
        Categorical encoding: 'one-hot', 'label'
    manifold_method : str, default='UMAP'
        Manifold method: 'UMAP', 'TSNE', 'Isomap'
    manifold_dim : int, default=2
        Manifold dimensions
    verbose : bool, default=True
        Print progress
    **kwargs
        Additional arguments

    Returns
    -------
    TaGraGraph
        The constructed graph with metadata

    Examples
    --------
    >>> import tagra
    >>> graph = tagra.from_dataframe(df, target='label', method='knn', k=5)
    >>> print(f"Graph has {graph.number_of_nodes()} nodes")
    """
    from tagra.preprocessing import preprocess
    from tagra.construction import build_graph

    # Preprocess
    target_columns = [target] if isinstance(target, str) else target
    df_processed, positions = preprocess(
        input_dataframe=df,
        target_columns=target_columns,
        numeric_scaling=scaling,
        categorical_encoding=encoding,
        manifold_method=manifold_method,
        manifold_dim=manifold_dim,
        verbose=verbose,
        **kwargs
    )

    # Build graph
    graph = build_graph(
        input_dataframe=df,
        preprocessed_dataframe=df_processed,
        method=method,
        k=k,
        distance_threshold=distance_threshold,
        similarity_threshold=similarity_threshold,
        verbose=verbose
    )

    # Store manifold positions
    graph.manifold_positions = positions

    return graph


def analyze(graph, target_attribute=None, **kwargs):
    """
    Analyze a TaGra graph.

    Parameters
    ----------
    graph : TaGraGraph or nx.Graph
        Graph to analyze
    target_attribute : str, optional
        Target attribute for homophily analysis
    **kwargs
        Additional arguments passed to analyze()

    Returns
    -------
    dict
        Dictionary of computed metrics

    Examples
    --------
    >>> metrics = tagra.analyze(graph, target_attribute='label')
    >>> print(f"Homophily: {metrics['homophily_score']:.4f}")
    """
    from tagra.analysis import analyze as _analyze
    return _analyze(graph, target_attributes=target_attribute, **kwargs)


def visualize(graph, output=None, method='matplotlib', **kwargs):
    """
    Visualize a TaGra graph.

    Parameters
    ----------
    graph : TaGraGraph or nx.Graph
        Graph to visualize
    output : str, optional
        Output file path
    method : str, default='matplotlib'
        Visualization method: 'matplotlib', 'cytoscape'
    **kwargs
        Additional arguments

    Examples
    --------
    >>> tagra.visualize(graph, output='graph.png')
    >>> tagra.visualize(graph, method='cytoscape')  # In Jupyter
    """
    if method == 'matplotlib':
        from tagra.visualization import matplotlib_graph_visualization
        pos = graph.get_positions_dict() if hasattr(graph, 'get_positions_dict') else None
        target = kwargs.pop('target_attribute', None)
        matplotlib_graph_visualization(graph, target, output, pos=pos, **kwargs)
    elif method == 'cytoscape':
        if not CYTOSCAPE_AVAILABLE:
            raise ImportError("ipycytoscape required for Cytoscape visualization")
        pos = graph.get_positions_dict() if hasattr(graph, 'get_positions_dict') else None
        return visualize_cytoscape(graph, pos=pos, output_path=output, **kwargs)


__all__ = [
    # Version
    '__version__',
    # Core types
    'TaGraGraph',
    'GraphMetadata',
    'ScalingMethod',
    'EncodingMethod',
    'ConstructionMethod',
    'ManifoldMethod',
    # Exceptions
    'TaGraError',
    'ConfigurationError',
    'PreprocessingError',
    'GraphConstructionError',
    'AnalysisError',
    # Configuration
    'load_config',
    'save_config',
    # New API
    'from_dataframe',
    'analyze',
    'visualize',
    # Legacy API (deprecated but still working)
    'preprocess_dataframe',
    'create_graph',
    'analyze_graph',
    # Cytoscape
    'CytoscapeVisualizer',
    'visualize_cytoscape',
    'cytoscape_graph_visualization',
    'CYTOSCAPE_AVAILABLE'
]
