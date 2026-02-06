"""
TaGra Visualization Module.

This module provides visualization functions for graphs, heatmaps,
distributions, and community composition.

Examples
--------
Graph visualization:

>>> from tagra.visualization import matplotlib_graph_visualization
>>> matplotlib_graph_visualization(G, 'label', 'graph.png')

Probability heatmap:

>>> from tagra.visualization import heat_map_prob
>>> heat_map_prob(probs, df_neigh, 'label', 'heatmap.png')

Cytoscape visualization:

>>> from tagra.visualization import CytoscapeVisualizer
>>> viz = CytoscapeVisualizer(G, target_attribute='label')
>>> viz.show()
"""

from .graph_plot import matplotlib_graph_visualization, create_graph_legend
from .heatmap import heat_map_prob, correlation_heatmap
from .distribution import plot_distribution, plot_degree_distribution, plot_histogram
from .community_plot import plot_community_composition, plot_community_sizes

# Cytoscape visualization (optional dependency)
try:
    from .cytoscape import (
        CytoscapeVisualizer,
        visualize_cytoscape,
        cytoscape_graph_visualization,
        IPYCYTOSCAPE_AVAILABLE
    )
    CYTOSCAPE_AVAILABLE = True
except ImportError:
    CYTOSCAPE_AVAILABLE = False
    CytoscapeVisualizer = None
    visualize_cytoscape = None
    cytoscape_graph_visualization = None
    IPYCYTOSCAPE_AVAILABLE = False

__all__ = [
    # Graph plotting
    'matplotlib_graph_visualization',
    'create_graph_legend',
    # Heatmaps
    'heat_map_prob',
    'correlation_heatmap',
    # Distributions
    'plot_distribution',
    'plot_degree_distribution',
    'plot_histogram',
    # Community plots
    'plot_community_composition',
    'plot_community_sizes',
    # Cytoscape
    'CytoscapeVisualizer',
    'visualize_cytoscape',
    'cytoscape_graph_visualization',
    'CYTOSCAPE_AVAILABLE',
    'IPYCYTOSCAPE_AVAILABLE'
]
