"""
TaGra Backward Compatibility Module.

This module provides compatibility with the pre-0.3.0 API.
All functions in this module issue deprecation warnings and
delegate to the new modular implementation.

Note
----
This module will be removed in version 1.0.0.
"""

from .legacy import (
    preprocess_dataframe,
    create_graph,
    analyze_graph,
    analyze_neighborhood_attributes,
    print_neighbors_prob,
    heat_map_prob,
    plot_distribution,
    plot_community_composition,
    matplotlib_graph_visualization,
    measure_mixing_matrix
)

__all__ = [
    'preprocess_dataframe',
    'create_graph',
    'analyze_graph',
    'analyze_neighborhood_attributes',
    'print_neighbors_prob',
    'heat_map_prob',
    'plot_distribution',
    'plot_community_composition',
    'matplotlib_graph_visualization',
    'measure_mixing_matrix'
]
