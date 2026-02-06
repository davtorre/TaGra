"""
Cytoscape visualization module for TaGra.

This module provides interactive graph visualization using Cytoscape.js
via ipycytoscape (for Jupyter notebooks) and export functionality for
Cytoscape Desktop.

This is a re-export of the functionality from tagra.cytoscape_vis
organized within the visualization subpackage.
"""

# Re-export from the original module
from ..cytoscape_vis import (
    CytoscapeVisualizer,
    visualize_cytoscape,
    cytoscape_graph_visualization,
    IPYCYTOSCAPE_AVAILABLE,
    DEFAULT_PALETTE,
    DEFAULT_STYLESHEET
)

__all__ = [
    'CytoscapeVisualizer',
    'visualize_cytoscape',
    'cytoscape_graph_visualization',
    'IPYCYTOSCAPE_AVAILABLE',
    'DEFAULT_PALETTE',
    'DEFAULT_STYLESHEET'
]
