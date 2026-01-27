# TaGra - Table to Graph
# Generate graphs from tabular data through manifold learning

from tagra.preprocessing import preprocess_dataframe
from tagra.graph import create_graph
from tagra.analysis import analyze_graph
from tagra.config import load_config

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

__version__ = "0.2.4"

__all__ = [
    'preprocess_dataframe',
    'create_graph',
    'analyze_graph',
    'load_config',
    'CytoscapeVisualizer',
    'visualize_cytoscape',
    'cytoscape_graph_visualization',
    'CYTOSCAPE_AVAILABLE',
    '__version__'
]
