"""
TaGra IO Module.

This module provides unified file reading and writing functionality
for dataframes, graphs, and exports to various formats.

Examples
--------
Reading data:

>>> from tagra.io import read_dataframe, read_graph
>>> df = read_dataframe('data.csv')
>>> G = read_graph('graph.graphml')

Writing data:

>>> from tagra.io import save_dataframe, save_graph
>>> save_dataframe(df, 'output.csv')
>>> save_graph(G, 'output.pickle')

Exporting for visualization tools:

>>> from tagra.io import export_cytoscape, export_graphml
>>> export_cytoscape(G, 'graph.cyjs', positions=pos)
>>> export_graphml(G, 'graph.graphml')
"""

from .readers import (
    read_dataframe,
    read_graph,
    read_column_info,
    get_supported_formats
)

from .writers import (
    save_dataframe,
    save_graph,
    save_column_info,
    generate_timestamped_filename
)

from .exporters import (
    export_cytoscape,
    export_graphml,
    export_adjacency_matrix
)

__all__ = [
    # Readers
    'read_dataframe',
    'read_graph',
    'read_column_info',
    'get_supported_formats',
    # Writers
    'save_dataframe',
    'save_graph',
    'save_column_info',
    'generate_timestamped_filename',
    # Exporters
    'export_cytoscape',
    'export_graphml',
    'export_adjacency_matrix'
]
