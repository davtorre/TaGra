"""
Graph visualization using matplotlib.

This module provides functions for visualizing graphs using matplotlib
with optional node coloring based on attributes.
"""

from datetime import datetime
from typing import Optional, Dict, Any, Union

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# Configure matplotlib defaults
plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 22,
    'axes.labelsize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 18,
    'figure.titlesize': 26
})


def matplotlib_graph_visualization(
    graph: Union[nx.Graph, 'TaGraGraph'],
    attribute: Optional[str] = None,
    outpath: Optional[str] = None,
    verbose: bool = True,
    palette: str = 'seismic',
    pos: Optional[Union[Dict[int, tuple], np.ndarray]] = None,
    node_size: int = 50,
    figsize: tuple = (10, 10),
    with_labels: bool = False
) -> Optional[plt.Figure]:
    """
    Visualize a graph using matplotlib.

    Creates a network visualization with optional node coloring based
    on a specified attribute.

    Parameters
    ----------
    graph : nx.Graph or TaGraGraph
        Graph to visualize
    attribute : str, optional
        Node attribute for coloring
    outpath : str, optional
        Path to save the figure
    verbose : bool, default=True
        Print progress messages
    palette : str, default='seismic'
        Matplotlib colormap name
    pos : dict or np.ndarray, optional
        Node positions. If None, uses spring layout.
    node_size : int, default=50
        Size of nodes
    figsize : tuple, default=(10, 10)
        Figure size
    with_labels : bool, default=False
        Whether to show node labels

    Returns
    -------
    Optional[plt.Figure]
        The figure object if not saved

    Examples
    --------
    >>> matplotlib_graph_visualization(G, 'label', 'graph.png')
    >>> matplotlib_graph_visualization(G, pos=manifold_positions)
    """
    NONE_STR = 'None'

    # Handle TaGraGraph
    if hasattr(graph, 'to_networkx'):
        G = graph.to_networkx()
        if pos is None and hasattr(graph, 'get_positions_dict'):
            pos = graph.get_positions_dict()
    else:
        G = graph

    fig = plt.figure(figsize=figsize)

    # Compute or convert positions
    if pos is None:
        pos = nx.spring_layout(G, seed=2112)
        title_string = "Graph of Relations"
    else:
        if isinstance(pos, np.ndarray):
            pos = {i: (pos[i, 0], pos[i, 1]) for i in range(len(pos))}
        title_string = "Graph of Relations with manifold learning"

    # Determine node colors
    cmap = plt.get_cmap(palette)

    if attribute is not None:
        y = np.array([G.nodes[node].get(attribute, NONE_STR) for node in G.nodes()])
        unique = np.unique(y)
        unique_to_int = {key: index for index, key in enumerate(unique)}
        color_array = [cmap(r) for r in np.linspace(0, 1, len(unique))]
        node_color = [color_array[unique_to_int[key]] for key in y]
    else:
        color = cmap(0)
        node_color = [color for _ in G.nodes()]

    # Draw the graph
    nx.draw(
        G, pos,
        with_labels=with_labels,
        node_size=node_size,
        font_color="white",
        font_size=10,
        node_color=node_color
    )
    plt.title(title_string)

    # Save or return
    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        if verbose:
            print(f'{datetime.now()}: Graph saved in {outpath}')
        plt.close()
        return None
    else:
        return fig


def create_graph_legend(
    attribute_values: list,
    palette: str = 'seismic',
    title: Optional[str] = None
) -> plt.Figure:
    """
    Create a standalone legend for graph visualization.

    Parameters
    ----------
    attribute_values : list
        List of unique attribute values
    palette : str, default='seismic'
        Matplotlib colormap name
    title : str, optional
        Legend title

    Returns
    -------
    plt.Figure
        Figure containing the legend
    """
    cmap = plt.get_cmap(palette)
    colors = [cmap(r) for r in np.linspace(0, 1, len(attribute_values))]

    fig, ax = plt.subplots(figsize=(3, len(attribute_values) * 0.5))
    ax.axis('off')

    for i, (value, color) in enumerate(zip(attribute_values, colors)):
        ax.scatter([], [], c=[color], label=str(value), s=100)

    ax.legend(loc='center', frameon=False, title=title)
    fig.tight_layout()

    return fig
