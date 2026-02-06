"""
Community composition visualizations.

This module provides functions for visualizing the composition
of detected communities.
"""

from datetime import datetime
from typing import Optional, List, Union, Dict, Any

import numpy as np
import matplotlib.pyplot as plt


def plot_community_composition(
    graph: Union['nx.Graph', 'TaGraGraph'],
    attribute_name: Optional[str],
    communities: List[List[int]],
    outpath: Optional[str] = None,
    verbose: bool = True,
    palette: str = 'seismic',
    figsize: tuple = (8, 6)
) -> Optional[plt.Figure]:
    """
    Plot the composition of communities by attribute.

    Creates a stacked bar chart showing how different attribute values
    are distributed across communities.

    Parameters
    ----------
    graph : nx.Graph or TaGraGraph
        Graph with node attributes
    attribute_name : str, optional
        Node attribute for composition analysis
    communities : List[List[int]]
        List of communities (each as list of node IDs)
    outpath : str, optional
        Path to save the figure
    verbose : bool, default=True
        Print progress messages
    palette : str, default='seismic'
        Matplotlib colormap name
    figsize : tuple, default=(8, 6)
        Figure size

    Returns
    -------
    Optional[plt.Figure]
        Figure object if not saved

    Examples
    --------
    >>> plot_community_composition(G, 'label', communities, 'communities.png')
    """
    import networkx as nx

    NONE_STR = 'None'

    if hasattr(graph, 'to_networkx'):
        G = graph.to_networkx()
    else:
        G = graph

    # Get unique labels
    if attribute_name is not None:
        labels_per_node = [G.nodes[n].get(attribute_name, NONE_STR) for n in G.nodes]
        unique_labels = set(labels_per_node)
    else:
        labels_per_node = [0 for _ in G.nodes()]
        unique_labels = {0}

    # Compute community compositions
    community_compositions = {}
    cmap = plt.get_cmap(palette)

    if attribute_name is not None:
        for comm_id, community in enumerate(communities):
            if len(community) <= 1:
                continue
            labels_community = [G.nodes[node].get(attribute_name, NONE_STR) for node in community]
            community_compositions[comm_id] = {label: 0 for label in unique_labels}
            measured_labels, counts = np.unique(labels_community, return_counts=True)
            for label, count in zip(measured_labels, counts):
                community_compositions[comm_id][label] = count
    else:
        for comm_id, community in enumerate(communities):
            labels_community = [0 for _ in community]
            labels, counts = np.unique(labels_community, return_counts=True)
            community_compositions[comm_id] = {label: count for label, count in zip(labels, counts)}

    if len(community_compositions) == 0:
        if verbose:
            print(f"{datetime.now()}: No communities with more than 1 node found")
        return None

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    indices = list(community_compositions.keys())
    bar_width = 0.9
    bottoms = [0] * len(indices)
    colors = {label: cmap(i) for label, i in zip(unique_labels, np.linspace(0, 1, len(unique_labels)))}

    # Plot stacked bars
    for label in unique_labels:
        values = [community_compositions[idx].get(label, 0) for idx in indices]
        ax.bar(indices, values, bar_width,
               label=f"{attribute_name}={label}" if attribute_name else str(label),
               bottom=bottoms, color=colors[label])
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    # Configure axes
    y_max = max(max(bottoms) * 1.1, 1)
    ax.set_xticks([])
    ax.set_xlabel('Community ID')
    ax.set_ylabel('Counts')
    ax.set_title('Counts of outcomes by community ID')
    ax.set_ylim([0, y_max])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout()

    # Save or return
    if outpath:
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"{datetime.now()}: Community composition saved in {outpath}")
        plt.close()
        return None
    else:
        return fig


def plot_community_sizes(
    communities: List[List[int]],
    outpath: Optional[str] = None,
    verbose: bool = True,
    figsize: tuple = (8, 6),
    color: str = 'steelblue'
) -> Optional[plt.Figure]:
    """
    Plot community size distribution.

    Parameters
    ----------
    communities : List[List[int]]
        List of communities
    outpath : str, optional
        Path to save the figure
    verbose : bool, default=True
        Print progress messages
    figsize : tuple, default=(8, 6)
        Figure size
    color : str, default='steelblue'
        Bar color

    Returns
    -------
    Optional[plt.Figure]
        Figure object if not saved
    """
    sizes = [len(c) for c in communities]

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(range(len(sizes)), sizes, color=color, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Community Index')
    ax.set_ylabel('Size (# nodes)')
    ax.set_title('Community Size Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"{datetime.now()}: Community sizes saved in {outpath}")
        plt.close()
        return None
    else:
        return fig
