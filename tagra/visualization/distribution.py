"""
Distribution plots for TaGra.

This module provides functions for plotting degree distributions
and other statistical distributions.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Union

import numpy as np
import matplotlib.pyplot as plt


def plot_distribution(
    data_dict: Dict[str, Any],
    outpath: Optional[str] = None,
    verbose: bool = True,
    bins: Optional[Union[int, List[int]]] = None,
    double_log: bool = True,
    figsize: tuple = (8, 6)
) -> Optional[plt.Figure]:
    """
    Plot a distribution histogram or scatter plot.

    Parameters
    ----------
    data_dict : Dict[str, Any]
        Dictionary containing:
        - 'data': List of values
        - 'title': Plot title
        - 'xlabel': X-axis label
        - 'ylabel': Y-axis label
    outpath : str, optional
        Path to save the figure
    verbose : bool, default=True
        Print progress messages
    bins : int or list, optional
        Bins for histogram. If None, uses range(0, max(data))
    double_log : bool, default=True
        Use log-log scale
    figsize : tuple, default=(8, 6)
        Figure size

    Returns
    -------
    Optional[plt.Figure]
        Figure object if not saved

    Examples
    --------
    >>> data = {'data': degrees, 'title': 'Degree Distribution',
    ...         'xlabel': 'Degree', 'ylabel': 'Count'}
    >>> plot_distribution(data, 'degree.png')
    """
    data = data_dict['data']

    if not data:
        if verbose:
            print(f"{datetime.now()}: No data to plot")
        return None

    if bins is None:
        bins = range(0, max(data) + 1)

    hist, bin_edges = np.histogram(data, bins=bins)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(bin_edges[:-1], hist, alpha=0.75, edgecolor='black')

    ax.set_title(data_dict.get('title', 'Distribution'))
    ax.set_xlabel(data_dict.get('xlabel', 'Value'))
    ax.set_ylabel(data_dict.get('ylabel', 'Count'))

    if double_log:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim((1, None))
        ax.set_ylim((1, None))

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Save or return
    if outpath:
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"{datetime.now()}: {data_dict.get('title', 'Distribution')} saved in {outpath}")
        plt.close()
        return None
    else:
        return fig


def plot_degree_distribution(
    graph: Union['nx.Graph', 'TaGraGraph'],
    outpath: Optional[str] = None,
    verbose: bool = True,
    double_log: bool = True,
    figsize: tuple = (8, 6)
) -> Optional[plt.Figure]:
    """
    Plot the degree distribution of a graph.

    Parameters
    ----------
    graph : nx.Graph or TaGraGraph
        Graph to analyze
    outpath : str, optional
        Path to save the figure
    verbose : bool, default=True
        Print progress messages
    double_log : bool, default=True
        Use log-log scale
    figsize : tuple, default=(8, 6)
        Figure size

    Returns
    -------
    Optional[plt.Figure]
        Figure object if not saved
    """
    import networkx as nx

    if hasattr(graph, 'to_networkx'):
        G = graph.to_networkx()
    else:
        G = graph

    degrees = [d for _, d in G.degree()]

    data_dict = {
        'data': degrees,
        'title': 'Degree Distribution',
        'xlabel': 'Degree',
        'ylabel': 'Number of Nodes'
    }

    return plot_distribution(data_dict, outpath, verbose, double_log=double_log, figsize=figsize)


def plot_histogram(
    data: List[float],
    title: str = 'Histogram',
    xlabel: str = 'Value',
    ylabel: str = 'Frequency',
    outpath: Optional[str] = None,
    verbose: bool = True,
    bins: int = 30,
    figsize: tuple = (8, 6),
    color: str = 'steelblue',
    alpha: float = 0.7
) -> Optional[plt.Figure]:
    """
    Create a simple histogram.

    Parameters
    ----------
    data : List[float]
        Data to plot
    title : str, default='Histogram'
        Plot title
    xlabel : str, default='Value'
        X-axis label
    ylabel : str, default='Frequency'
        Y-axis label
    outpath : str, optional
        Path to save the figure
    verbose : bool, default=True
        Print progress messages
    bins : int, default=30
        Number of bins
    figsize : tuple, default=(8, 6)
        Figure size
    color : str, default='steelblue'
        Bar color
    alpha : float, default=0.7
        Transparency

    Returns
    -------
    Optional[plt.Figure]
        Figure object if not saved
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(data, bins=bins, color=color, alpha=alpha, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if outpath:
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"{datetime.now()}: {title} saved in {outpath}")
        plt.close()
        return None
    else:
        return fig
