"""
Heatmap visualizations for TaGra.

This module provides heatmap visualizations for neighborhood
probability matrices and other grid-based data.
"""

from datetime import datetime
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def heat_map_prob(
    probabilities: Dict[Tuple, float],
    df_neigh: pd.DataFrame,
    label_col: str,
    prob_heatmap_path: Optional[str] = None,
    verbose: bool = True,
    cmap: str = 'seismic',
    figsize: tuple = (8, 6),
    vmin: float = 0,
    vmax: float = 1
) -> Optional[plt.Figure]:
    """
    Create a heatmap of neighborhood probabilities.

    Visualizes P(j|i) - the probability that a neighbor of a node
    with label i has label j.

    Parameters
    ----------
    probabilities : Dict[Tuple, float]
        Dictionary mapping (i, j) to P(j|i)
    df_neigh : pd.DataFrame
        Neighborhood analysis dataframe
    label_col : str
        Name of the label column
    prob_heatmap_path : str, optional
        Path to save the heatmap
    verbose : bool, default=True
        Print progress messages
    cmap : str, default='seismic'
        Colormap name
    figsize : tuple, default=(8, 6)
        Figure size
    vmin : float, default=0
        Minimum value for colormap
    vmax : float, default=1
        Maximum value for colormap

    Returns
    -------
    Optional[plt.Figure]
        Figure object if not saved

    Examples
    --------
    >>> heat_map_prob(probs, df_neigh, 'label', 'heatmap.png')
    """
    # Get labels and convert to string
    labels = sorted([str(label) for label in df_neigh[f'node_{label_col}'].unique()])
    probabilities = {(str(k[0]), str(k[1])): v for k, v in probabilities.items()}

    # Build probability matrix
    prob_matrix = pd.DataFrame(index=labels, columns=labels, data=0.0)
    for (i, j), prob in probabilities.items():
        if i in prob_matrix.index and j in prob_matrix.columns:
            prob_matrix.loc[i, j] = prob

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(prob_matrix.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(cax)

    # Add text annotations
    text_size = max(5, 36 - len(labels))
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = float(prob_matrix.iloc[i, j])
            # Choose text color based on background
            color = 'w' if value < 0.35 or value > 0.65 else 'black'
            ax.text(j, i, f"{value:.2f}",
                   ha="center", va="center",
                   color=color, fontsize=text_size)

    # Configure axes
    ax.set_title('Probability Distribution Heatmap')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xlabel('Label j')
    plt.ylabel('Label i')

    # Save or return
    if prob_heatmap_path:
        plt.savefig(prob_heatmap_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"{datetime.now()}: Probability heatmap saved in {prob_heatmap_path}")
        plt.close()
        return None
    else:
        return fig


def correlation_heatmap(
    df: pd.DataFrame,
    outpath: Optional[str] = None,
    verbose: bool = True,
    cmap: str = 'coolwarm',
    figsize: tuple = (10, 8),
    method: str = 'pearson'
) -> Optional[plt.Figure]:
    """
    Create a correlation heatmap for dataframe columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with numeric columns
    outpath : str, optional
        Path to save the heatmap
    verbose : bool, default=True
        Print progress messages
    cmap : str, default='coolwarm'
        Colormap name
    figsize : tuple, default=(10, 8)
        Figure size
    method : str, default='pearson'
        Correlation method

    Returns
    -------
    Optional[plt.Figure]
        Figure object if not saved
    """
    # Compute correlation matrix
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr(method=method)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(corr, cmap=cmap, vmin=-1, vmax=1)
    fig.colorbar(cax)

    # Configure axes
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='left')
    ax.set_yticklabels(corr.columns)
    ax.set_title('Correlation Matrix')

    plt.tight_layout()

    # Save or return
    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"{datetime.now()}: Correlation heatmap saved in {outpath}")
        plt.close()
        return None
    else:
        return fig
