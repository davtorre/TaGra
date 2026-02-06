"""
Distance threshold graph constructor.

This module provides graph construction using a distance threshold.
"""

from datetime import datetime
from typing import Optional
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

from .base import GraphConstructor


class DistanceThresholdConstructor(GraphConstructor):
    """
    Graph constructor using distance threshold.

    Creates edges between nodes that are within a specified Euclidean
    distance of each other in the feature space.

    Parameters
    ----------
    distance_threshold : float
        Maximum distance for connecting two nodes
    verbose : bool, default=True
        Whether to print progress messages

    Examples
    --------
    >>> constructor = DistanceThresholdConstructor(distance_threshold=1.0)
    >>> constructor.construct(G, values)
    """

    def __init__(
        self,
        distance_threshold: float,
        verbose: bool = True
    ):
        super().__init__(verbose=verbose)
        self.distance_threshold = distance_threshold

    @property
    def method_name(self) -> str:
        return "distance"

    def construct(
        self,
        G: nx.Graph,
        values: np.ndarray,
        distance_threshold: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Add edges based on distance threshold.

        Parameters
        ----------
        G : nx.Graph
            NetworkX graph with nodes already added
        values : np.ndarray
            Numeric values array of shape (n_nodes, n_features)
        distance_threshold : float, optional
            Override the default distance threshold
        **kwargs
            Additional parameters (ignored)
        """
        self.validate_input(values)

        threshold = distance_threshold if distance_threshold is not None else self.distance_threshold
        self.params = {'distance_threshold': threshold, 'method': 'distance'}

        if threshold is None or threshold <= 0:
            raise ValueError("distance_threshold must be a positive number")

        if self.verbose:
            print(f"{datetime.now()}: Building distance graph with threshold={threshold}...")

        # Build KD-tree and find all pairs within threshold
        tree = cKDTree(values)
        pairs = tree.query_pairs(threshold)

        # Add all edges
        n_edges_added = 0
        for i, j in pairs:
            if not G.has_edge(i, j):
                G.add_edge(i, j)
                n_edges_added += 1

        if self.verbose:
            print(f"{datetime.now()}: Added {n_edges_added} edges using distance threshold={threshold}")
