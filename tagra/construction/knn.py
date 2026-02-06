"""
K-Nearest Neighbors graph constructor.

This module provides graph construction using K-nearest neighbors algorithm.
"""

from datetime import datetime
from typing import Optional
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

from .base import GraphConstructor


class KNNConstructor(GraphConstructor):
    """
    Graph constructor using K-Nearest Neighbors.

    Creates edges between each node and its k nearest neighbors based on
    Euclidean distance in the feature space.

    Parameters
    ----------
    k : int, default=5
        Number of nearest neighbors to connect
    verbose : bool, default=True
        Whether to print progress messages

    Examples
    --------
    >>> constructor = KNNConstructor(k=5)
    >>> constructor.construct(G, values)
    """

    def __init__(self, k: int = 5, verbose: bool = True):
        super().__init__(verbose=verbose)
        self.k = k

    @property
    def method_name(self) -> str:
        return "knn"

    def construct(
        self,
        G: nx.Graph,
        values: np.ndarray,
        k: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Add edges based on k-nearest neighbors.

        Parameters
        ----------
        G : nx.Graph
            NetworkX graph with nodes already added
        values : np.ndarray
            Numeric values array of shape (n_nodes, n_features)
        k : int, optional
            Override the default k value
        **kwargs
            Additional parameters (ignored)
        """
        self.validate_input(values)

        k_use = k if k is not None else self.k
        self.params = {'k': k_use, 'method': 'knn'}

        if self.verbose:
            print(f"{datetime.now()}: Building KNN graph with k={k_use}...")

        # Build KD-tree for efficient neighbor search
        tree = cKDTree(values)

        # For each node, find k nearest neighbors and add edges
        n_edges_added = 0
        for i in G.nodes():
            # Query k+1 neighbors (includes self)
            distances, indices = tree.query(values[i], k=k_use + 1)

            # Add edges to all neighbors except self (first one)
            for j in indices[1:]:
                if not G.has_edge(i, j):
                    G.add_edge(i, j)
                    n_edges_added += 1

        if self.verbose:
            print(f"{datetime.now()}: Added {n_edges_added} edges using KNN (k={k_use})")
