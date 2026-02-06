"""
Similarity threshold graph constructor.

This module provides graph construction using cosine similarity threshold.
"""

from datetime import datetime
from typing import Optional
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from .base import GraphConstructor


class SimilarityThresholdConstructor(GraphConstructor):
    """
    Graph constructor using cosine similarity threshold.

    Creates edges between nodes whose cosine similarity in the feature
    space exceeds a specified threshold.

    Parameters
    ----------
    similarity_threshold : float
        Minimum cosine similarity for connecting two nodes (0 to 1)
    verbose : bool, default=True
        Whether to print progress messages

    Examples
    --------
    >>> constructor = SimilarityThresholdConstructor(similarity_threshold=0.8)
    >>> constructor.construct(G, values)
    """

    def __init__(
        self,
        similarity_threshold: float,
        verbose: bool = True
    ):
        super().__init__(verbose=verbose)
        self.similarity_threshold = similarity_threshold

    @property
    def method_name(self) -> str:
        return "similarity"

    def construct(
        self,
        G: nx.Graph,
        values: np.ndarray,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Add edges based on cosine similarity threshold.

        Parameters
        ----------
        G : nx.Graph
            NetworkX graph with nodes already added
        values : np.ndarray
            Numeric values array of shape (n_nodes, n_features)
        similarity_threshold : float, optional
            Override the default similarity threshold
        **kwargs
            Additional parameters (ignored)
        """
        self.validate_input(values)

        threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold
        self.params = {'similarity_threshold': threshold, 'method': 'similarity'}

        if threshold is None or threshold < 0 or threshold > 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

        if self.verbose:
            print(f"{datetime.now()}: Building similarity graph with threshold={threshold}...")

        # Compute cosine similarity matrix
        sim_matrix = cosine_similarity(values)

        # Find all pairs above threshold (excluding self-loops)
        n_edges_added = 0
        indices = np.where(sim_matrix >= threshold)

        for i, j in zip(*indices):
            if i != j and not G.has_edge(i, j):
                G.add_edge(i, j)
                n_edges_added += 1

        if self.verbose:
            print(f"{datetime.now()}: Added {n_edges_added} edges using similarity threshold={threshold}")
