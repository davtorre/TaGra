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


# Supported Minkowski metrics and their p-norm values for cKDTree
METRIC_TO_P = {
    'euclidean': 2.0,
    'manhattan': 1.0,
    'chebyshev': np.inf,
}


class DistanceThresholdConstructor(GraphConstructor):
    """
    Graph constructor using distance threshold.

    Creates undirected edges between nodes that are within a specified
    distance of each other in the feature space.

    Parameters
    ----------
    distance_threshold : float
        Maximum distance for connecting two nodes.
    metric : str, default='euclidean'
        Distance metric.  One of ``'euclidean'`` (L2), ``'manhattan'``
        (L1), or ``'chebyshev'`` (L-infinity).
    verbose : bool, default=True
        Whether to print progress messages.

    Raises
    ------
    ValueError
        If ``metric`` is not one of the supported values.

    Examples
    --------
    >>> constructor = DistanceThresholdConstructor(distance_threshold=1.0)
    >>> constructor.construct(G, values)

    >>> constructor = DistanceThresholdConstructor(
    ...     distance_threshold=1.0, metric='manhattan'
    ... )
    >>> constructor.construct(G, values)
    """

    def __init__(
        self,
        distance_threshold: float,
        metric: str = 'euclidean',
        verbose: bool = True,
    ):
        super().__init__(verbose=verbose)
        if metric not in METRIC_TO_P:
            raise ValueError(
                f"Unsupported metric '{metric}'. "
                f"Choose from: {list(METRIC_TO_P)}."
            )
        self.distance_threshold = distance_threshold
        self.metric = metric

    @property
    def method_name(self) -> str:
        return "distance"

    def construct(
        self,
        G: nx.Graph,
        values: np.ndarray,
        distance_threshold: Optional[float] = None,
        metric: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Add undirected edges based on distance threshold.

        Parameters
        ----------
        G : nx.Graph
            NetworkX graph with nodes already added.
        values : np.ndarray
            Numeric values array of shape (n_nodes, n_features).
        distance_threshold : float, optional
            Override the instance distance threshold.
        metric : str, optional
            Override the instance metric.  One of ``'euclidean'``,
            ``'manhattan'``, or ``'chebyshev'``.
        **kwargs
            Additional parameters (ignored).
        """
        self.validate_input(values)

        threshold = distance_threshold if distance_threshold is not None else self.distance_threshold
        metric_use = metric if metric is not None else self.metric

        if threshold is None or threshold <= 0:
            raise ValueError("distance_threshold must be a positive number.")
        if metric_use not in METRIC_TO_P:
            raise ValueError(
                f"Unsupported metric '{metric_use}'. "
                f"Choose from: {list(METRIC_TO_P)}."
            )

        self.params = {
            'distance_threshold': threshold,
            'metric': metric_use,
            'method': 'distance',
        }

        if self.verbose:
            print(
                f"{datetime.now()}: Building distance graph "
                f"(threshold={threshold}, metric={metric_use})..."
            )

        p = METRIC_TO_P[metric_use]
        tree = cKDTree(values)
        pairs = tree.query_pairs(threshold, p=p)

        n_edges_added = 0
        for i, j in pairs:
            if not G.has_edge(i, j):
                G.add_edge(i, j)
                n_edges_added += 1

        if self.verbose:
            print(
                f"{datetime.now()}: Added {n_edges_added} edges "
                f"(threshold={threshold}, metric={metric_use})."
            )
