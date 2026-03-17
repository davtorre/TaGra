"""
DBSCAN-equivalent directed graph constructor.

This module provides graph construction that is exactly equivalent to DBSCAN:
a directed distance-threshold graph where node i gets outgoing edges only if
it qualifies as a core point (has >= min_samples - 1 neighbors within eps,
excluding itself).

Clustering semantics of the resulting nx.DiGraph:
- Isolated nodes (no in- or out-edges)  -> noise points
- Weakly connected components           -> clusters
- Nodes with outgoing edges             -> core points
- Nodes with only incoming edges        -> border points
"""

from datetime import datetime
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


class DBSCANGraphConstructor(GraphConstructor):
    """
    Directed graph constructor exactly equivalent to DBSCAN.

    For each node i, all neighbors within distance ``eps`` are found.
    If the number of such neighbors (excluding i itself) is >=
    ``min_samples - 1``, directed edges i -> j are added for every
    neighbor j.  Otherwise, no outgoing edges are added from i.

    The resulting nx.DiGraph encodes the full DBSCAN semantics:

    * **Isolated nodes** (degree 0) correspond to DBSCAN noise points.
    * **Weakly connected components** correspond to DBSCAN clusters.
    * **Nodes with out-degree > 0** correspond to DBSCAN core points.
    * **Nodes with in-degree > 0 and out-degree == 0** correspond to
      DBSCAN border points.

    Parameters
    ----------
    eps : float
        Neighborhood radius.  Two points are neighbors if their distance
        is <= ``eps``.
    min_samples : int
        Minimum number of neighbors within ``eps`` (excluding self) for
        a point to be considered a core point.  Equivalent to DBSCAN's
        ``min_samples - 1`` neighbor threshold.
    metric : str, default='euclidean'
        Distance metric.  One of ``'euclidean'`` (L2), ``'manhattan'``
        (L1), or ``'chebyshev'`` (L-infinity).
    verbose : bool, default=True
        Whether to print progress messages.

    Raises
    ------
    ValueError
        If ``eps <= 0``, ``min_samples < 1``, or ``metric`` is not
        supported.

    Notes
    -----
    The graph passed to :meth:`construct` must be a ``nx.DiGraph``.
    Passing an undirected ``nx.Graph`` raises ``TypeError``.

    Examples
    --------
    >>> import networkx as nx
    >>> constructor = DBSCANGraphConstructor(eps=0.5, min_samples=5)
    >>> G = nx.DiGraph()
    >>> for i in range(n): G.add_node(i)
    >>> constructor.construct(G, values)
    """

    def __init__(
        self,
        eps: float,
        min_samples: int,
        metric: str = 'euclidean',
        verbose: bool = True,
    ):
        super().__init__(verbose=verbose)
        if eps <= 0:
            raise ValueError("eps must be a positive number.")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1.")
        if metric not in METRIC_TO_P:
            raise ValueError(
                f"Unsupported metric '{metric}'. "
                f"Choose from: {list(METRIC_TO_P)}."
            )
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    @property
    def method_name(self) -> str:
        return "dbscan"

    def construct(
        self,
        G: nx.DiGraph,
        values: np.ndarray,
        **kwargs,
    ) -> None:
        """
        Add directed edges to the graph based on the DBSCAN core-point
        criterion.

        Parameters
        ----------
        G : nx.DiGraph
            Directed NetworkX graph with nodes already added.
        values : np.ndarray
            Numeric feature matrix of shape ``(n_nodes, n_features)``.
        **kwargs
            Additional parameters (ignored).

        Raises
        ------
        TypeError
            If ``G`` is not a ``nx.DiGraph``.
        """
        if not isinstance(G, nx.DiGraph):
            raise TypeError(
                "DBSCANGraphConstructor requires a directed graph "
                f"(nx.DiGraph). Got: {type(G).__name__}."
            )

        self.validate_input(values)

        p = METRIC_TO_P[self.metric]
        core_threshold = self.min_samples - 1

        self.params = {
            'eps': self.eps,
            'min_samples': self.min_samples,
            'metric': self.metric,
            'method': 'dbscan',
        }

        if self.verbose:
            print(
                f"{datetime.now()}: Building DBSCAN directed graph "
                f"(eps={self.eps}, min_samples={self.min_samples}, "
                f"metric={self.metric})..."
            )

        tree = cKDTree(values)

        # query_ball_point returns, for each point i, the list of all
        # point indices (including i itself) within distance eps.
        neighborhoods = tree.query_ball_point(values, self.eps, p=p)

        n_edges_added = 0
        n_core = 0

        for i, raw_neighbors in enumerate(neighborhoods):
            neighbors = [j for j in raw_neighbors if j != i]
            if len(neighbors) >= core_threshold:
                n_core += 1
                for j in neighbors:
                    if not G.has_edge(i, j):
                        G.add_edge(i, j)
                        n_edges_added += 1

        if self.verbose:
            print(
                f"{datetime.now()}: Added {n_edges_added} directed edges. "
                f"Core nodes: {n_core}/{values.shape[0]}."
            )
