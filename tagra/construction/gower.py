"""
Gower distance graph constructor.

This module provides graph construction using Gower distance, which handles
mixed-type feature matrices (continuous, binary, ordinal) natively without
requiring preprocessing decisions.

Gower distance between two points is the mean of per-feature partial
distances, each normalised to [0, 1]:

- Binary   features  : 1 if values differ, 0 if equal
- Nominal  features  : 1 if values differ, 0 if equal (any unordered category)
- Ordinal  features  : |x_i - x_j| / range_k  (same as 'range' continuous)
- Continuous features: pluggable — see ``continuous_metric`` parameter

References
----------
Gower, J.C. (1971). A General Coefficient of Similarity and Some of Its
Properties. *Biometrics*, 27(4), 857–871.
"""

from datetime import datetime
from typing import List, Optional
import numpy as np
import networkx as nx

from .base import GraphConstructor


CONTINUOUS_METRICS = ('range', 'quadratic', 'gaussian', 'laplacian')


class GowerDistanceConstructor(GraphConstructor):
    """
    Graph constructor using Gower distance for mixed-type data.

    Handles heterogeneous EHR feature types natively:

    - **Binary** features (0/1 flags): partial distance = 1 if values differ,
      0 if equal.
    - **Nominal** features (unordered categories, e.g. chest pain type,
      thalassemia code): partial distance = 1 if values differ, 0 if equal.
      Semantically identical to binary but accepts any hashable value, not
      just 0/1.  Use this instead of one-hot encoding to keep the feature
      space compact.
    - **Ordinal** features (e.g. ASA score, AJCC stage): range-normalised
      absolute difference, identical to the ``'range'`` continuous metric.
    - **Continuous** features (e.g. Age, AnesTime): pluggable partial distance
      controlled by ``continuous_metric``.

    The overall Gower distance is the mean of all per-feature partial distances
    and lies in [0, 1].  Nodes are connected when their Gower distance is
    at or below ``distance_threshold``.

    Parameters
    ----------
    distance_threshold : float
        Maximum Gower distance for connecting two nodes.  Must be in (0, 1].
        Typical useful range for EHR data: 0.1 – 0.5.
    feature_types : list of str, optional
        Per-column type labels.  Each entry must be one of
        ``'continuous'``, ``'binary'``, ``'ordinal'``, or ``'nominal'``.
        Length must equal the number of columns in the feature matrix.
        If ``None``, all features are treated as ``'continuous'``.
    continuous_metric : str, default ``'range'``
        Partial distance function applied to continuous (and ordinal) features:

        ``'range'``
            ``d = |x_i - x_j| / range_k``  — standard Gower partial distance.
            Linear, bounded, scale-invariant.

        ``'quadratic'``
            ``d = (|x_i - x_j| / range_k) ** 2``  — quadratic penalty,
            emphasises large differences; suppresses small ones.

        ``'gaussian'``
            ``d = 1 - exp(-(x_i - x_j)^2 / (2 * σ_k^2))``  where
            ``σ_k`` is the feature standard deviation.  Rapidly saturates
            at 1 for large differences; emphasises local structure.

        ``'laplacian'``
            ``d = 1 - exp(-|x_i - x_j| / σ_k)``  where ``σ_k`` is the
            feature standard deviation.  Slower saturation than Gaussian;
            robust to outliers.

        For ``'gaussian'`` and ``'laplacian'``, features with zero standard
        deviation fall back to binary treatment (0 if equal, 1 if unequal).
        For ``'range'`` and ``'quadratic'``, zero-range features fall back
        the same way.

    verbose : bool, default ``True``
        Whether to print progress messages.

    Raises
    ------
    ValueError
        If ``distance_threshold`` is not in (0, 1], ``continuous_metric``
        is unknown, or ``feature_types`` contains unknown type labels.

    Examples
    --------
    All features continuous (standard Gower):

    >>> constructor = GowerDistanceConstructor(distance_threshold=0.3)
    >>> constructor.construct(G, values)

    Mixed types, Gaussian metric for continuous features:

    >>> ftypes = ['continuous', 'binary', 'binary', 'ordinal', 'continuous']
    >>> constructor = GowerDistanceConstructor(
    ...     distance_threshold=0.25,
    ...     feature_types=ftypes,
    ...     continuous_metric='gaussian',
    ... )
    >>> constructor.construct(G, values)
    """

    def __init__(
        self,
        distance_threshold: float,
        feature_types: Optional[List[str]] = None,
        continuous_metric: str = 'range',
        verbose: bool = True,
    ):
        super().__init__(verbose=verbose)
        if distance_threshold <= 0 or distance_threshold > 1:
            raise ValueError(
                "distance_threshold must be in (0, 1] for Gower distance."
            )
        if continuous_metric not in CONTINUOUS_METRICS:
            raise ValueError(
                f"Unsupported continuous_metric '{continuous_metric}'. "
                f"Choose from: {list(CONTINUOUS_METRICS)}."
            )
        if feature_types is not None:
            valid = {'continuous', 'binary', 'ordinal', 'nominal'}
            unknown = set(feature_types) - valid
            if unknown:
                raise ValueError(
                    f"Unknown feature type(s): {unknown}. "
                    f"Each entry must be one of: {valid}."
                )
        self.distance_threshold = distance_threshold
        self.feature_types = feature_types
        self.continuous_metric = continuous_metric

    @property
    def method_name(self) -> str:
        return "gower"

    def construct(
        self,
        G: nx.Graph,
        values: np.ndarray,
        **kwargs,
    ) -> None:
        """
        Add undirected edges based on Gower distance threshold.

        Parameters
        ----------
        G : nx.Graph
            NetworkX graph with nodes already added.
        values : np.ndarray
            Feature matrix of shape ``(n_nodes, n_features)``.
        **kwargs
            Additional parameters (ignored).

        Raises
        ------
        ValueError
            If ``feature_types`` length does not match the number of columns.
        """
        self.validate_input(values)

        n, p = values.shape

        feature_types = self.feature_types if self.feature_types is not None \
            else ['continuous'] * p

        if len(feature_types) != p:
            raise ValueError(
                f"feature_types has {len(feature_types)} entries "
                f"but data has {p} features."
            )

        self.params = {
            'distance_threshold': self.distance_threshold,
            'continuous_metric': self.continuous_metric,
            'feature_types': list(feature_types),
            'method': 'gower',
        }

        if self.verbose:
            counts = {t: feature_types.count(t)
                      for t in ('continuous', 'ordinal', 'binary', 'nominal')
                      if t in feature_types}
            print(
                f"{datetime.now()}: Building Gower graph "
                f"(threshold={self.distance_threshold}, "
                f"continuous_metric={self.continuous_metric}, "
                f"feature_types={counts})..."
            )

        D = self._gower_matrix(values.astype(float), feature_types)

        # Lower triangle mask (i > j) to avoid duplicate edges
        mask = np.tril(D <= self.distance_threshold, k=-1)
        rows, cols = np.where(mask)

        n_edges_added = 0
        for i, j in zip(rows, cols):
            if not G.has_edge(i, j):
                G.add_edge(i, j)
                n_edges_added += 1

        if self.verbose:
            print(
                f"{datetime.now()}: Added {n_edges_added} edges "
                f"(threshold={self.distance_threshold})."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gower_matrix(
        self,
        values: np.ndarray,
        feature_types: List[str],
    ) -> np.ndarray:
        """
        Compute the full n×n Gower distance matrix.

        Returns
        -------
        D : np.ndarray, shape (n, n), dtype float64
            Symmetric matrix with values in [0, 1].
        """
        n, p = values.shape
        D_sum = np.zeros((n, n), dtype=np.float64)

        for k, ftype in enumerate(feature_types):
            col = values[:, k]
            if ftype in ('binary', 'nominal'):
                # indicator distance: 0 if equal, 1 if unequal
                # works for any value (0/1 flags or arbitrary category codes)
                d_k = (col[:, None] != col[None, :]).astype(np.float64)
            else:
                # 'continuous' and 'ordinal' both use the continuous metric
                d_k = self._continuous_partial(col)
            D_sum += d_k

        return D_sum / p

    def _continuous_partial(self, col: np.ndarray) -> np.ndarray:
        """
        Partial distance matrix for a single continuous or ordinal feature.

        All returned values are in [0, 1].
        """
        abs_diff = np.abs(col[:, None] - col[None, :])

        if self.continuous_metric in ('range', 'quadratic'):
            r = col.max() - col.min()
            if r == 0:
                return (abs_diff > 0).astype(np.float64)
            d = abs_diff / r
            return d ** 2 if self.continuous_metric == 'quadratic' else d

        sigma = col.std()
        if sigma == 0:
            return (abs_diff > 0).astype(np.float64)

        if self.continuous_metric == 'gaussian':
            diff_sq = (col[:, None] - col[None, :]) ** 2
            return 1.0 - np.exp(-diff_sq / (2.0 * sigma ** 2))

        # laplacian
        return 1.0 - np.exp(-abs_diff / sigma)
