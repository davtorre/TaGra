"""
Label propagation for semi-supervised learning on graphs.

This module provides graph-based label propagation for propagating
labels from labeled nodes to unlabeled nodes.
"""

from datetime import datetime
from typing import Optional, Dict, Any, Union, List
import numpy as np
import networkx as nx


class LabelPropagator:
    """
    Semi-supervised label propagation on graphs.

    Propagates known labels through the graph structure to predict
    labels for unlabeled nodes.

    Parameters
    ----------
    max_iter : int, default=100
        Maximum number of iterations
    tol : float, default=1e-6
        Convergence tolerance
    alpha : float, default=0.5
        Clamping factor (0 = fully propagate, 1 = keep original)
    verbose : bool, default=True
        Print progress messages

    Attributes
    ----------
    labels_ : np.ndarray
        Predicted labels after fitting
    n_iter_ : int
        Number of iterations until convergence
    classes_ : np.ndarray
        Unique class labels

    Examples
    --------
    >>> propagator = LabelPropagator()
    >>> propagator.fit(G, labels, mask)
    >>> predictions = propagator.predict()
    """

    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-6,
        alpha: float = 0.5,
        verbose: bool = True
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.verbose = verbose

        self.labels_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self.classes_: Optional[np.ndarray] = None
        self._label_distributions: Optional[np.ndarray] = None

    def fit(
        self,
        graph: Union[nx.Graph, 'TaGraGraph'],
        labels: Union[np.ndarray, List, Dict[int, Any]],
        labeled_mask: Optional[np.ndarray] = None
    ) -> 'LabelPropagator':
        """
        Fit the label propagation model.

        Parameters
        ----------
        graph : nx.Graph or TaGraGraph
            Graph structure
        labels : array-like or dict
            Known labels. Use -1 or None for unlabeled nodes.
            If dict, maps node ID to label.
        labeled_mask : np.ndarray, optional
            Boolean mask indicating labeled nodes

        Returns
        -------
        LabelPropagator
            Self for method chaining
        """
        if hasattr(graph, 'to_networkx'):
            G = graph.to_networkx()
        else:
            G = graph

        n_nodes = G.number_of_nodes()
        nodes = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        if self.verbose:
            print(f"{datetime.now()}: Fitting label propagation on {n_nodes} nodes...")

        # Convert labels to array
        if isinstance(labels, dict):
            label_array = np.full(n_nodes, -1, dtype=object)
            for node, label in labels.items():
                if node in node_to_idx:
                    label_array[node_to_idx[node]] = label
        else:
            label_array = np.array(labels)

        # Determine labeled mask
        if labeled_mask is None:
            labeled_mask = np.array([l != -1 and l is not None for l in label_array])

        # Get unique classes (excluding unlabeled)
        labeled_labels = label_array[labeled_mask]
        self.classes_ = np.unique(labeled_labels)
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        if self.verbose:
            n_labeled = np.sum(labeled_mask)
            print(f"{datetime.now()}: {n_labeled} labeled nodes, {n_classes} classes")

        # Initialize label distributions
        Y = np.zeros((n_nodes, n_classes))
        for i, label in enumerate(label_array):
            if labeled_mask[i]:
                Y[i, class_to_idx[label]] = 1.0

        # Build adjacency matrix (normalized)
        A = nx.to_numpy_array(G, nodelist=nodes)
        # Row-normalize
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        W = A / row_sums

        # Clamped label matrix
        Y_clamped = Y.copy()

        # Iterate
        for iteration in range(self.max_iter):
            Y_new = self.alpha * Y_clamped + (1 - self.alpha) * (W @ Y)

            # Re-clamp labeled nodes
            Y_new[labeled_mask] = Y_clamped[labeled_mask]

            # Check convergence
            diff = np.abs(Y_new - Y).max()
            Y = Y_new

            if diff < self.tol:
                if self.verbose:
                    print(f"{datetime.now()}: Converged after {iteration + 1} iterations")
                break

        self.n_iter_ = iteration + 1
        self._label_distributions = Y

        # Get final labels
        self.labels_ = np.array([self.classes_[np.argmax(Y[i])]
                                 for i in range(n_nodes)])

        if self.verbose:
            print(f"{datetime.now()}: Label propagation complete")

        return self

    def predict(self) -> np.ndarray:
        """
        Get predicted labels.

        Returns
        -------
        np.ndarray
            Predicted labels for all nodes
        """
        if self.labels_ is None:
            raise RuntimeError("Model must be fitted before predict")
        return self.labels_.copy()

    def predict_proba(self) -> np.ndarray:
        """
        Get label probability distributions.

        Returns
        -------
        np.ndarray
            Probability distribution over classes for each node
            Shape: (n_nodes, n_classes)
        """
        if self._label_distributions is None:
            raise RuntimeError("Model must be fitted before predict_proba")
        return self._label_distributions.copy()

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'max_iter': self.max_iter,
            'tol': self.tol,
            'alpha': self.alpha,
            'n_iter': self.n_iter_,
            'classes': self.classes_.tolist() if self.classes_ is not None else None
        }


def propagate_labels(
    graph: Union[nx.Graph, 'TaGraGraph'],
    labels: Union[np.ndarray, Dict[int, Any]],
    labeled_mask: Optional[np.ndarray] = None,
    max_iter: int = 100,
    alpha: float = 0.5,
    verbose: bool = True
) -> np.ndarray:
    """
    Convenience function for label propagation.

    Parameters
    ----------
    graph : nx.Graph or TaGraGraph
        Graph structure
    labels : array-like or dict
        Known labels
    labeled_mask : np.ndarray, optional
        Boolean mask for labeled nodes
    max_iter : int, default=100
        Maximum iterations
    alpha : float, default=0.5
        Clamping factor
    verbose : bool, default=True
        Print progress

    Returns
    -------
    np.ndarray
        Predicted labels for all nodes
    """
    propagator = LabelPropagator(
        max_iter=max_iter,
        alpha=alpha,
        verbose=verbose
    )
    propagator.fit(graph, labels, labeled_mask)
    return propagator.predict()
