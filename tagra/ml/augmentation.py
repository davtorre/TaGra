"""
Graph-aware data augmentation.

This module provides data augmentation techniques that leverage
graph structure for generating synthetic samples.
"""

from datetime import datetime
from typing import Optional, Dict, Any, Union, List, Tuple
import numpy as np
import pandas as pd
import networkx as nx


class GraphAwareAugmenter:
    """
    Data augmentation using graph neighborhood information.

    Generates synthetic samples by interpolating between nodes
    and their neighbors in the graph.

    Parameters
    ----------
    n_samples : int, default=1
        Number of synthetic samples per original sample
    alpha_range : tuple, default=(0.3, 0.7)
        Range for interpolation weights
    use_neighbors_only : bool, default=True
        Only interpolate with connected neighbors
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool, default=True
        Print progress messages

    Examples
    --------
    >>> augmenter = GraphAwareAugmenter(n_samples=2)
    >>> X_aug, y_aug = augmenter.augment(G, X, y)
    """

    def __init__(
        self,
        n_samples: int = 1,
        alpha_range: Tuple[float, float] = (0.3, 0.7),
        use_neighbors_only: bool = True,
        random_state: Optional[int] = None,
        verbose: bool = True
    ):
        self.n_samples = n_samples
        self.alpha_range = alpha_range
        self.use_neighbors_only = use_neighbors_only
        self.random_state = random_state
        self.verbose = verbose

        if random_state is not None:
            np.random.seed(random_state)

    def augment(
        self,
        graph: Union[nx.Graph, 'TaGraGraph'],
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        target_class: Optional[Any] = None
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Optional[Union[np.ndarray, pd.Series]]]:
        """
        Generate augmented samples using graph structure.

        Parameters
        ----------
        graph : nx.Graph or TaGraGraph
            Graph structure
        X : np.ndarray or pd.DataFrame
            Feature matrix (n_samples, n_features)
        y : array-like, optional
            Labels
        target_class : any, optional
            Only augment samples of this class

        Returns
        -------
        Tuple
            (X_augmented, y_augmented) including original and synthetic samples
        """
        if hasattr(graph, 'to_networkx'):
            G = graph.to_networkx()
        else:
            G = graph

        is_dataframe = isinstance(X, pd.DataFrame)
        if is_dataframe:
            columns = X.columns
            X_arr = X.values
        else:
            X_arr = X

        n_original = len(X_arr)

        if self.verbose:
            print(f"{datetime.now()}: Augmenting {n_original} samples...")

        # Determine which samples to augment
        if target_class is not None and y is not None:
            y_arr = np.array(y) if not isinstance(y, np.ndarray) else y
            augment_mask = y_arr == target_class
        else:
            augment_mask = np.ones(n_original, dtype=bool)
            y_arr = np.array(y) if y is not None else None

        synthetic_X = []
        synthetic_y = []

        nodes = list(G.nodes())
        for node_idx in range(n_original):
            if not augment_mask[node_idx]:
                continue

            node = nodes[node_idx] if node_idx < len(nodes) else node_idx

            # Get neighbors
            if self.use_neighbors_only and node in G:
                neighbors = list(G.neighbors(node))
                if not neighbors:
                    continue
            else:
                # Use all nodes as potential neighbors
                neighbors = [n for n in nodes if n != node]

            # Generate synthetic samples
            for _ in range(self.n_samples):
                # Pick random neighbor
                neighbor = neighbors[np.random.randint(len(neighbors))]
                neighbor_idx = nodes.index(neighbor) if neighbor in nodes else neighbor

                if neighbor_idx >= len(X_arr):
                    continue

                # Interpolate
                alpha = np.random.uniform(*self.alpha_range)
                synthetic_sample = alpha * X_arr[node_idx] + (1 - alpha) * X_arr[neighbor_idx]
                synthetic_X.append(synthetic_sample)

                # Keep same label
                if y_arr is not None:
                    synthetic_y.append(y_arr[node_idx])

        if not synthetic_X:
            if self.verbose:
                print(f"{datetime.now()}: No synthetic samples generated")
            return (X, y)

        synthetic_X = np.array(synthetic_X)
        X_augmented = np.vstack([X_arr, synthetic_X])

        if is_dataframe:
            X_augmented = pd.DataFrame(X_augmented, columns=columns)

        if y_arr is not None:
            y_augmented = np.concatenate([y_arr, synthetic_y])
            if isinstance(y, pd.Series):
                y_augmented = pd.Series(y_augmented)
        else:
            y_augmented = None

        if self.verbose:
            print(f"{datetime.now()}: Generated {len(synthetic_X)} synthetic samples")
            print(f"{datetime.now()}: Total samples: {len(X_augmented)}")

        return (X_augmented, y_augmented)

    def augment_minority(
        self,
        graph: Union[nx.Graph, 'TaGraGraph'],
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        target_ratio: float = 1.0
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series]]:
        """
        Augment minority class to balance dataset.

        Parameters
        ----------
        graph : nx.Graph or TaGraGraph
            Graph structure
        X : array-like
            Feature matrix
        y : array-like
            Labels
        target_ratio : float, default=1.0
            Target ratio of minority to majority class

        Returns
        -------
        Tuple
            Balanced (X, y) with augmented minority samples
        """
        y_arr = np.array(y) if not isinstance(y, np.ndarray) else y
        unique, counts = np.unique(y_arr, return_counts=True)

        majority_count = max(counts)
        minority_class = unique[np.argmin(counts)]
        minority_count = min(counts)

        target_count = int(majority_count * target_ratio)
        n_to_generate = target_count - minority_count

        if n_to_generate <= 0:
            if self.verbose:
                print(f"{datetime.now()}: No augmentation needed")
            return (X, y)

        if self.verbose:
            print(f"{datetime.now()}: Augmenting minority class '{minority_class}'")
            print(f"{datetime.now()}: Generating {n_to_generate} samples")

        # Compute samples per original
        n_minority = np.sum(y_arr == minority_class)
        samples_per = max(1, n_to_generate // n_minority)

        original_n_samples = self.n_samples
        self.n_samples = samples_per

        X_aug, y_aug = self.augment(graph, X, y, target_class=minority_class)

        self.n_samples = original_n_samples

        return (X_aug, y_aug)
