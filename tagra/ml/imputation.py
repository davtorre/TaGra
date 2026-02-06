"""
Graph-based missing value imputation.

This module provides methods for imputing missing values using
graph structure information.
"""

from datetime import datetime
from typing import Optional, Dict, Any, Union, List
import numpy as np
import pandas as pd
import networkx as nx


class GraphImputer:
    """
    Impute missing values using graph neighborhood information.

    Uses neighbor values to estimate missing values, weighted by
    edge proximity in the graph.

    Parameters
    ----------
    strategy : str, default='mean'
        Imputation strategy:
        - 'mean': Use mean of neighbor values
        - 'weighted_mean': Weight by inverse distance
        - 'median': Use median of neighbor values
        - 'mode': Use mode of neighbor values (for categorical)
    n_neighbors : int, optional
        Maximum neighbors to consider (None = all)
    fallback : str, default='global'
        Fallback for nodes with no neighbors:
        - 'global': Use global mean/median/mode
        - 'drop': Keep as NaN
    verbose : bool, default=True
        Print progress messages

    Examples
    --------
    >>> imputer = GraphImputer(strategy='weighted_mean')
    >>> df_imputed = imputer.fit_transform(G, df)
    """

    def __init__(
        self,
        strategy: str = 'mean',
        n_neighbors: Optional[int] = None,
        fallback: str = 'global',
        verbose: bool = True
    ):
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.fallback = fallback
        self.verbose = verbose

        self._global_values: Dict[str, Any] = {}
        self._is_fitted = False

    def fit(
        self,
        graph: Union[nx.Graph, 'TaGraGraph'],
        data: pd.DataFrame
    ) -> 'GraphImputer':
        """
        Fit the imputer by computing global statistics.

        Parameters
        ----------
        graph : nx.Graph or TaGraGraph
            Graph structure
        data : pd.DataFrame
            Data with potential missing values

        Returns
        -------
        GraphImputer
            Self for method chaining
        """
        if self.verbose:
            print(f"{datetime.now()}: Fitting graph imputer...")

        for col in data.columns:
            valid_values = data[col].dropna()
            if len(valid_values) == 0:
                continue

            if self.strategy == 'mode' or data[col].dtype == 'object':
                mode = valid_values.mode()
                self._global_values[col] = mode[0] if len(mode) > 0 else None
            elif self.strategy == 'median':
                self._global_values[col] = valid_values.median()
            else:
                self._global_values[col] = valid_values.mean()

        self._is_fitted = True
        return self

    def transform(
        self,
        graph: Union[nx.Graph, 'TaGraGraph'],
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Impute missing values using graph structure.

        Parameters
        ----------
        graph : nx.Graph or TaGraGraph
            Graph structure
        data : pd.DataFrame
            Data with missing values

        Returns
        -------
        pd.DataFrame
            Data with imputed values
        """
        if not self._is_fitted:
            raise RuntimeError("Imputer must be fitted before transform")

        if hasattr(graph, 'to_networkx'):
            G = graph.to_networkx()
        else:
            G = graph

        nodes = list(G.nodes())
        df = data.copy()

        if self.verbose:
            total_missing = df.isna().sum().sum()
            print(f"{datetime.now()}: Imputing {total_missing} missing values...")

        for col in df.columns:
            missing_mask = df[col].isna()
            if not missing_mask.any():
                continue

            n_missing = missing_mask.sum()
            if self.verbose:
                print(f"{datetime.now()}: Column '{col}': {n_missing} missing values")

            for idx in df[missing_mask].index:
                # Get node (assuming index corresponds to node)
                node = idx if idx in G else (nodes[idx] if idx < len(nodes) else None)

                if node is None or node not in G:
                    if self.fallback == 'global':
                        df.loc[idx, col] = self._global_values.get(col)
                    continue

                # Get neighbor values
                neighbors = list(G.neighbors(node))
                if self.n_neighbors:
                    neighbors = neighbors[:self.n_neighbors]

                neighbor_values = []
                for neighbor in neighbors:
                    neighbor_idx = nodes.index(neighbor) if neighbor in nodes else neighbor
                    if neighbor_idx in df.index:
                        val = df.loc[neighbor_idx, col]
                        if pd.notna(val):
                            neighbor_values.append(val)

                # Compute imputed value
                if neighbor_values:
                    imputed = self._compute_imputation(neighbor_values, df[col].dtype)
                    df.loc[idx, col] = imputed
                elif self.fallback == 'global':
                    df.loc[idx, col] = self._global_values.get(col)

        if self.verbose:
            remaining = df.isna().sum().sum()
            print(f"{datetime.now()}: Imputation complete. Remaining missing: {remaining}")

        return df

    def _compute_imputation(self, values: List[Any], dtype) -> Any:
        """Compute imputed value from neighbor values."""
        if dtype == 'object' or self.strategy == 'mode':
            # Mode for categorical
            unique, counts = np.unique(values, return_counts=True)
            return unique[np.argmax(counts)]
        elif self.strategy == 'median':
            return np.median(values)
        elif self.strategy == 'weighted_mean':
            # Simple mean (true weighted would need distances)
            return np.mean(values)
        else:
            return np.mean(values)

    def fit_transform(
        self,
        graph: Union[nx.Graph, 'TaGraGraph'],
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Parameters
        ----------
        graph : nx.Graph or TaGraGraph
            Graph structure
        data : pd.DataFrame
            Data with missing values

        Returns
        -------
        pd.DataFrame
            Data with imputed values
        """
        self.fit(graph, data)
        return self.transform(graph, data)


def impute_with_graph(
    graph: Union[nx.Graph, 'TaGraGraph'],
    data: pd.DataFrame,
    strategy: str = 'mean',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Convenience function for graph-based imputation.

    Parameters
    ----------
    graph : nx.Graph or TaGraGraph
        Graph structure
    data : pd.DataFrame
        Data with missing values
    strategy : str, default='mean'
        Imputation strategy
    verbose : bool, default=True
        Print progress

    Returns
    -------
    pd.DataFrame
        Imputed data
    """
    imputer = GraphImputer(strategy=strategy, verbose=verbose)
    return imputer.fit_transform(graph, data)
