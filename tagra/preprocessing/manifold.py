"""
Manifold learning transformers for dimensionality reduction.

This module provides wrappers around various manifold learning methods
(UMAP, t-SNE, Isomap) for reducing high-dimensional data to lower dimensions.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np

from .base import Transformer


class ManifoldReducer(Transformer):
    """
    Reduce dimensionality using manifold learning methods.

    Supports UMAP, t-SNE, and Isomap for embedding high-dimensional
    data into lower dimensions (typically 2D for visualization).

    Parameters
    ----------
    method : str, default='UMAP'
        Manifold method: 'UMAP', 'TSNE', or 'Isomap'
    n_components : int, default=2
        Target dimensionality
    n_neighbors : int, default=15
        Number of neighbors for UMAP and Isomap
    min_dist : float, default=0.1
        Minimum distance for UMAP
    random_state : int, optional
        Random seed for reproducibility
    verbose : bool, default=True
        Whether to print progress messages

    Attributes
    ----------
    manifold_ : object
        The fitted manifold learning model
    embedding_ : np.ndarray
        The transformed embedding
    """

    METHODS = ['UMAP', 'TSNE', 'Isomap']

    def __init__(
        self,
        method: str = 'UMAP',
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: Optional[int] = 42,
        verbose: bool = True
    ):
        super().__init__(verbose=verbose)
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Use one of {self.METHODS}")
        self.method = method
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.manifold_ = None
        self.embedding_: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return f"manifold_{self.method.lower()}"

    def _create_manifold(self):
        """Create the manifold learning model."""
        if self.method == 'UMAP':
            from umap.umap_ import UMAP
            return UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                random_state=self.random_state
            )
        elif self.method == 'TSNE':
            from sklearn.manifold import TSNE
            return TSNE(
                n_components=self.n_components,
                random_state=self.random_state
            )
        elif self.method == 'Isomap':
            from sklearn.manifold import Isomap
            return Isomap(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors
            )

    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'ManifoldReducer':
        """
        Fit the manifold model.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Numeric columns to use for manifold learning

        Returns
        -------
        ManifoldReducer
            Self for method chaining
        """
        if not columns:
            if self.verbose:
                print(f"{datetime.now()}: No columns provided for manifold learning.")
            self.is_fitted = True
            return self

        if len(columns) < self.n_components:
            if self.verbose:
                print(f"{datetime.now()}: Not enough columns ({len(columns)}) for {self.n_components}D manifold. Skipping.")
            self.is_fitted = True
            return self

        if self.verbose:
            print(f"{datetime.now()}: Fitting {self.method} manifold...")

        self.manifold_ = self._create_manifold()
        values = df[columns].values
        self.manifold_.fit(values)
        self.is_fitted = True

        if self.verbose:
            print(f"{datetime.now()}: Fitted {self.method} on {len(columns)} features.")

        return self

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Transform data using the fitted manifold model.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Columns that were used for fitting

        Returns
        -------
        pd.DataFrame
            Dataframe with manifold columns replacing original numeric columns
        """
        if not self.is_fitted:
            raise RuntimeError("Manifold must be fitted before transform")

        if not columns or self.manifold_ is None:
            return df

        if self.verbose:
            print(f"{datetime.now()}: Transforming data with {self.method}...")

        values = df[columns].values

        # TSNE doesn't have a separate transform method
        if self.method == 'TSNE':
            self.embedding_ = self.manifold_.fit_transform(values)
        else:
            self.embedding_ = self.manifold_.transform(values)

        # Create new dataframe with manifold columns
        df_result = df.copy()
        df_result = df_result.drop(columns=columns)

        manifold_columns = [f'manifold_{i}' for i in range(self.n_components)]
        for i, col in enumerate(manifold_columns):
            df_result[col] = self.embedding_[:, i]

        if self.verbose:
            print(f"{datetime.now()}: Created {len(manifold_columns)} manifold dimensions.")

        return df_result

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Numeric columns for manifold learning

        Returns
        -------
        pd.DataFrame
            Dataframe with manifold columns
        """
        if not columns:
            return df

        if len(columns) < self.n_components:
            if self.verbose:
                print(f"{datetime.now()}: Not enough columns for manifold. Skipping.")
            return df

        if self.verbose:
            print(f"{datetime.now()}: Applying {self.method} manifold learning...")

        self.manifold_ = self._create_manifold()
        values = df[columns].values
        self.embedding_ = self.manifold_.fit_transform(values)
        self.is_fitted = True

        # Create new dataframe with manifold columns
        df_result = df.copy()
        df_result = df_result.drop(columns=columns)

        manifold_columns = [f'manifold_{i}' for i in range(self.n_components)]
        for i, col in enumerate(manifold_columns):
            df_result[col] = self.embedding_[:, i]

        if self.verbose:
            print(f"{datetime.now()}: Applied {self.method} with n_components={self.n_components}, "
                  f"n_neighbors={self.n_neighbors if hasattr(self.manifold_, 'n_neighbors') else 'N/A'}")

        return df_result

    def get_embedding(self) -> Optional[np.ndarray]:
        """
        Get the manifold embedding.

        Returns
        -------
        Optional[np.ndarray]
            The embedding array, or None if not computed
        """
        return self.embedding_

    def get_params(self) -> Dict[str, Any]:
        """Get manifold parameters."""
        return {
            'method': self.method,
            'n_components': self.n_components,
            'n_neighbors': self.n_neighbors,
            'min_dist': self.min_dist,
            'random_state': self.random_state
        }


def get_manifold_reducer(
    method: str,
    n_components: int = 2,
    verbose: bool = True,
    **kwargs
) -> ManifoldReducer:
    """
    Get a manifold reducer by method name.

    Parameters
    ----------
    method : str
        Manifold method: 'UMAP', 'TSNE', or 'Isomap'
    n_components : int, default=2
        Target dimensionality
    verbose : bool, default=True
        Verbosity flag
    **kwargs
        Additional arguments for the reducer

    Returns
    -------
    ManifoldReducer
        The manifold reducer
    """
    return ManifoldReducer(
        method=method,
        n_components=n_components,
        verbose=verbose,
        **kwargs
    )
