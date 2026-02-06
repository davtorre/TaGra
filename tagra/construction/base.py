"""
Base class for graph constructors.

This module defines the abstract base class that all graph construction
methods must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
import networkx as nx


class GraphConstructor(ABC):
    """
    Abstract base class for graph construction methods.

    All graph construction methods (KNN, distance threshold, similarity threshold)
    inherit from this class and implement the `construct` method.

    Parameters
    ----------
    verbose : bool, default=True
        Whether to print progress messages

    Attributes
    ----------
    verbose : bool
        Verbosity flag
    params : Dict[str, Any]
        Parameters used for construction (set during construct())
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.params: Dict[str, Any] = {}

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the name of this construction method."""
        pass

    @abstractmethod
    def construct(
        self,
        G: nx.Graph,
        values: np.ndarray,
        **kwargs
    ) -> None:
        """
        Add edges to the graph based on the construction method.

        Parameters
        ----------
        G : nx.Graph
            NetworkX graph with nodes already added
        values : np.ndarray
            Numeric values array of shape (n_nodes, n_features)
        **kwargs
            Method-specific parameters

        Notes
        -----
        This method modifies the graph in place by adding edges.
        """
        pass

    def validate_input(self, values: np.ndarray) -> None:
        """
        Validate input values array.

        Parameters
        ----------
        values : np.ndarray
            Input values to validate

        Raises
        ------
        ValueError
            If values are invalid
        """
        if values is None or len(values) == 0:
            raise ValueError("Values array cannot be empty")
        if len(values.shape) != 2:
            raise ValueError(f"Values must be 2D array, got shape {values.shape}")
        if values.shape[1] == 0:
            raise ValueError("No features in values array")

    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters used for construction.

        Returns
        -------
        Dict[str, Any]
            Parameters dictionary
        """
        return self.params.copy()
