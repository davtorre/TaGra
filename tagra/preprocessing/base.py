"""
Base class for preprocessing transformers.

This module defines the abstract base class that all preprocessing
transformers must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np


class Transformer(ABC):
    """
    Abstract base class for preprocessing transformers.

    All preprocessing transformers (scalers, encoders, etc.) inherit from
    this class and implement the fit, transform, and fit_transform methods.

    Parameters
    ----------
    verbose : bool, default=True
        Whether to print progress messages

    Attributes
    ----------
    verbose : bool
        Verbosity flag
    is_fitted : bool
        Whether the transformer has been fitted
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.is_fitted = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this transformer."""
        pass

    @abstractmethod
    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'Transformer':
        """
        Fit the transformer to the data.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Columns to fit on

        Returns
        -------
        Transformer
            Self for method chaining
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Transform the data.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Columns to transform

        Returns
        -------
        pd.DataFrame
            Transformed dataframe
        """
        pass

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Columns to fit and transform

        Returns
        -------
        pd.DataFrame
            Transformed dataframe
        """
        self.fit(df, columns)
        return self.transform(df, columns)

    def get_params(self) -> Dict[str, Any]:
        """
        Get transformer parameters.

        Returns
        -------
        Dict[str, Any]
            Parameters dictionary
        """
        return {}
