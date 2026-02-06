"""
Scaling transformers for numeric columns.

This module provides wrappers around scikit-learn scalers for
standardization and normalization of numeric data.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler as SKStandardScaler
from sklearn.preprocessing import MinMaxScaler as SKMinMaxScaler

from .base import Transformer


class StandardScaler(Transformer):
    """
    Standardize features by removing mean and scaling to unit variance.

    Wraps sklearn.preprocessing.StandardScaler with a pandas-friendly interface.

    Parameters
    ----------
    verbose : bool, default=True
        Whether to print progress messages

    Attributes
    ----------
    scaler_ : SKStandardScaler
        The underlying sklearn scaler
    columns_ : List[str]
        Columns the scaler was fitted on
    """

    def __init__(self, verbose: bool = True):
        super().__init__(verbose=verbose)
        self.scaler_: Optional[SKStandardScaler] = None
        self.columns_: List[str] = []

    @property
    def name(self) -> str:
        return "standard_scaler"

    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'StandardScaler':
        """
        Fit the standard scaler to the data.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Numeric columns to fit on

        Returns
        -------
        StandardScaler
            Self for method chaining
        """
        if not columns:
            if self.verbose:
                print(f"{datetime.now()}: No columns provided for standard scaling.")
            self.is_fitted = True
            return self

        self.scaler_ = SKStandardScaler()
        self.scaler_.fit(df[columns])
        self.columns_ = columns
        self.is_fitted = True

        if self.verbose:
            print(f"{datetime.now()}: Fitted standard scaler on {len(columns)} columns.")

        return self

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Transform the data using the fitted scaler.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Columns to transform

        Returns
        -------
        pd.DataFrame
            Dataframe with scaled columns
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")

        if not columns or self.scaler_ is None:
            return df

        df_result = df.copy()
        df_result[columns] = self.scaler_.transform(df[columns])

        if self.verbose:
            print(f"{datetime.now()}: Applied standard scaling to {len(columns)} columns.")

        return df_result

    def get_params(self) -> Dict[str, Any]:
        """Get scaler parameters."""
        if self.scaler_ is None:
            return {'method': 'standard'}
        return {
            'method': 'standard',
            'mean': self.scaler_.mean_.tolist() if self.scaler_.mean_ is not None else None,
            'std': self.scaler_.scale_.tolist() if self.scaler_.scale_ is not None else None,
            'columns': self.columns_
        }


class MinMaxScaler(Transformer):
    """
    Scale features to a given range (default 0 to 1).

    Wraps sklearn.preprocessing.MinMaxScaler with a pandas-friendly interface.

    Parameters
    ----------
    feature_range : tuple, default=(0, 1)
        Desired range of transformed data
    verbose : bool, default=True
        Whether to print progress messages

    Attributes
    ----------
    scaler_ : SKMinMaxScaler
        The underlying sklearn scaler
    columns_ : List[str]
        Columns the scaler was fitted on
    """

    def __init__(self, feature_range: tuple = (0, 1), verbose: bool = True):
        super().__init__(verbose=verbose)
        self.feature_range = feature_range
        self.scaler_: Optional[SKMinMaxScaler] = None
        self.columns_: List[str] = []

    @property
    def name(self) -> str:
        return "minmax_scaler"

    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'MinMaxScaler':
        """
        Fit the minmax scaler to the data.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Numeric columns to fit on

        Returns
        -------
        MinMaxScaler
            Self for method chaining
        """
        if not columns:
            if self.verbose:
                print(f"{datetime.now()}: No columns provided for minmax scaling.")
            self.is_fitted = True
            return self

        self.scaler_ = SKMinMaxScaler(feature_range=self.feature_range)
        self.scaler_.fit(df[columns])
        self.columns_ = columns
        self.is_fitted = True

        if self.verbose:
            print(f"{datetime.now()}: Fitted minmax scaler on {len(columns)} columns.")

        return self

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Transform the data using the fitted scaler.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Columns to transform

        Returns
        -------
        pd.DataFrame
            Dataframe with scaled columns
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")

        if not columns or self.scaler_ is None:
            return df

        df_result = df.copy()
        df_result[columns] = self.scaler_.transform(df[columns])

        if self.verbose:
            print(f"{datetime.now()}: Applied minmax scaling to {len(columns)} columns.")

        return df_result

    def get_params(self) -> Dict[str, Any]:
        """Get scaler parameters."""
        if self.scaler_ is None:
            return {'method': 'minmax', 'feature_range': self.feature_range}
        return {
            'method': 'minmax',
            'feature_range': self.feature_range,
            'min': self.scaler_.data_min_.tolist() if self.scaler_.data_min_ is not None else None,
            'max': self.scaler_.data_max_.tolist() if self.scaler_.data_max_ is not None else None,
            'columns': self.columns_
        }


def get_scaler(method: str, verbose: bool = True) -> Transformer:
    """
    Get a scaler by method name.

    Parameters
    ----------
    method : str
        Scaling method: 'standard' or 'minmax'
    verbose : bool, default=True
        Verbosity flag

    Returns
    -------
    Transformer
        The appropriate scaler

    Raises
    ------
    ValueError
        If method is not recognized
    """
    if method == 'standard':
        return StandardScaler(verbose=verbose)
    elif method == 'minmax':
        return MinMaxScaler(verbose=verbose)
    else:
        raise ValueError(f"Unknown scaling method: {method}. Use 'standard' or 'minmax'.")
