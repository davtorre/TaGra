"""
Missing value handlers for preprocessing.

This module provides strategies for handling missing values (NaN)
in dataframes.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np

from .base import Transformer


class MissingValueHandler(Transformer):
    """
    Handle missing values in a dataframe.

    Supports multiple strategies for dealing with NaN values:
    - 'drop_row': Remove rows with any missing values
    - 'drop_column': Remove columns exceeding a threshold of missing values
    - 'infer': Fill numeric with mean, categorical with mode
    - 'fill_value': Fill with a specified value
    - 'fill_mean': Fill numeric columns with mean
    - 'fill_median': Fill numeric columns with median
    - 'fill_mode': Fill with mode

    Parameters
    ----------
    strategy : str, default='infer'
        Strategy for handling missing values
    threshold : float, default=0.5
        For 'drop_column', the max proportion of missing values allowed
    fill_value : Any, optional
        Value to use for 'fill_value' strategy
    verbose : bool, default=True
        Whether to print progress messages

    Attributes
    ----------
    fill_values_ : Dict[str, Any]
        Values used to fill each column (for fitted strategies)
    """

    STRATEGIES = ['drop_row', 'drop_column', 'infer', 'fill_value',
                  'fill_mean', 'fill_median', 'fill_mode']

    def __init__(
        self,
        strategy: str = 'infer',
        threshold: float = 0.5,
        fill_value: Any = None,
        verbose: bool = True
    ):
        super().__init__(verbose=verbose)
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Use one of {self.STRATEGIES}")
        self.strategy = strategy
        self.threshold = threshold
        self.fill_value = fill_value
        self.fill_values_: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "missing_value_handler"

    def fit(
        self,
        df: pd.DataFrame,
        columns: List[str],
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None
    ) -> 'MissingValueHandler':
        """
        Fit the handler by computing fill values.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            All columns to consider
        numeric_columns : List[str], optional
            Numeric columns (for strategy-specific handling)
        categorical_columns : List[str], optional
            Categorical columns (for strategy-specific handling)

        Returns
        -------
        MissingValueHandler
            Self for method chaining
        """
        numeric_columns = numeric_columns or []
        categorical_columns = categorical_columns or []

        if self.strategy in ['infer', 'fill_mean', 'fill_median']:
            for col in numeric_columns:
                if col in df.columns:
                    if self.strategy == 'fill_median':
                        self.fill_values_[col] = df[col].median()
                    else:
                        self.fill_values_[col] = df[col].mean()

        if self.strategy in ['infer', 'fill_mode']:
            for col in categorical_columns:
                if col in df.columns:
                    mode = df[col].mode()
                    self.fill_values_[col] = mode[0] if len(mode) > 0 else None

        self.is_fitted = True

        if self.verbose:
            print(f"{datetime.now()}: Fitted missing value handler with strategy '{self.strategy}'.")

        return self

    def transform(
        self,
        df: pd.DataFrame,
        columns: List[str],
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply the missing value strategy.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            All columns to consider
        numeric_columns : List[str], optional
            Numeric columns
        categorical_columns : List[str], optional
            Categorical columns

        Returns
        -------
        pd.DataFrame
            Dataframe with missing values handled
        """
        if not self.is_fitted and self.strategy not in ['drop_row', 'drop_column', 'fill_value']:
            raise RuntimeError("Handler must be fitted before transform for this strategy")

        numeric_columns = numeric_columns or []
        categorical_columns = categorical_columns or []

        df_result = df.copy()

        if self.strategy == 'drop_row':
            n_before = len(df_result)
            df_result = df_result.dropna()
            n_dropped = n_before - len(df_result)
            if self.verbose:
                print(f"{datetime.now()}: Dropped {n_dropped} rows with missing values.")

        elif self.strategy == 'drop_column':
            n_before = len(df_result.columns)
            thresh = int(self.threshold * len(df_result))
            df_result = df_result.dropna(axis=1, thresh=thresh)
            n_dropped = n_before - len(df_result.columns)
            if self.verbose:
                print(f"{datetime.now()}: Dropped {n_dropped} columns exceeding {self.threshold} missing ratio.")

        elif self.strategy == 'fill_value':
            df_result = df_result.fillna(self.fill_value)
            if self.verbose:
                print(f"{datetime.now()}: Filled missing values with {self.fill_value}.")

        elif self.strategy in ['infer', 'fill_mean', 'fill_median']:
            for col in numeric_columns:
                if col in df_result.columns and col in self.fill_values_:
                    n_missing = df_result[col].isna().sum()
                    if n_missing > 0:
                        df_result[col] = df_result[col].fillna(self.fill_values_[col])
                        if self.verbose:
                            print(f"{datetime.now()}: Filled {n_missing} NaN values in '{col}' with {self.fill_values_[col]:.4f}.")

            if self.strategy in ['infer', 'fill_mode']:
                for col in categorical_columns:
                    if col in df_result.columns and col in self.fill_values_:
                        n_missing = df_result[col].isna().sum()
                        if n_missing > 0 and self.fill_values_[col] is not None:
                            df_result[col] = df_result[col].fillna(self.fill_values_[col])
                            if self.verbose:
                                print(f"{datetime.now()}: Filled {n_missing} NaN values in '{col}' with mode '{self.fill_values_[col]}'.")

        elif self.strategy == 'fill_mode':
            for col in columns:
                if col in df_result.columns and col in self.fill_values_:
                    n_missing = df_result[col].isna().sum()
                    if n_missing > 0 and self.fill_values_[col] is not None:
                        df_result[col] = df_result[col].fillna(self.fill_values_[col])

        return df_result

    def fit_transform(
        self,
        df: pd.DataFrame,
        columns: List[str],
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            All columns
        numeric_columns : List[str], optional
            Numeric columns
        categorical_columns : List[str], optional
            Categorical columns

        Returns
        -------
        pd.DataFrame
            Transformed dataframe
        """
        self.fit(df, columns, numeric_columns, categorical_columns)
        return self.transform(df, columns, numeric_columns, categorical_columns)

    def get_params(self) -> Dict[str, Any]:
        """Get handler parameters."""
        return {
            'strategy': self.strategy,
            'threshold': self.threshold,
            'fill_value': self.fill_value,
            'fill_values': self.fill_values_
        }
