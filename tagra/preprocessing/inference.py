"""
Column type inference utilities.

This module provides utilities for automatically inferring column types
(numeric, categorical, target, ignore) from dataframe content.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


class ColumnInference:
    """
    Infer column types from dataframe content.

    Automatically classifies columns as numeric, categorical, or to be
    ignored based on their data types and content.

    Parameters
    ----------
    numeric_threshold : float, default=0.05
        Unique ratio threshold for classifying ambiguous columns as numeric
        (higher ratio = more unique values = more likely numeric)
    verbose : bool, default=True
        Whether to print progress messages

    Attributes
    ----------
    numeric_columns : List[str]
        Inferred numeric columns
    categorical_columns : List[str]
        Inferred categorical columns
    ignore_columns : List[str]
        Columns to ignore (datetime, bool, etc.)
    """

    def __init__(
        self,
        numeric_threshold: float = 0.05,
        verbose: bool = True
    ):
        self.numeric_threshold = numeric_threshold
        self.verbose = verbose
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.ignore_columns: List[str] = []

    def infer(
        self,
        df: pd.DataFrame,
        known_numeric: Optional[List[str]] = None,
        known_categorical: Optional[List[str]] = None,
        known_target: Optional[List[str]] = None,
        known_ignore: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Infer column types from the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        known_numeric : List[str], optional
            Columns explicitly marked as numeric
        known_categorical : List[str], optional
            Columns explicitly marked as categorical
        known_target : List[str], optional
            Target columns (will be added to ignore)
        known_ignore : List[str], optional
            Columns to ignore

        Returns
        -------
        Dict[str, List[str]]
            Dictionary with 'numeric', 'categorical', and 'ignore' lists
        """
        known_numeric = known_numeric or []
        known_categorical = known_categorical or []
        known_target = known_target or []
        known_ignore = known_ignore or []

        # Start with known columns
        self.numeric_columns = list(known_numeric)
        self.categorical_columns = list(known_categorical)
        self.ignore_columns = list(known_ignore) + list(known_target)

        # Infer unknown columns
        for col in df.columns:
            if col in self.numeric_columns or col in self.categorical_columns or col in self.ignore_columns:
                continue

            col_type = self._infer_column_type(df[col])

            if col_type == 'numeric':
                self.numeric_columns.append(col)
                if self.verbose:
                    print(f"{datetime.now()}: Column '{col}' inferred as numeric.")
            elif col_type == 'categorical':
                self.categorical_columns.append(col)
                if self.verbose:
                    print(f"{datetime.now()}: Column '{col}' inferred as categorical.")
            else:
                self.ignore_columns.append(col)
                if self.verbose:
                    print(f"{datetime.now()}: Column '{col}' inferred as ignore.")

        return {
            'numeric': self.numeric_columns,
            'categorical': self.categorical_columns,
            'ignore': self.ignore_columns
        }

    def _infer_column_type(self, series: pd.Series) -> str:
        """
        Infer the type of a single column.

        Parameters
        ----------
        series : pd.Series
            Column data

        Returns
        -------
        str
            Column type: 'numeric', 'categorical', or 'ignore'
        """
        dtype = series.dtype

        # Numeric types
        if dtype in [np.float64, np.float32, np.int64, np.int32, np.float16, np.int16, np.int8]:
            return 'numeric'

        # Boolean and datetime -> ignore
        if dtype == 'bool' or np.issubdtype(dtype, np.datetime64):
            return 'ignore'

        # Object (string) types -> categorical
        if dtype == 'object':
            return 'categorical'

        # For other types, use unique ratio heuristic
        unique_ratio = len(series.unique()) / len(series) if len(series) > 0 else 0

        if unique_ratio > self.numeric_threshold:
            return 'numeric'
        else:
            return 'categorical'


def infer_column_types(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    target_columns: Optional[List[str]] = None,
    ignore_columns: Optional[List[str]] = None,
    numeric_threshold: float = 0.05,
    verbose: bool = True
) -> Dict[str, List[str]]:
    """
    Convenience function to infer column types.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    numeric_columns : List[str], optional
        Known numeric columns
    categorical_columns : List[str], optional
        Known categorical columns
    target_columns : List[str], optional
        Target columns
    ignore_columns : List[str], optional
        Columns to ignore
    numeric_threshold : float, default=0.05
        Threshold for unique ratio classification
    verbose : bool, default=True
        Print progress messages

    Returns
    -------
    Dict[str, List[str]]
        Dictionary with 'numeric', 'categorical', and 'ignore' lists
    """
    inference = ColumnInference(
        numeric_threshold=numeric_threshold,
        verbose=verbose
    )
    return inference.infer(
        df,
        known_numeric=numeric_columns,
        known_categorical=categorical_columns,
        known_target=target_columns,
        known_ignore=ignore_columns
    )


def validate_columns(
    df: pd.DataFrame,
    columns: List[str],
    column_type: str = "column"
) -> List[str]:
    """
    Validate that columns exist in dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        Columns to validate
    column_type : str, default="column"
        Type name for error messages

    Returns
    -------
    List[str]
        Validated columns

    Raises
    ------
    ValueError
        If any column is not found
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{column_type} not found: {missing}")
    return columns


def ensure_list(value: Any) -> List:
    """
    Ensure a value is a list.

    Parameters
    ----------
    value : Any
        Value that may be a single item or list

    Returns
    -------
    List
        The value as a list
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
