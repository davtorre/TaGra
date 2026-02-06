"""
Encoding transformers for categorical columns.

This module provides encoders for transforming categorical data
into numeric representations.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder

from .base import Transformer


class OneHotEncoder(Transformer):
    """
    Encode categorical features using one-hot encoding.

    Uses pandas.get_dummies for one-hot encoding, which creates new
    binary columns for each category.

    Parameters
    ----------
    drop_first : bool, default=False
        Whether to drop the first category to avoid multicollinearity
    verbose : bool, default=True
        Whether to print progress messages

    Attributes
    ----------
    categories_ : Dict[str, List]
        Categories found for each column
    """

    def __init__(self, drop_first: bool = False, verbose: bool = True):
        super().__init__(verbose=verbose)
        self.drop_first = drop_first
        self.categories_: Dict[str, List] = {}

    @property
    def name(self) -> str:
        return "one_hot_encoder"

    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'OneHotEncoder':
        """
        Fit the encoder by learning categories.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Categorical columns to fit on

        Returns
        -------
        OneHotEncoder
            Self for method chaining
        """
        if not columns:
            if self.verbose:
                print(f"{datetime.now()}: No columns provided for one-hot encoding.")
            self.is_fitted = True
            return self

        for col in columns:
            self.categories_[col] = df[col].unique().tolist()

        self.is_fitted = True

        if self.verbose:
            print(f"{datetime.now()}: Fitted one-hot encoder on {len(columns)} columns.")

        return self

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Transform categorical columns to one-hot encoding.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Columns to transform

        Returns
        -------
        pd.DataFrame
            Dataframe with one-hot encoded columns
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted before transform")

        if not columns:
            return df

        df_result = pd.get_dummies(df, columns=columns, drop_first=self.drop_first)

        if self.verbose:
            n_new_cols = len(df_result.columns) - len(df.columns) + len(columns)
            print(f"{datetime.now()}: Applied one-hot encoding, created {n_new_cols} new columns.")

        return df_result

    def get_params(self) -> Dict[str, Any]:
        """Get encoder parameters."""
        return {
            'method': 'one-hot',
            'drop_first': self.drop_first,
            'categories': self.categories_
        }


class LabelEncoder(Transformer):
    """
    Encode categorical features as integers.

    Each category is assigned a unique integer value.

    Parameters
    ----------
    verbose : bool, default=True
        Whether to print progress messages

    Attributes
    ----------
    encoders_ : Dict[str, SKLabelEncoder]
        Fitted label encoders for each column
    """

    def __init__(self, verbose: bool = True):
        super().__init__(verbose=verbose)
        self.encoders_: Dict[str, SKLabelEncoder] = {}

    @property
    def name(self) -> str:
        return "label_encoder"

    def fit(self, df: pd.DataFrame, columns: List[str]) -> 'LabelEncoder':
        """
        Fit label encoders for each column.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Categorical columns to fit on

        Returns
        -------
        LabelEncoder
            Self for method chaining
        """
        if not columns:
            if self.verbose:
                print(f"{datetime.now()}: No columns provided for label encoding.")
            self.is_fitted = True
            return self

        for col in columns:
            encoder = SKLabelEncoder()
            encoder.fit(df[col].astype(str))
            self.encoders_[col] = encoder

        self.is_fitted = True

        if self.verbose:
            print(f"{datetime.now()}: Fitted label encoder on {len(columns)} columns.")

        return self

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Transform categorical columns to integer labels.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Columns to transform

        Returns
        -------
        pd.DataFrame
            Dataframe with label encoded columns
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted before transform")

        if not columns:
            return df

        df_result = df.copy()
        for col in columns:
            if col in self.encoders_:
                df_result[col] = self.encoders_[col].transform(df[col].astype(str))

        if self.verbose:
            print(f"{datetime.now()}: Applied label encoding to {len(columns)} columns.")

        return df_result

    def inverse_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Inverse transform integer labels back to categories.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with encoded columns
        columns : List[str]
            Columns to inverse transform

        Returns
        -------
        pd.DataFrame
            Dataframe with original categories
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted before inverse_transform")

        df_result = df.copy()
        for col in columns:
            if col in self.encoders_:
                df_result[col] = self.encoders_[col].inverse_transform(df[col].astype(int))

        return df_result

    def get_params(self) -> Dict[str, Any]:
        """Get encoder parameters."""
        classes = {}
        for col, encoder in self.encoders_.items():
            classes[col] = encoder.classes_.tolist()
        return {
            'method': 'label',
            'classes': classes
        }


def get_encoder(method: str, verbose: bool = True, **kwargs) -> Transformer:
    """
    Get an encoder by method name.

    Parameters
    ----------
    method : str
        Encoding method: 'one-hot' or 'label'
    verbose : bool, default=True
        Verbosity flag
    **kwargs
        Additional arguments for the encoder

    Returns
    -------
    Transformer
        The appropriate encoder

    Raises
    ------
    ValueError
        If method is not recognized
    """
    if method == 'one-hot':
        return OneHotEncoder(verbose=verbose, **kwargs)
    elif method == 'label':
        return LabelEncoder(verbose=verbose)
    else:
        raise ValueError(f"Unknown encoding method: {method}. Use 'one-hot' or 'label'.")
