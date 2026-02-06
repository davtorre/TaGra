"""
Preprocessing pipeline for composable transformations.

This module provides a pipeline class that chains multiple
preprocessing transformers together.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import pickle
import os

from .base import Transformer
from .scaling import get_scaler
from .encoding import get_encoder
from .missing import MissingValueHandler
from .manifold import ManifoldReducer
from .inference import infer_column_types, ensure_list, validate_columns


class PreprocessingPipeline:
    """
    A composable preprocessing pipeline.

    Chains together multiple preprocessing steps: column inference,
    missing value handling, scaling, encoding, and manifold learning.

    Parameters
    ----------
    scaling_method : str, default='standard'
        Method for scaling numeric columns: 'standard' or 'minmax'
    encoding_method : str, default='one-hot'
        Method for encoding categorical columns: 'one-hot' or 'label'
    nan_action : str, default='infer'
        Strategy for handling missing values
    nan_threshold : float, default=0.5
        Threshold for nan_action='drop column'
    manifold_method : str, optional
        Manifold learning method: 'UMAP', 'TSNE', 'Isomap', or None
    manifold_dim : int, default=2
        Dimensionality for manifold learning
    verbose : bool, default=True
        Print progress messages

    Attributes
    ----------
    transformers : List[Transformer]
        List of transformers in the pipeline
    column_info : Dict[str, List[str]]
        Information about column classifications
    manifold_positions : Optional[np.ndarray]
        Manifold embedding positions
    """

    def __init__(
        self,
        scaling_method: str = 'standard',
        encoding_method: str = 'one-hot',
        nan_action: str = 'infer',
        nan_threshold: float = 0.5,
        manifold_method: Optional[str] = 'UMAP',
        manifold_dim: int = 2,
        verbose: bool = True
    ):
        self.scaling_method = scaling_method
        self.encoding_method = encoding_method
        self.nan_action = nan_action
        self.nan_threshold = nan_threshold
        self.manifold_method = manifold_method
        self.manifold_dim = manifold_dim
        self.verbose = verbose

        self.transformers: List[Transformer] = []
        self.column_info: Dict[str, List[str]] = {}
        self.manifold_positions: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        ignore_columns: Optional[List[str]] = None,
        unknown_column_action: str = 'infer',
        numeric_threshold: float = 0.05
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        Fit the pipeline and transform the data.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        numeric_columns : List[str], optional
            Columns to treat as numeric
        categorical_columns : List[str], optional
            Columns to treat as categorical
        target_columns : List[str], optional
            Target columns (not preprocessed but kept)
        ignore_columns : List[str], optional
            Columns to ignore
        unknown_column_action : str, default='infer'
            How to handle columns not in any list: 'infer' or 'ignore'
        numeric_threshold : float, default=0.05
            Threshold for inferring numeric columns

        Returns
        -------
        Tuple[pd.DataFrame, Optional[np.ndarray]]
            Preprocessed dataframe and manifold positions (if any)
        """
        # Ensure lists
        numeric_columns = ensure_list(numeric_columns)
        categorical_columns = ensure_list(categorical_columns)
        target_columns = ensure_list(target_columns)
        ignore_columns = ensure_list(ignore_columns)

        # Validate known columns
        for cols, name in [
            (numeric_columns, "Numeric columns"),
            (categorical_columns, "Categorical columns"),
            (target_columns, "Target columns"),
            (ignore_columns, "Ignore columns")
        ]:
            if cols:
                validate_columns(df, cols, name)

        # Infer or ignore unknown columns
        if unknown_column_action == 'infer':
            inferred = infer_column_types(
                df,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                target_columns=target_columns,
                ignore_columns=ignore_columns,
                numeric_threshold=numeric_threshold,
                verbose=self.verbose
            )
            numeric_columns = inferred['numeric']
            categorical_columns = inferred['categorical']
            ignore_columns = inferred['ignore']
        elif unknown_column_action == 'ignore':
            # Add unknown columns to ignore
            known = set(numeric_columns + categorical_columns + target_columns + ignore_columns)
            unknown = [col for col in df.columns if col not in known]
            ignore_columns = ignore_columns + unknown

        # Store column info
        self.column_info = {
            'numeric': numeric_columns,
            'categorical': categorical_columns,
            'target': target_columns,
            'ignore': ignore_columns
        }

        # Start with a copy
        df_result = df.copy()

        # Handle target columns - combine if multiple
        target_col_name = None
        if target_columns:
            if len(target_columns) > 1:
                target_col_name = tuple(target_columns)
                df_result[target_col_name] = df_result[target_columns].apply(tuple, axis=1)
                df_result = df_result.drop(columns=target_columns)
            else:
                target_col_name = target_columns[0]

        # Step 1: Handle missing values
        missing_handler = MissingValueHandler(
            strategy=self.nan_action,
            threshold=self.nan_threshold,
            verbose=self.verbose
        )
        df_result = missing_handler.fit_transform(
            df_result,
            columns=numeric_columns + categorical_columns,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns
        )
        self.transformers.append(missing_handler)

        # Step 2: Scale numeric columns
        if numeric_columns and self.scaling_method:
            scaler = get_scaler(self.scaling_method, verbose=self.verbose)
            df_result = scaler.fit_transform(df_result, numeric_columns)
            self.transformers.append(scaler)

        # Step 3: Encode categorical columns
        if categorical_columns and self.encoding_method:
            encoder = get_encoder(self.encoding_method, verbose=self.verbose)
            df_result = encoder.fit_transform(df_result, categorical_columns)
            self.transformers.append(encoder)

        # Step 4: Apply manifold learning
        if self.manifold_method and numeric_columns:
            # Get current numeric columns (after encoding, new columns may have been added)
            current_numeric = [col for col in df_result.columns
                              if col in numeric_columns or col.startswith(tuple(f"{c}_" for c in categorical_columns))]
            # Filter to only actual numeric columns
            current_numeric = df_result.select_dtypes(include=['number']).columns.tolist()
            # Remove target from manifold
            if target_col_name and target_col_name in current_numeric:
                current_numeric.remove(target_col_name)

            if len(current_numeric) >= self.manifold_dim:
                manifold = ManifoldReducer(
                    method=self.manifold_method,
                    n_components=self.manifold_dim,
                    verbose=self.verbose
                )
                df_result = manifold.fit_transform(df_result, current_numeric)
                self.manifold_positions = manifold.get_embedding()
                self.transformers.append(manifold)

        self._is_fitted = True

        return df_result, self.manifold_positions

    def get_column_info(self) -> Dict[str, List[str]]:
        """
        Get column classification information.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary with column type lists
        """
        return self.column_info.copy()

    def save_column_info(self, filepath: str) -> None:
        """
        Save column info to a pickle file.

        Parameters
        ----------
        filepath : str
            Path for the pickle file
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.column_info, f)
        if self.verbose:
            print(f"{datetime.now()}: Saved column info to {filepath}")

    def get_params(self) -> Dict[str, Any]:
        """
        Get pipeline parameters.

        Returns
        -------
        Dict[str, Any]
            Pipeline parameters
        """
        return {
            'scaling_method': self.scaling_method,
            'encoding_method': self.encoding_method,
            'nan_action': self.nan_action,
            'nan_threshold': self.nan_threshold,
            'manifold_method': self.manifold_method,
            'manifold_dim': self.manifold_dim,
            'transformers': [t.get_params() for t in self.transformers]
        }
