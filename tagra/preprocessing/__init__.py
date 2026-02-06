"""
TaGra Preprocessing Module.

This module provides a composable preprocessing pipeline for preparing
tabular data for graph construction.

The module supports:
- Automatic column type inference
- Missing value handling (drop, impute)
- Numeric scaling (standard, minmax)
- Categorical encoding (one-hot, label)
- Manifold learning (UMAP, t-SNE, Isomap)

Examples
--------
Basic usage with the convenience function:

>>> from tagra.preprocessing import preprocess
>>> df_processed, positions = preprocess(
...     df,
...     target_columns=['label'],
...     scaling_method='standard',
...     manifold_method='UMAP'
... )

Using the pipeline directly:

>>> from tagra.preprocessing import PreprocessingPipeline
>>> pipeline = PreprocessingPipeline(
...     scaling_method='standard',
...     encoding_method='one-hot',
...     manifold_method='UMAP'
... )
>>> df_processed, positions = pipeline.fit_transform(df)
"""

import datetime
import os
import pickle
from typing import Optional, Union, List, Tuple, Any

import numpy as np
import pandas as pd

from .base import Transformer
from .pipeline import PreprocessingPipeline
from .scaling import StandardScaler, MinMaxScaler, get_scaler
from .encoding import OneHotEncoder, LabelEncoder, get_encoder
from .missing import MissingValueHandler
from .manifold import ManifoldReducer, get_manifold_reducer
from .inference import ColumnInference, infer_column_types


def preprocess(
    input_dataframe: Optional[Union[str, pd.DataFrame]] = None,
    output_directory: str = "results/",
    preprocessed_filename: Optional[str] = None,
    inferred_columns_filename: Optional[str] = None,
    numeric_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    target_columns: Optional[List[str]] = None,
    unknown_column_action: str = 'infer',
    ignore_columns: Optional[List[str]] = None,
    numeric_threshold: float = 0.05,
    numeric_scaling: str = 'standard',
    categorical_encoding: str = 'one-hot',
    nan_action: str = 'infer',
    nan_threshold: float = 0.5,
    verbose: bool = True,
    manifold_method: str = 'UMAP',
    manifold_dim: int = 2,
    overwrite: bool = False
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Preprocess a dataframe for graph construction.

    This is the main entry point for preprocessing in TaGra. It handles
    loading data, inferring column types, handling missing values,
    scaling, encoding, and manifold learning.

    Parameters
    ----------
    input_dataframe : str or pd.DataFrame
        Input dataframe or path to file
    output_directory : str, default='results/'
        Directory for output files
    preprocessed_filename : str, optional
        Name for preprocessed file
    inferred_columns_filename : str, optional
        Name for column info pickle file
    numeric_columns : List[str], optional
        Columns to treat as numeric
    categorical_columns : List[str], optional
        Columns to treat as categorical
    target_columns : List[str], optional
        Target columns
    unknown_column_action : str, default='infer'
        How to handle unknown columns
    ignore_columns : List[str], optional
        Columns to ignore
    numeric_threshold : float, default=0.05
        Threshold for numeric inference
    numeric_scaling : str, default='standard'
        Scaling method
    categorical_encoding : str, default='one-hot'
        Encoding method
    nan_action : str, default='infer'
        Missing value strategy
    nan_threshold : float, default=0.5
        Threshold for dropping columns
    verbose : bool, default=True
        Print progress
    manifold_method : str, default='UMAP'
        Manifold learning method
    manifold_dim : int, default=2
        Manifold dimensions
    overwrite : bool, default=False
        Overwrite existing files

    Returns
    -------
    Tuple[pd.DataFrame, Optional[np.ndarray]]
        Preprocessed dataframe and manifold positions
    """
    if verbose:
        print(f"--------------------------\nPreprocessing options\n--------------------------\n\n"
              f"\tOptions:\n"
              f"\tinput_path: {input_dataframe}, output_directory: {output_directory}, "
              f"preprocessed_filename: {preprocessed_filename}\n"
              f"\tnumeric_columns: {numeric_columns}, categorical_columns: {categorical_columns}, "
              f"target_columns: {target_columns}, \n"
              f"\tunknown_column_action: {unknown_column_action}, ignore_columns: {ignore_columns}, \n"
              f"\tnumeric_threshold: {numeric_threshold}, numeric_scaling: {numeric_scaling}, \n"
              f"\tcategorical_encoding: {categorical_encoding}, nan_action: {nan_action}, \n"
              f"\tnan_threshold: {nan_threshold}, verbose: {verbose}, \n"
              f"\tmanifold_method: {manifold_method}, manifold_dim: {manifold_dim}\n")

    # Output path management
    if output_directory is None:
        output_directory = './'
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
        print(f"{datetime.datetime.now()}: Output directory created: {output_directory}.")

    # Load dataframe
    df = _load_input(input_dataframe)

    # Generate output filename
    if preprocessed_filename is None:
        if isinstance(input_dataframe, str):
            basename = os.path.basename(input_dataframe)
            base, ext = os.path.splitext(basename)
            if overwrite:
                preprocessed_filename = f"{base}_preprocessed{ext}"
            else:
                preprocessed_filename = f"{base}_preprocessed_{datetime.datetime.now().strftime('%Y%m%d%H%M')}{ext}"
        else:
            if overwrite:
                preprocessed_filename = "preprocessed.pickle"
            else:
                preprocessed_filename = f"preprocessed_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.pickle"

    output_path = os.path.join(output_directory, preprocessed_filename)
    if verbose:
        print(f"{datetime.datetime.now()}: Output path for the preprocessed file: {output_path}.")

    # Set up column info path
    inferred_columns_dictionary_path = None
    if inferred_columns_filename is not None:
        if not inferred_columns_filename.endswith('.pickle'):
            raise ValueError("Invalid inferred_columns_filename. Must be a path to a pickle file.")
        inferred_columns_dictionary_path = os.path.join(output_directory, inferred_columns_filename)
        if verbose:
            print(f"{datetime.datetime.now()}: Inferred columns dictionary path: {inferred_columns_dictionary_path}.")

    # Ensure columns are lists
    numeric_columns = _ensure_list(numeric_columns)
    categorical_columns = _ensure_list(categorical_columns)
    target_columns = _ensure_list(target_columns)
    ignore_columns = _ensure_list(ignore_columns)

    # Validate columns exist
    for cols, name in [
        (numeric_columns, "Numeric column"),
        (categorical_columns, "Categorical column"),
        (target_columns, "Target column"),
        (ignore_columns, "Ignore column")
    ]:
        for col in cols:
            if col not in df.columns:
                raise ValueError(f"{name} {col} not found.")

    # Create and run pipeline
    pipeline = PreprocessingPipeline(
        scaling_method=numeric_scaling,
        encoding_method=categorical_encoding,
        nan_action=nan_action,
        nan_threshold=nan_threshold,
        manifold_method=manifold_method,
        manifold_dim=manifold_dim,
        verbose=verbose
    )

    # Print report
    if verbose:
        print(f"--------------------------\nDataframe short report\n--------------------------\n\n")
        print(f"{df.shape[0]} rows and {df.shape[1]} columns")
        print(f"column list: {list(df.columns)}")
        print(f"nans:\n{df.isna().sum()}")

    df_result, manifold_positions = pipeline.fit_transform(
        df,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        target_columns=target_columns,
        ignore_columns=ignore_columns,
        unknown_column_action=unknown_column_action,
        numeric_threshold=numeric_threshold
    )

    # Print target proportions
    if target_columns:
        target_col_name = tuple(target_columns) if len(target_columns) > 1 else target_columns[0]
        if target_col_name in df_result.columns:
            unique_targets = np.unique(df_result[target_col_name].values)
            N_col = df_result.shape[0]
            print(f"Target class proportions")
            for target in unique_targets:
                n_target = df_result[df_result[target_col_name] == target].shape[0]
                print(f"\t{target}: {n_target / N_col * 100}%")
        print(f"--------------------------\nEnd of the report.")

    # Save column info
    if inferred_columns_dictionary_path is not None:
        pipeline.save_column_info(inferred_columns_dictionary_path)

    # Save preprocessed dataframe
    _save_output(df_result, output_path, verbose)

    return df_result, manifold_positions


def _load_input(input_dataframe: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """Load dataframe from file or return copy."""
    if isinstance(input_dataframe, str):
        if input_dataframe.endswith('.csv'):
            peek_df = pd.read_csv(input_dataframe, nrows=1)
            if peek_df.columns[0].startswith('Unnamed') or peek_df.columns[0].isdigit():
                return pd.read_csv(input_dataframe, index_col=0)
            return pd.read_csv(input_dataframe)
        elif input_dataframe.endswith('.xlsx'):
            return pd.read_excel(input_dataframe, index_col=None)
        elif input_dataframe.endswith('.pickle'):
            return pd.read_pickle(input_dataframe)
        elif input_dataframe.endswith('.json'):
            return pd.read_json(input_dataframe)
        elif input_dataframe.endswith('.parquet'):
            return pd.read_parquet(input_dataframe)
        elif input_dataframe.endswith('.hdf') or input_dataframe.endswith('.h5'):
            return pd.read_hdf(input_dataframe)
        else:
            supported = ", ".join(["CSV", "Excel (.xlsx)", "Pickle", "JSON", "Parquet", "HDF5 (.hdf, .h5)"])
            raise ValueError(f"Unsupported format. Use: {supported}")
    elif isinstance(input_dataframe, pd.DataFrame):
        return input_dataframe.copy()
    else:
        raise ValueError("Invalid input_path. Must be a path to a file or a pandas DataFrame.")


def _save_output(df: pd.DataFrame, output_path: str, verbose: bool) -> None:
    """Save dataframe to file."""
    if output_path.endswith('.pickle'):
        df.to_pickle(output_path)
    elif output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    elif output_path.endswith('.xlsx'):
        df.to_excel(output_path, index=False)
    elif output_path.endswith('.json'):
        df.to_json(output_path)
    elif output_path.endswith('.parquet'):
        df.to_parquet(output_path, index=False)
    elif output_path.endswith('.hdf') or output_path.endswith('.h5'):
        df.to_hdf(output_path, key='data', index=False)
    if verbose:
        print(f"{datetime.datetime.now()}: Saved preprocessed DataFrame to {output_path}.")


def _ensure_list(value: Any) -> List:
    """Ensure value is a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


__all__ = [
    'preprocess',
    'Transformer',
    'PreprocessingPipeline',
    'StandardScaler',
    'MinMaxScaler',
    'get_scaler',
    'OneHotEncoder',
    'LabelEncoder',
    'get_encoder',
    'MissingValueHandler',
    'ManifoldReducer',
    'get_manifold_reducer',
    'ColumnInference',
    'infer_column_types'
]
