"""
TaGra Graph Construction Module.

This module provides extensible graph construction methods for building
graphs from preprocessed tabular data.

The module supports three built-in construction methods:
- KNN: Connect each node to its k nearest neighbors
- Distance: Connect nodes within a distance threshold
- Similarity: Connect nodes with cosine similarity above threshold

Custom construction methods can be added via the ConstructorRegistry.

Examples
--------
Basic usage with the convenience function:

>>> from tagra.construction import build_graph
>>> graph = build_graph(df, preprocessed_df, method='knn', k=5)

Using constructors directly:

>>> from tagra.construction import KNNConstructor
>>> constructor = KNNConstructor(k=5)
>>> constructor.construct(G, values)

Registering a custom constructor:

>>> from tagra.construction import ConstructorRegistry, GraphConstructor
>>> class MyConstructor(GraphConstructor):
...     @property
...     def method_name(self):
...         return "custom"
...     def construct(self, G, values, **kwargs):
...         # Custom edge creation logic
...         pass
>>> ConstructorRegistry.register(MyConstructor)
"""

from datetime import datetime
import os
import pickle
from typing import Optional, Union, List

import networkx as nx
import numpy as np
import pandas as pd

from .base import GraphConstructor
from .knn import KNNConstructor
from .distance import DistanceThresholdConstructor
from .similarity import SimilarityThresholdConstructor
from .registry import ConstructorRegistry

from ..core import TaGraGraph, GraphMetadata, ConstructionMethod


def build_graph(
    input_dataframe: Optional[Union[str, pd.DataFrame]] = None,
    preprocessed_dataframe: Optional[Union[str, pd.DataFrame]] = None,
    inferred_columns_filename: Optional[str] = None,
    numeric_columns: Optional[List[str]] = None,
    output_directory: Optional[str] = None,
    graph_filename: Optional[str] = None,
    method: str = "knn",
    k: int = 5,
    distance_threshold: Optional[float] = None,
    similarity_threshold: Optional[float] = None,
    verbose: bool = True,
    overwrite: bool = False,
) -> TaGraGraph:
    """
    Build a graph from dataframes using the specified construction method.

    This is the main entry point for graph construction in TaGra. It creates
    a TaGraGraph with nodes corresponding to dataframe rows and edges based
    on the specified construction method.

    Parameters
    ----------
    input_dataframe : str or pd.DataFrame, optional
        Original dataframe (path or object) - used for node attributes
    preprocessed_dataframe : str or pd.DataFrame, optional
        Preprocessed dataframe (path or object) - used for edge computation
    inferred_columns_filename : str, optional
        Path to pickle file with inferred column information
    numeric_columns : List[str], optional
        Columns to use for distance/similarity computation.
        If None, uses all numeric columns from preprocessed dataframe.
    output_directory : str, optional
        Directory to save the output graph
    graph_filename : str, optional
        Name of output file (must end with .graphml)
    method : str, default='knn'
        Construction method: 'knn', 'distance', or 'similarity'
    k : int, default=5
        Number of neighbors for KNN method
    distance_threshold : float, optional
        Threshold for distance method
    similarity_threshold : float, optional
        Threshold for similarity method
    verbose : bool, default=True
        Print progress messages
    overwrite : bool, default=False
        Overwrite existing files

    Returns
    -------
    TaGraGraph
        Constructed graph with metadata

    Raises
    ------
    ValueError
        If required parameters are missing or invalid

    Examples
    --------
    >>> graph = build_graph(
    ...     input_dataframe=df,
    ...     preprocessed_dataframe=df_preprocessed,
    ...     method='knn',
    ...     k=5
    ... )
    """
    # Validate inputs
    if input_dataframe is None and preprocessed_dataframe is None:
        raise ValueError("Either input_dataframe or preprocessed_dataframe must be provided.")

    # Output path management
    output_directory = output_directory or "./"
    os.makedirs(output_directory, exist_ok=True)

    if graph_filename is None:
        base = (
            os.path.splitext(os.path.basename(input_dataframe))[0]
            if isinstance(input_dataframe, str)
            else "graph"
        )
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        graph_filename = f"{base}.graphml" if overwrite else f"{base}_{timestamp}.graphml"
    elif not graph_filename.endswith(".graphml"):
        raise ValueError("graph_filename must end with '.graphml'.")

    output_path = os.path.join(output_directory, graph_filename)
    if verbose:
        print(f"{datetime.now()}: Output path: {output_path}.")

    # Load dataframes
    df = _load_dataframe(input_dataframe)
    df_preprocessed = _load_dataframe(preprocessed_dataframe) if preprocessed_dataframe is not None else df.copy()

    # Ensure dataframes have the same number of rows
    if df.shape[0] != df_preprocessed.shape[0]:
        df = df.dropna().copy()
        if verbose:
            print(f"{datetime.now()}: Dropped rows with NaN values from the original dataframe.")

    # Create graph and add nodes
    G = nx.Graph()
    df = df.loc[df_preprocessed.index, :].reset_index(drop=True)
    df_preprocessed = df_preprocessed.reset_index(drop=True)
    for i, row in df.iterrows():
        G.add_node(i, **row.to_dict())

    # Prepare numeric data
    if numeric_columns is None:
        numeric_columns = df_preprocessed.select_dtypes(include=["number"]).columns.tolist()
    values = df_preprocessed[numeric_columns].values

    if values.shape[1] == 0:
        raise ValueError("No numeric columns found in the preprocessed dataframe.")

    if verbose:
        print(f"{datetime.now()}: Using numeric columns: {numeric_columns}")

    # Get constructor for the method
    constructor = _get_constructor(
        method=method,
        k=k,
        distance_threshold=distance_threshold,
        similarity_threshold=similarity_threshold,
        verbose=verbose
    )

    # Build edges
    constructor.construct(G, values)

    # Create TaGraGraph with metadata
    try:
        construction_method = ConstructionMethod(method)
    except ValueError:
        construction_method = ConstructionMethod.KNN

    metadata = GraphMetadata(
        construction_method=construction_method,
        k=k if method == 'knn' else None,
        distance_threshold=distance_threshold if method == 'distance' else None,
        similarity_threshold=similarity_threshold if method == 'similarity' else None,
        numeric_columns=numeric_columns,
        source_file=input_dataframe if isinstance(input_dataframe, str) else None,
        n_original_rows=len(df),
        n_original_columns=len(df.columns)
    )

    tagra_graph = TaGraGraph(graph=G, metadata=metadata)

    # Save graph
    with open(output_path, "wb") as f:
        pickle.dump(G, f)
    if verbose:
        print(f"{datetime.now()}: Saved graph to {output_path}.")

    return tagra_graph


def _load_dataframe(data: Union[str, pd.DataFrame, None]) -> pd.DataFrame:
    """Load a dataframe from a file or return a copy if already a DataFrame."""
    if data is None:
        raise ValueError("Data cannot be None")
    if isinstance(data, str):
        if data.endswith(".pickle"):
            return pd.read_pickle(data)
        elif data.endswith(".csv"):
            return pd.read_csv(data)
        elif data.endswith(".xlsx"):
            return pd.read_excel(data)
        elif data.endswith(".parquet"):
            return pd.read_parquet(data)
        elif data.endswith(".json"):
            return pd.read_json(data)
        else:
            # Default to CSV
            return pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        return data.copy()
    raise ValueError("Input must be a file path or a pandas DataFrame.")


def _get_constructor(
    method: str,
    k: int = 5,
    distance_threshold: Optional[float] = None,
    similarity_threshold: Optional[float] = None,
    verbose: bool = True
) -> GraphConstructor:
    """Get the appropriate constructor for the method."""
    if method == "knn":
        return KNNConstructor(k=k, verbose=verbose)
    elif method == "distance":
        if distance_threshold is None:
            raise ValueError("distance_threshold required for distance method")
        return DistanceThresholdConstructor(
            distance_threshold=distance_threshold,
            verbose=verbose
        )
    elif method == "similarity":
        if similarity_threshold is None:
            raise ValueError("similarity_threshold required for similarity method")
        return SimilarityThresholdConstructor(
            similarity_threshold=similarity_threshold,
            verbose=verbose
        )
    else:
        # Try to get from registry
        constructor = ConstructorRegistry.create(method, verbose=verbose)
        if constructor is None:
            raise ValueError(f"Unsupported method: {method}")
        return constructor


__all__ = [
    'GraphConstructor',
    'KNNConstructor',
    'DistanceThresholdConstructor',
    'SimilarityThresholdConstructor',
    'ConstructorRegistry',
    'build_graph'
]
