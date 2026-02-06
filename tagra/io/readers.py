"""
Data readers for TaGra.

This module provides unified file reading functionality supporting
multiple data formats.
"""

from datetime import datetime
from typing import Optional, Union, Dict, Any
import os
import pickle

import pandas as pd
import networkx as nx

from ..exceptions import IOError


SUPPORTED_FORMATS = {
    'csv': ['.csv'],
    'excel': ['.xlsx', '.xls'],
    'pickle': ['.pickle', '.pkl'],
    'json': ['.json'],
    'parquet': ['.parquet'],
    'hdf5': ['.hdf', '.h5', '.hdf5']
}


def read_dataframe(
    filepath: str,
    format: Optional[str] = None,
    verbose: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Read a dataframe from file.

    Supports CSV, Excel, Pickle, JSON, Parquet, and HDF5 formats.
    Format is auto-detected from file extension if not specified.

    Parameters
    ----------
    filepath : str
        Path to the data file
    format : str, optional
        File format. If None, auto-detected from extension.
        Options: 'csv', 'excel', 'pickle', 'json', 'parquet', 'hdf5'
    verbose : bool, default=True
        Print progress messages
    **kwargs
        Additional arguments passed to the pandas reader

    Returns
    -------
    pd.DataFrame
        Loaded dataframe

    Raises
    ------
    IOError
        If file cannot be read or format is unsupported

    Examples
    --------
    >>> df = read_dataframe('data.csv')
    >>> df = read_dataframe('data.xlsx', sheet_name='Sheet1')
    """
    if not os.path.exists(filepath):
        raise IOError(f"File not found: {filepath}")

    # Auto-detect format from extension
    if format is None:
        ext = os.path.splitext(filepath)[1].lower()
        format = _detect_format(ext)
        if format is None:
            supported = ", ".join(sum(SUPPORTED_FORMATS.values(), []))
            raise IOError(f"Unsupported file extension: {ext}. Supported: {supported}")

    if verbose:
        print(f"{datetime.now()}: Reading {format} file: {filepath}")

    try:
        if format == 'csv':
            return _read_csv(filepath, **kwargs)
        elif format == 'excel':
            return pd.read_excel(filepath, **kwargs)
        elif format == 'pickle':
            return pd.read_pickle(filepath, **kwargs)
        elif format == 'json':
            return pd.read_json(filepath, **kwargs)
        elif format == 'parquet':
            return pd.read_parquet(filepath, **kwargs)
        elif format == 'hdf5':
            key = kwargs.pop('key', 'data')
            return pd.read_hdf(filepath, key=key, **kwargs)
        else:
            raise IOError(f"Unsupported format: {format}")
    except Exception as e:
        raise IOError(f"Failed to read file {filepath}: {str(e)}")


def _read_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Read CSV with smart index detection.

    Checks if the first column looks like an index (unnamed or numeric)
    and handles it appropriately.
    """
    # Peek at first row to check for index column
    if 'index_col' not in kwargs:
        peek_df = pd.read_csv(filepath, nrows=1)
        first_col = peek_df.columns[0]
        if first_col.startswith('Unnamed') or first_col.isdigit():
            kwargs['index_col'] = 0
    return pd.read_csv(filepath, **kwargs)


def _detect_format(ext: str) -> Optional[str]:
    """Detect format from file extension."""
    for format_name, extensions in SUPPORTED_FORMATS.items():
        if ext in extensions:
            return format_name
    return None


def read_graph(
    filepath: str,
    verbose: bool = True
) -> nx.Graph:
    """
    Read a graph from file.

    Supports pickle (.graphml, .pickle) and GraphML formats.

    Parameters
    ----------
    filepath : str
        Path to the graph file
    verbose : bool, default=True
        Print progress messages

    Returns
    -------
    nx.Graph
        Loaded graph

    Raises
    ------
    IOError
        If file cannot be read

    Examples
    --------
    >>> G = read_graph('graph.graphml')
    """
    if not os.path.exists(filepath):
        raise IOError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if verbose:
        print(f"{datetime.now()}: Reading graph from: {filepath}")

    try:
        if ext in ['.pickle', '.pkl', '.graphml']:
            # TaGra saves graphs as pickled NetworkX objects with .graphml extension
            with open(filepath, 'rb') as f:
                graph = pickle.load(f)
            if isinstance(graph, nx.Graph):
                return graph
            elif isinstance(graph, dict) and 'graph' in graph:
                # New TaGraGraph format
                return graph['graph']
            else:
                raise IOError(f"Unexpected content in pickle file: {type(graph)}")
        elif ext == '.gml':
            return nx.read_gml(filepath)
        elif ext == '.gexf':
            return nx.read_gexf(filepath)
        elif ext == '.edgelist':
            return nx.read_edgelist(filepath)
        else:
            raise IOError(f"Unsupported graph format: {ext}")
    except Exception as e:
        raise IOError(f"Failed to read graph from {filepath}: {str(e)}")


def read_column_info(
    filepath: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Read column information from pickle file.

    Parameters
    ----------
    filepath : str
        Path to the pickle file
    verbose : bool, default=True
        Print progress messages

    Returns
    -------
    Dict[str, Any]
        Column information dictionary

    Examples
    --------
    >>> info = read_column_info('columns.pickle')
    >>> print(info['numeric_columns'])
    """
    if not os.path.exists(filepath):
        raise IOError(f"File not found: {filepath}")

    if verbose:
        print(f"{datetime.now()}: Reading column info from: {filepath}")

    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise IOError(f"Failed to read column info from {filepath}: {str(e)}")


def get_supported_formats() -> Dict[str, list]:
    """
    Get dictionary of supported file formats.

    Returns
    -------
    Dict[str, list]
        Format names mapped to their extensions
    """
    return SUPPORTED_FORMATS.copy()
