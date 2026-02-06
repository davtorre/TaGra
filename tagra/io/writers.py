"""
Data writers for TaGra.

This module provides unified file writing functionality for dataframes
and graphs.
"""

from datetime import datetime
from typing import Optional, Union, Dict, Any
import os
import pickle

import pandas as pd
import networkx as nx

from ..exceptions import IOError


def save_dataframe(
    df: pd.DataFrame,
    filepath: str,
    format: Optional[str] = None,
    verbose: bool = True,
    **kwargs
) -> None:
    """
    Save a dataframe to file.

    Supports CSV, Excel, Pickle, JSON, Parquet, and HDF5 formats.
    Format is auto-detected from file extension if not specified.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save
    filepath : str
        Output file path
    format : str, optional
        File format. If None, auto-detected from extension.
    verbose : bool, default=True
        Print progress messages
    **kwargs
        Additional arguments passed to the pandas writer

    Raises
    ------
    IOError
        If file cannot be written

    Examples
    --------
    >>> save_dataframe(df, 'output.csv')
    >>> save_dataframe(df, 'output.xlsx', sheet_name='Results')
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    # Auto-detect format from extension
    if format is None:
        ext = os.path.splitext(filepath)[1].lower()
        format = _detect_format(ext)
        if format is None:
            raise IOError(f"Unsupported file extension: {ext}")

    if verbose:
        print(f"{datetime.now()}: Saving dataframe to: {filepath}")

    try:
        if format == 'csv':
            index = kwargs.pop('index', False)
            df.to_csv(filepath, index=index, **kwargs)
        elif format == 'excel':
            index = kwargs.pop('index', False)
            df.to_excel(filepath, index=index, **kwargs)
        elif format == 'pickle':
            df.to_pickle(filepath, **kwargs)
        elif format == 'json':
            df.to_json(filepath, **kwargs)
        elif format == 'parquet':
            index = kwargs.pop('index', False)
            df.to_parquet(filepath, index=index, **kwargs)
        elif format == 'hdf5':
            key = kwargs.pop('key', 'data')
            df.to_hdf(filepath, key=key, **kwargs)
        else:
            raise IOError(f"Unsupported format: {format}")
    except Exception as e:
        raise IOError(f"Failed to save dataframe to {filepath}: {str(e)}")


def _detect_format(ext: str) -> Optional[str]:
    """Detect format from file extension."""
    format_map = {
        '.csv': 'csv',
        '.xlsx': 'excel', '.xls': 'excel',
        '.pickle': 'pickle', '.pkl': 'pickle',
        '.json': 'json',
        '.parquet': 'parquet',
        '.hdf': 'hdf5', '.h5': 'hdf5', '.hdf5': 'hdf5'
    }
    return format_map.get(ext)


def save_graph(
    graph: Union[nx.Graph, 'TaGraGraph'],
    filepath: str,
    format: str = 'pickle',
    verbose: bool = True
) -> None:
    """
    Save a graph to file.

    Parameters
    ----------
    graph : nx.Graph or TaGraGraph
        Graph to save
    filepath : str
        Output file path
    format : str, default='pickle'
        Format: 'pickle', 'graphml', 'gexf', 'edgelist'
    verbose : bool, default=True
        Print progress messages

    Raises
    ------
    IOError
        If file cannot be written

    Examples
    --------
    >>> save_graph(G, 'graph.pickle')
    >>> save_graph(G, 'graph.graphml', format='graphml')
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    if verbose:
        print(f"{datetime.now()}: Saving graph to: {filepath}")

    # Handle TaGraGraph
    if hasattr(graph, 'to_networkx'):
        nx_graph = graph.to_networkx()
    else:
        nx_graph = graph

    try:
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(nx_graph, f)
        elif format == 'graphml':
            # Note: NetworkX graphml doesn't support all attribute types
            # For full fidelity, use pickle
            nx.write_graphml(nx_graph, filepath)
        elif format == 'gexf':
            nx.write_gexf(nx_graph, filepath)
        elif format == 'edgelist':
            nx.write_edgelist(nx_graph, filepath)
        else:
            raise IOError(f"Unsupported graph format: {format}")

        if verbose:
            print(f"{datetime.now()}: Graph saved ({nx_graph.number_of_nodes()} nodes, "
                  f"{nx_graph.number_of_edges()} edges)")
    except Exception as e:
        raise IOError(f"Failed to save graph to {filepath}: {str(e)}")


def save_column_info(
    column_info: Dict[str, Any],
    filepath: str,
    verbose: bool = True
) -> None:
    """
    Save column information to pickle file.

    Parameters
    ----------
    column_info : Dict[str, Any]
        Column information dictionary
    filepath : str
        Output file path
    verbose : bool, default=True
        Print progress messages

    Examples
    --------
    >>> save_column_info(info, 'columns.pickle')
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    if verbose:
        print(f"{datetime.now()}: Saving column info to: {filepath}")

    try:
        with open(filepath, 'wb') as f:
            pickle.dump(column_info, f)
    except Exception as e:
        raise IOError(f"Failed to save column info to {filepath}: {str(e)}")


def generate_timestamped_filename(
    base: str,
    ext: str,
    overwrite: bool = False
) -> str:
    """
    Generate a filename with optional timestamp.

    Parameters
    ----------
    base : str
        Base filename (without extension)
    ext : str
        File extension (with or without dot)
    overwrite : bool, default=False
        If True, don't add timestamp

    Returns
    -------
    str
        Generated filename

    Examples
    --------
    >>> generate_timestamped_filename('output', '.csv')
    'output_202401151230.csv'
    >>> generate_timestamped_filename('output', 'csv', overwrite=True)
    'output.csv'
    """
    if not ext.startswith('.'):
        ext = '.' + ext

    if overwrite:
        return f"{base}{ext}"
    else:
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        return f"{base}_{timestamp}{ext}"
