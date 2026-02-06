"""
Neighborhood analysis functions.

This module provides functions for analyzing node neighborhoods
and computing neighborhood-based statistics.
"""

from datetime import datetime
from typing import Dict, Any, Optional, Union, List, Tuple
import random

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def analyze_neighborhoods(
    graph: Union[nx.Graph, 'TaGraGraph'],
    target_attribute: str,
    return_probs: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Analyze attributes in the neighborhoods of each node.

    Parameters
    ----------
    graph : nx.Graph or TaGraGraph
        Graph to analyze
    target_attribute : str
        Node attribute to analyze
    return_probs : bool, default=False
        If True, return probabilities instead of counts
    verbose : bool, default=True
        Print progress messages

    Returns
    -------
    pd.DataFrame
        DataFrame with each row representing a node:
        - node_index: Node identifier
        - node_{attribute}: Node's attribute value
        - degree: Node degree
        - n_{value} or p_{value}: Count or probability of each attribute value

    Examples
    --------
    >>> df_neigh = analyze_neighborhoods(G, 'label')
    >>> print(df_neigh.head())
    """
    NONE_STR = 'None'

    # Handle TaGraGraph
    if hasattr(graph, 'to_networkx'):
        G = graph.to_networkx()
    else:
        G = graph

    if verbose:
        print(f"{datetime.now()}: Analyzing neighborhood attributes for '{target_attribute}'...")

    # Get unique attribute values
    unique_attributes = set([
        G.nodes[n].get(target_attribute, NONE_STR) for n in G.nodes
    ])

    data = []
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        neighbor_attrs = [G.nodes[n].get(target_attribute, NONE_STR) for n in neighbors]

        attr_counts = {}
        attr_counts[f"node_{target_attribute}"] = G.nodes[node].get(target_attribute, NONE_STR)
        attr_counts["node_index"] = node
        attr_counts["degree"] = len(neighbors)

        for attr in unique_attributes:
            if return_probs and attr_counts["degree"] > 0:
                attr_counts[f"p_{attr}"] = neighbor_attrs.count(attr) / attr_counts["degree"]
            else:
                attr_counts[f"n_{attr}"] = neighbor_attrs.count(attr)

        data.append(attr_counts)

    # Build column order
    prefix = 'p' if return_probs else 'n'
    cols = ["node_index", f"node_{target_attribute}", "degree"] + \
           [f"{prefix}_{attr}" for attr in unique_attributes]

    df = pd.DataFrame(data, columns=cols)

    if verbose:
        print(f"{datetime.now()}: Analyzed {len(df)} node neighborhoods.")

    return df


def compute_neighborhood_probabilities(
    df_neigh: pd.DataFrame,
    target_attribute: str
) -> Dict[Tuple, float]:
    """
    Compute neighborhood transition probabilities.

    Computes P(j|i) - the probability that a neighbor of a node with
    attribute i has attribute j.

    Parameters
    ----------
    df_neigh : pd.DataFrame
        Neighborhood analysis dataframe from analyze_neighborhoods()
    target_attribute : str
        The target attribute name

    Returns
    -------
    Dict[Tuple, float]
        Dictionary mapping (i, j) to P(j|i)

    Examples
    --------
    >>> probs = compute_neighborhood_probabilities(df_neigh, 'label')
    >>> print(f"P(B|A) = {probs[('A', 'B')]:.4f}")
    """
    probabilities = {}
    label_col = f'node_{target_attribute}'

    for label_i in df_neigh[label_col].unique():
        nodes_with_label_i = df_neigh[df_neigh[label_col] == label_i]
        total_degree_i = nodes_with_label_i['degree'].sum()

        for label_j in df_neigh[label_col].unique():
            col_name = f'n_{label_j}'
            if col_name in df_neigh.columns:
                total_neighbors_with_label_j = nodes_with_label_i[col_name].sum()
                probabilities[(label_i, label_j)] = (
                    total_neighbors_with_label_j / total_degree_i if total_degree_i else 0
                )

    return probabilities


def compute_homophily(
    graph: Union[nx.Graph, 'TaGraGraph'],
    target_attribute: str,
    n_permutations: int = 100,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Compute homophily score and statistical significance.

    Homophily measures the tendency of nodes to connect to others
    with the same attribute value.

    Parameters
    ----------
    graph : nx.Graph or TaGraGraph
        Graph to analyze
    target_attribute : str
        Node attribute to analyze
    n_permutations : int, default=100
        Number of permutations for significance test
    verbose : bool, default=True
        Print progress messages

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - homophily_score: Observed homophily (0-1)
        - homophily_p_value: Permutation test p-value
        - homophily_z_score: Z-score relative to random

    Examples
    --------
    >>> stats = compute_homophily(G, 'label')
    >>> print(f"Homophily: {stats['homophily_score']:.4f}")
    """
    if hasattr(graph, 'to_networkx'):
        G = graph.to_networkx()
    else:
        G = graph

    if verbose:
        print(f"{datetime.now()}: Computing homophily for '{target_attribute}'...")

    # Get neighborhood analysis
    df_neigh = analyze_neighborhoods(G, target_attribute, verbose=False)
    target_values = df_neigh[f'node_{target_attribute}'].unique()

    if len(target_values) <= 1:
        return {
            'homophily_score': 1.0,
            'homophily_p_value': 1.0,
            'homophily_z_score': 0.0
        }

    # Compute observed homophily
    probs = compute_neighborhood_probabilities(df_neigh, target_attribute)
    diagonal_sum = sum(probs.get((i, i), 0) for i in target_values)
    homophily_score = diagonal_sum / len(target_values)

    if verbose:
        print(f"{datetime.now()}: Observed homophily score: {homophily_score:.4f}")

    # Permutation test
    if verbose:
        print(f"{datetime.now()}: Running permutation test ({n_permutations} permutations)...")

    permutation_scores = []
    attr_values = [G.nodes[n].get(target_attribute, 'None') for n in G.nodes()]

    for i in range(n_permutations):
        if verbose and i % 25 == 0:
            print(f"{datetime.now()}: Permutation {i}/{n_permutations}...")

        # Shuffle attributes
        G_perm = G.copy()
        shuffled = attr_values.copy()
        random.shuffle(shuffled)

        for idx, node in enumerate(G.nodes()):
            G_perm.nodes[node][target_attribute] = shuffled[idx]

        # Compute permuted homophily
        df_perm = analyze_neighborhoods(G_perm, target_attribute, verbose=False)
        perm_probs = compute_neighborhood_probabilities(df_perm, target_attribute)
        perm_diagonal = sum(perm_probs.get((i, i), 0) for i in target_values)
        permutation_scores.append(perm_diagonal / len(target_values))

    # Compute p-value and z-score
    p_value = sum(d >= homophily_score for d in permutation_scores) / n_permutations

    if len(permutation_scores) > 1:
        perm_mean = np.mean(permutation_scores)
        perm_std = np.std(permutation_scores)
        z_score = (homophily_score - perm_mean) / perm_std if perm_std > 0 else 0
    else:
        z_score = 0

    return {
        'homophily_score': homophily_score,
        'homophily_p_value': p_value,
        'homophily_z_score': z_score
    }


def compute_chi_square(
    df_neigh: pd.DataFrame,
    target_attribute: str,
    verbose: bool = True
) -> Dict[str, Optional[float]]:
    """
    Compute chi-square test for neighborhood patterns.

    Tests whether the distribution of neighbor attributes differs
    significantly from expected by chance.

    Parameters
    ----------
    df_neigh : pd.DataFrame
        Neighborhood analysis dataframe
    target_attribute : str
        Target attribute name
    verbose : bool, default=True
        Print progress messages

    Returns
    -------
    Dict[str, Optional[float]]
        Dictionary with 'chi2_stat' and 'chi2_p_value'
    """
    label_col = f'node_{target_attribute}'
    target_values = df_neigh[label_col].unique()

    if verbose:
        print(f"{datetime.now()}: Computing chi-square test...")

    # Build contingency table
    contingency_table = []
    for target_value in target_values:
        row = []
        nodes_with_value = df_neigh[df_neigh[label_col] == target_value]
        for neigh_value in target_values:
            col_name = f'n_{neigh_value}'
            if col_name in df_neigh.columns:
                total_neighbors = nodes_with_value[col_name].sum()
                row.append(total_neighbors)
        contingency_table.append(row)

    # Perform chi-square test
    if len(contingency_table) > 1 and all(sum(row) > 0 for row in contingency_table):
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        return {'chi2_stat': chi2, 'chi2_p_value': p_value}
    else:
        if verbose:
            print(f"{datetime.now()}: Insufficient data for chi-square test")
        return {'chi2_stat': None, 'chi2_p_value': None}
