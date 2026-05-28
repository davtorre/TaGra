"""
Unit test: GowerDistanceConstructor with NaN values.

Verifies that:
1. The distance between two fully-observed rows equals the standard Gower result.
2. When one feature is NaN in one row, that feature is excluded from the
   distance for all pairs involving that row.
3. Two rows with no features in common receive distance 1.0.
4. No NaN appears in the output distance matrix.
5. Backward compatibility: data without NaN gives the same result as before.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
import networkx as nx
from tagra.construction.gower import GowerDistanceConstructor


def _build_and_get_matrix(values, feature_types, threshold=1.0, metric="range"):
    """Run the constructor and return the internal distance matrix."""
    constructor = GowerDistanceConstructor(
        distance_threshold=threshold,
        feature_types=feature_types,
        continuous_metric=metric,
        verbose=False,
    )
    return constructor._gower_matrix(
        values.astype(float), feature_types
    )


def test_no_nan_backward_compat():
    """Without NaN, result must match the original sum/p formula."""
    vals = np.array([[0.0, 1.0],
                     [1.0, 0.0]])
    ftypes = ["continuous", "binary"]
    D = _build_and_get_matrix(vals, ftypes)
    # continuous feature: range=1, d=|0-1|/1=1.0
    # binary feature: different → d=1.0
    # mean = (1.0 + 1.0) / 2 = 1.0
    assert np.isclose(D[0, 1], 1.0), f"Expected 1.0, got {D[0, 1]}"
    assert np.isclose(D[0, 0], 0.0), f"Diagonal should be 0, got {D[0, 0]}"
    print("PASS: no-NaN backward compatibility")


def test_nan_feature_excluded():
    """
    Row 0: [0.0, 1.0]
    Row 1: [NaN, 0.0]  ← feature 0 is NaN
    Only feature 1 (binary) is valid for pair (0,1): d = 1 if different.
    Expected d(0,1) = 1.0  (1 differs → 1.0, only 1 feature in common)
    """
    vals = np.array([[0.0, 1.0],
                     [np.nan, 0.0]])
    ftypes = ["continuous", "binary"]
    D = _build_and_get_matrix(vals, ftypes)
    assert np.isclose(D[0, 1], 1.0), f"Expected 1.0, got {D[0, 1]}"
    print("PASS: NaN feature excluded from distance")


def test_nan_feature_same_value():
    """
    Row 0: [0.0, 1.0]
    Row 1: [NaN, 1.0]  ← feature 0 NaN, feature 1 same
    Only feature 1 valid → d = 0.0
    """
    vals = np.array([[0.0, 1.0],
                     [np.nan, 1.0]])
    ftypes = ["continuous", "binary"]
    D = _build_and_get_matrix(vals, ftypes)
    assert np.isclose(D[0, 1], 0.0), f"Expected 0.0, got {D[0, 1]}"
    print("PASS: NaN feature excluded, remaining features equal → d=0")


def test_no_common_features():
    """
    Row 0: [0.0, NaN]
    Row 1: [NaN, 1.0]
    No feature is valid for both → distance should be 1.0 (maximum)
    """
    vals = np.array([[0.0, np.nan],
                     [np.nan, 1.0]])
    ftypes = ["continuous", "binary"]
    D = _build_and_get_matrix(vals, ftypes)
    assert np.isclose(D[0, 1], 1.0), f"Expected 1.0 (no common features), got {D[0, 1]}"
    print("PASS: no common features → d=1.0")


def test_no_nan_in_output():
    """Output matrix must never contain NaN, even with heavy missingness."""
    rng = np.random.default_rng(42)
    n, p = 20, 5
    vals = rng.standard_normal((n, p))
    # Inject 40% NaN randomly
    mask = rng.random((n, p)) < 0.4
    vals[mask] = np.nan
    ftypes = ["continuous"] * p
    D = _build_and_get_matrix(vals, ftypes)
    assert not np.isnan(D).any(), "NaN found in output distance matrix"
    assert (D >= 0).all() and (D <= 1).all(), "Distance out of [0,1]"
    print("PASS: no NaN in output with heavy missingness")


def test_graph_construction_with_nan():
    """Full GowerDistanceConstructor.construct() call with NaN input."""
    vals = np.array([
        [0.0, 1.0, np.nan],
        [0.1, 1.0, 2.0],
        [0.9, 0.0, 1.0],
    ], dtype=float)
    ftypes = ["continuous", "binary", "ordinal"]
    G = nx.Graph()
    G.add_nodes_from(range(3))
    constructor = GowerDistanceConstructor(
        distance_threshold=0.5,
        feature_types=ftypes,
        verbose=False,
    )
    constructor.construct(G, vals)
    assert G.number_of_nodes() == 3
    print(f"PASS: construct() with NaN — {G.number_of_edges()} edge(s) added")


if __name__ == "__main__":
    test_no_nan_backward_compat()
    test_nan_feature_excluded()
    test_nan_feature_same_value()
    test_no_common_features()
    test_no_nan_in_output()
    test_graph_construction_with_nan()
    print("\nAll tests passed.")
