#!/usr/bin/env python
"""
Clustering comparison: DBSCAN vs TaGra graph-based clustering.

Compares standard DBSCAN with graph-based clustering using TaGra's
various graph construction methods + node degree filtering.

The key idea: DBSCAN is equivalent to building a distance-threshold graph,
filtering out nodes with degree < min_samples - 1 (non-core points),
and extracting connected components. TaGra generalizes this by allowing
alternative graph construction methods (KNN, cosine similarity).

Usage:
    python clustering_comparison.py -d path/to/dataframe.csv
    python clustering_comparison.py -d path/to/dataframe.csv -c clustering_config.json
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import v_measure_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from tagra.construction.knn import KNNConstructor
from tagra.construction.distance import DistanceThresholdConstructor
from tagra.construction.similarity import SimilarityThresholdConstructor
from tagra.construction.dbscan import DBSCANGraphConstructor
from tagra.construction.gower import GowerDistanceConstructor

# DBCV: optional dependency
try:
    from hdbscan.validity import validity_index as dbcv_score
    DBCV_AVAILABLE = True
except ImportError:
    try:
        from hdbscan import validity_index as dbcv_score
        DBCV_AVAILABLE = True
    except ImportError:
        DBCV_AVAILABLE = False


# ---------------------------------------------------------------------------
# Preprocessing 
# ---------------------------------------------------------------------------

# TODO: Replace with proper TaGra preprocessing pipeline.
# Currently selects numeric columns and applies standard scaling.

def preprocess(df):
    """
    Preprocessing placeholder.

    Returns scaled data (StandardScaler) suitable for Euclidean-based methods.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric columns found in the dataframe.")

    # ---- PREPROCESSING GOES HERE ----
    # Replace this block with:
    #   from tagra.preprocessing import preprocess as tagra_preprocess
    #   df_preprocessed, positions = tagra_preprocess(...)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)
    # ---------------------------------

    return pd.DataFrame(scaled, columns=numeric_df.columns)


def extract_raw_numeric(df):
    """Return raw (unscaled) numeric columns as a float array and column list."""
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.values.astype(float), list(numeric_df.columns)


def detect_feature_types(X_raw, columns):
    """
    Auto-detect feature types from raw data.

    Heuristic:
    - Binary  : exactly 2 unique non-NaN values
    - Ordinal : 3–6 unique non-NaN values
    - Continuous : 7+ unique non-NaN values
    """
    types = []
    for k in range(X_raw.shape[1]):
        col = X_raw[:, k]
        n_unique = len(np.unique(col[~np.isnan(col)]))
        if n_unique <= 2:
            types.append('binary')
        elif n_unique <= 6:
            types.append('ordinal')
        else:
            types.append('continuous')
    return types

# ---------------------------------------------------------------------------
# DBSCAN baseline
# ---------------------------------------------------------------------------

def run_dbscan(X, eps, min_samples):
    """Run DBSCAN and return cluster labels (-1 = noise)."""
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    return labels

# ---------------------------------------------------------------------------
# DBSCAN constructor (TaGra-native, should be equivalent to sklearn DBSCAN)
# ---------------------------------------------------------------------------

def run_dbscan_constructor(X, eps, min_samples):
    """
    Run TaGra's DBSCANGraphConstructor and extract cluster labels.

    Builds a directed graph where core points emit edges to all neighbors
    within eps. Clusters = weakly connected components; isolated nodes = noise.
    Should produce labels identical to sklearn DBSCAN (V-measure = 1).
    """
    n_samples = X.shape[0]
    G = nx.DiGraph()
    for i in range(n_samples):
        G.add_node(i)

    constructor = DBSCANGraphConstructor(eps=eps, min_samples=min_samples, verbose=False)
    constructor.construct(G, X)

    labels = np.full(n_samples, -1)
    cluster_id = 0
    for component in nx.weakly_connected_components(G):
        if len(component) == 1:
            node = next(iter(component))
            if G.in_degree(node) == 0 and G.out_degree(node) == 0:
                continue  # isolated node -> noise (-1)
        for node in component:
            labels[node] = cluster_id
        cluster_id += 1

    n_clusters = cluster_id
    n_noise = int(np.sum(labels == -1))
    return labels, n_clusters, n_noise


# ---------------------------------------------------------------------------
# Graph-based clustering
# ---------------------------------------------------------------------------

def build_graph_and_cluster(X, method, params, degree_filter):
    """
    Build a TaGra graph, apply node degree filter, extract connected components.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    method : str
        'distance', 'knn', or 'similarity'
    params : dict
        Constructor-specific parameters
    degree_filter : int
        Minimum node degree to keep (nodes below this become noise).
        Should be min_samples - 1 to match DBSCAN's core point criterion.

    Returns
    -------
    labels : np.ndarray
        Cluster labels (-1 = noise)
    info : dict
        Graph statistics (n_edges, n_clusters, n_noise, n_core_nodes)
    """
    n_samples = X.shape[0]

    # Create graph with nodes
    G = nx.Graph()
    for i in range(n_samples):
        G.add_node(i)

    values = X if isinstance(X, np.ndarray) else X.values

    # Build edges using TaGra constructor
    if method == "distance":
        constructor = DistanceThresholdConstructor(
            distance_threshold=params["distance_threshold"], verbose=False
        )
    elif method == "knn":
        constructor = KNNConstructor(k=params["k"], verbose=False)
    elif method == "similarity":
        constructor = SimilarityThresholdConstructor(
            similarity_threshold=params["similarity_threshold"], verbose=False
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    constructor.construct(G, values)
    n_edges = G.number_of_edges()

    # Degree filter: compute degrees ONCE on the original graph, then filter.
    # This mirrors DBSCAN's core point criterion: a point is core if it has
    # >= min_samples neighbors (including itself). In the graph, degree counts
    # neighbors only (no self-loops), so the threshold is min_samples - 1.
    degrees = dict(G.degree())
    low_degree_nodes = [n for n, d in degrees.items() if d < degree_filter]

    G_filtered = G.copy()
    G_filtered.remove_nodes_from(low_degree_nodes)

    # Connected components = clusters
    labels = np.full(n_samples, -1)
    cluster_id = 0
    for component in nx.connected_components(G_filtered):
        for node in component:
            labels[node] = cluster_id
        cluster_id += 1

    info = {
        "n_edges": n_edges,
        "n_clusters": cluster_id,
        "n_noise": int(np.sum(labels == -1)),
        "n_core_nodes": n_samples - len(low_degree_nodes),
    }
    return labels, info


# ---------------------------------------------------------------------------
# Gower graph clustering
# ---------------------------------------------------------------------------

def build_gower_and_cluster(X_raw, feature_types, params, degree_filter):
    """
    Build a Gower-distance graph and extract clusters via degree filtering.

    Gower distance is computed on raw (unscaled) data with auto-detected or
    explicitly provided feature types, so binary/ordinal semantics are preserved.

    Parameters
    ----------
    X_raw : np.ndarray
        Raw (unscaled) feature matrix.
    feature_types : list of str
        Per-column type labels ('continuous', 'binary', 'ordinal').
        Overridden by 'feature_types' key in params if present.
    params : dict
        Must contain 'distance_threshold'; optionally 'continuous_metric'
        and 'feature_types'.
    degree_filter : int

    Returns
    -------
    labels : np.ndarray
    info : dict
    """
    n_samples = X_raw.shape[0]

    G = nx.Graph()
    for i in range(n_samples):
        G.add_node(i)

    constructor = GowerDistanceConstructor(
        distance_threshold=params["distance_threshold"],
        feature_types=params.get("feature_types", feature_types),
        continuous_metric=params.get("continuous_metric", "range"),
        verbose=False,
    )
    constructor.construct(G, X_raw)
    n_edges = G.number_of_edges()

    degrees = dict(G.degree())
    low_degree_nodes = [n for n, d in degrees.items() if d < degree_filter]
    G_filtered = G.copy()
    G_filtered.remove_nodes_from(low_degree_nodes)

    labels = np.full(n_samples, -1)
    cluster_id = 0
    for component in nx.connected_components(G_filtered):
        for node in component:
            labels[node] = cluster_id
        cluster_id += 1

    info = {
        "n_edges": n_edges,
        "n_clusters": cluster_id,
        "n_noise": int(np.sum(labels == -1)),
        "n_core_nodes": n_samples - len(low_degree_nodes),
    }
    return labels, info


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_dbcv(X, labels):
    """Compute DBCV score. Returns None if unavailable or not computable."""
    if not DBCV_AVAILABLE:
        return None
    unique = set(labels)
    unique.discard(-1)
    if len(unique) < 2:
        return None
    mask = labels != -1
    if mask.sum() < 2:
        return None
    try:
        return float(dbcv_score(X[mask], labels[mask]))
    except Exception:
        return None


def compute_silhouette(X, labels):
    """Compute silhouette score on non-noise points."""
    unique = set(labels)
    unique.discard(-1)
    if len(unique) < 2:
        return None
    mask = labels != -1
    if mask.sum() < 2:
        return None
    try:
        return float(silhouette_score(X[mask], labels[mask]))
    except Exception:
        return None


def compute_v_measure(labels_ref, labels_test):
    """Compute V-measure between two clusterings."""
    return float(v_measure_score(labels_ref, labels_test))


# ---------------------------------------------------------------------------
# Report and plot
# ---------------------------------------------------------------------------

def generate_report(results, dbscan_config, degree_filter, dataset_info, output_path):
    """Write a plain text report."""
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CLUSTERING COMPARISON: DBSCAN vs TaGra Graph-Based Clustering\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("Dataset\n")
        f.write("-" * 40 + "\n")
        f.write(f"  File:     {dataset_info['file']}\n")
        f.write(f"  Samples:  {dataset_info['n_samples']}\n")
        f.write(f"  Features: {dataset_info['n_features']}\n\n")

        f.write("Configuration\n")
        f.write("-" * 40 + "\n")
        f.write(f"  DBSCAN eps:         {dbscan_config['eps']}\n")
        f.write(f"  DBSCAN min_samples: {dbscan_config['min_samples']}\n")
        f.write(f"  Default degree filter: {degree_filter} (= min_samples - 1, overridable per method)\n\n")

        f.write("Results\n")
        f.write("-" * 80 + "\n")
        header = (
            f"{'Method':<42} {'Clusters':>8} {'Noise%':>7} "
            f"{'DBCV':>8} {'Silhouet':>8} {'V-meas':>8}"
        )
        f.write(header + "\n")
        f.write("-" * 80 + "\n")

        for r in results:
            dbcv_s = f"{r['dbcv']:.4f}" if r["dbcv"] is not None else "N/A"
            sil_s = f"{r['silhouette']:.4f}" if r["silhouette"] is not None else "N/A"
            vm_s = f"{r['v_measure']:.4f}" if r["v_measure"] is not None else "-"
            noise_pct = f"{100 * r['n_noise'] / dataset_info['n_samples']:.1f}%"
            line = (
                f"{r['name']:<42} {r['n_clusters']:>8} {noise_pct:>7} "
                f"{dbcv_s:>8} {sil_s:>8} {vm_s:>8}"
            )
            f.write(line + "\n")

        f.write("\n")
        f.write("Legend\n")
        f.write("-" * 40 + "\n")
        f.write("  DBCV:      Density-Based Clustering Validation [-1, +1]\n")
        f.write("  Silhouet:  Silhouette Coefficient [-1, +1]\n")
        f.write("  V-meas:    V-measure vs DBSCAN [0, 1] (1 = identical clustering)\n")
        f.write("  Noise:     Points not assigned to any cluster\n")
        f.write("  df:        Degree filter applied to graph nodes\n")
        if not DBCV_AVAILABLE:
            f.write("\n  NOTE: DBCV unavailable. Install hdbscan: pip install hdbscan\n")


def generate_plot(results, output_path):
    """Generate a horizontal bar chart comparing clustering metrics."""
    names = [r["name"] for r in results]

    # Collect available metrics
    metrics = []
    if any(r["dbcv"] is not None for r in results):
        metrics.append(("dbcv", "DBCV [-1, +1]"))
    if any(r["silhouette"] is not None for r in results):
        metrics.append(("silhouette", "Silhouette [-1, +1]"))

    if not metrics:
        print("  No numeric metrics available for plotting.")
        return

    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(6 * n_metrics, max(4, len(names) * 0.45)),
    )
    if n_metrics == 1:
        axes = [axes]

    # Color by method type
    def pick_color(name):
        if "DBSCAN" in name:
            return "#2196F3"
        if "distance" in name:
            return "#4CAF50"
        if "knn" in name:
            return "#FF9800"
        if "similarity" in name:
            return "#9C27B0"
        return "#757575"

    colors = [pick_color(n) for n in names]

    for ax, (key, label) in zip(axes, metrics):
        vals = [r[key] if r[key] is not None else 0.0 for r in results]
        y_pos = range(len(names))
        ax.barh(y_pos, vals, color=colors)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel(label)
        ax.set_title(label)
        ax.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {output_path}")


def generate_vmeasure_heatmap(results, output_path):
    """Generate a pairwise V-measure heatmap across all clustering methods."""
    names = [r["name"] for r in results]
    n = len(names)

    # Build N x N matrix
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = v_measure_score(results[i]["labels"], results[j]["labels"])

    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), max(5, n * 0.75)))

    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="V-measure [0, 1]")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)

    # Annotate each cell
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            text_color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=text_color)

    ax.set_title("Pairwise V-measure between all clustering methods")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clustering comparison: DBSCAN vs TaGra graph-based clustering"
    )
    parser.add_argument(
        "-d", "--dataframe", required=True, help="Path to input CSV dataframe"
    )
    parser.add_argument(
        "-c", "--config", default="clustering_config.json", help="Path to config JSON"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    dbscan_cfg = config["dbscan"]
    graph_methods = config["graph_methods"]
    output_dir = config.get("output_dir", "clustering_results")

    eps = dbscan_cfg["eps"]
    min_samples = dbscan_cfg["min_samples"]
    degree_filter = min_samples - 1

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"Loading: {args.dataframe}")
    df = pd.read_csv(args.dataframe)
    print(f"  Shape: {df.shape}")

    # Preprocess
    print("Preprocessing...")
    df_prep = preprocess(df)
    X = df_prep.values
    print(f"  Numeric columns: {list(df_prep.columns)}")

    # Raw data and feature types for Gower (must not be StandardScaled)
    X_raw, raw_columns = extract_raw_numeric(df)
    feature_types = detect_feature_types(X_raw, raw_columns)
    type_counts = {t: feature_types.count(t) for t in set(feature_types)}
    print(f"  Feature types (Gower): {type_counts}")

    results = []

    # ---- DBSCAN baseline ----
    print(f"\nDBSCAN (eps={eps}, min_samples={min_samples})")
    dbscan_labels = run_dbscan(X, eps, min_samples)
    n_cl = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_ns = int(np.sum(dbscan_labels == -1))
    print(f"  Clusters: {n_cl}, Noise: {n_ns}")

    results.append({
        "name": f"DBSCAN (eps={eps}, ms={min_samples})",
        "n_clusters": n_cl,
        "n_noise": n_ns,
        "dbcv": compute_dbcv(X, dbscan_labels),
        "silhouette": compute_silhouette(X, dbscan_labels),
        "v_measure": None,
        "labels": dbscan_labels,
    })

    # ---- TaGra DBSCANGraphConstructor (should match sklearn DBSCAN exactly) ----
    print(f"\nDBSCANGraphConstructor (eps={eps}, min_samples={min_samples})")
    dgc_labels, dgc_n_cl, dgc_n_ns = run_dbscan_constructor(X, eps, min_samples)
    print(f"  Clusters: {dgc_n_cl}, Noise: {dgc_n_ns}")

    results.append({
        "name": f"DBSCANConstructor (eps={eps}, ms={min_samples})",
        "n_clusters": dgc_n_cl,
        "n_noise": dgc_n_ns,
        "dbcv": compute_dbcv(X, dgc_labels),
        "silhouette": compute_silhouette(X, dgc_labels),
        "v_measure": compute_v_measure(dbscan_labels, dgc_labels),
        "labels": dgc_labels,
    })

    # ---- TaGra graph-based clustering ----
    # Each entry in graph_methods[method] is a dict with method params and an
    # optional "degree_filter" key. Falls back to the global degree_filter
    # (min_samples - 1) when not specified.
    print(f"\nDefault degree filter: {degree_filter} (= min_samples - 1)")

    # Normalise config: support both old scalar/list format and new list-of-dicts
    def normalise_entries(method, raw):
        """Return list of (params_dict, df_value) from a raw config value."""
        entries = []
        if isinstance(raw, list) and all(isinstance(e, dict) for e in raw):
            # New format: list of dicts with optional degree_filter per entry
            for e in raw:
                e = dict(e)
                df_val = e.pop("degree_filter", degree_filter)
                entries.append((e, df_val))
        elif isinstance(raw, dict):
            # Old format: {"distance_threshold": [...]} or {"k": [...]} etc.
            if method == "distance":
                for dt in raw.get("distance_threshold", []):
                    entries.append(({"distance_threshold": dt}, degree_filter))
            elif method == "knn":
                for k in raw.get("k", []):
                    entries.append(({"k": k}, degree_filter))
            elif method == "similarity":
                for st in raw.get("similarity_threshold", []):
                    entries.append(({"similarity_threshold": st}, degree_filter))
        return entries

    method_configs = []
    for method in ("distance", "knn", "similarity", "gower"):
        if method in graph_methods:
            for params, df_val in normalise_entries(method, graph_methods[method]):
                method_configs.append((method, params, df_val))

    for method, params, df_val in method_configs:
        if method == "distance":
            p_str = f"t={params['distance_threshold']}"
        elif method == "knn":
            p_str = f"k={params['k']}"
        elif method == "similarity":
            p_str = f"t={params['similarity_threshold']}"
        else:  # gower
            metric = params.get('continuous_metric', 'range')
            p_str = f"t={params['distance_threshold']},{metric}"
        name = f"{method} ({p_str}, df={df_val})"

        print(f"\n  {name}")
        if method == "gower":
            labels, info = build_gower_and_cluster(X_raw, feature_types, params, df_val)
        else:
            labels, info = build_graph_and_cluster(X, method, params, df_val)
        print(f"    Edges: {info['n_edges']}, Core nodes: {info['n_core_nodes']}, "
              f"Clusters: {info['n_clusters']}, Noise: {info['n_noise']}")

        results.append({
            "name": name,
            "n_clusters": info["n_clusters"],
            "n_noise": info["n_noise"],
            "dbcv": compute_dbcv(X, labels),
            "silhouette": compute_silhouette(X, labels),
            "v_measure": compute_v_measure(dbscan_labels, labels),
            "labels": labels,
        })

    # ---- Output ----
    print("\n" + "=" * 50)
    dataset_info = {
        "file": args.dataframe,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
    }

    report_path = os.path.join(output_dir, "clustering_report.txt")
    plot_path = os.path.join(output_dir, "clustering_comparison.png")
    heatmap_path = os.path.join(output_dir, "vmeasure_heatmap.png")

    generate_report(results, dbscan_cfg, degree_filter, dataset_info, report_path)
    print(f"  Report saved: {report_path}")

    generate_plot(results, plot_path)
    generate_vmeasure_heatmap(results, heatmap_path)

    print("Done.")


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
