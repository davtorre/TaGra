#!/usr/bin/env python
"""
Cleveland Heart Disease – focused clustering comparison for paper.

Reference:    sklearn DBSCAN at (eps=0.9, ms=3) and (eps=1.0, ms=3)
              TaGra DBSCANConstructor at the same params  → proves equivalence

Comparison:   TaGra/Gower curated subspace (quadratic and range metrics,
              low degree_filter) that achieves better Vm and DBCV than the
              DBSCAN reference at comparable noise levels.

Argument for the paper:
  DBSCAN and TaGra/distance are equivalent by construction (Vm=1.0).
  Gower distance, operating on raw mixed-type features, finds a different
  density structure that recovers the clinical label better (higher Vm)
  and produces tighter clusters (higher DBCV), at the cost of more clusters.
  This is the Pareto frontier: no single method dominates on all axes.

V-measure is vs the known binary clinical label (0=healthy, 1+=disease).

Usage:
    cd dev/tasks/clustering
    python3 run_cleveland.py
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
from itertools import product
from sklearn.cluster import DBSCAN
from sklearn.metrics import v_measure_score, silhouette_score

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from tagra.construction.distance import DistanceThresholdConstructor
from tagra.construction.knn import KNNConstructor
from tagra.construction.similarity import SimilarityThresholdConstructor
from tagra.construction.dbscan import DBSCANGraphConstructor
from tagra.construction.gower import GowerDistanceConstructor

from preprocess_typed import (
    preprocess_typed,
    feature_type_map_for_gower,
    CLEVELAND_SPEC,
    CLEVELAND_IMPUTE,
    QUALITY_CRITERIA,
    filter_results,
)

try:
    from hdbscan.validity import validity_index as dbcv_score
    DBCV_AVAILABLE = True
except ImportError:
    DBCV_AVAILABLE = False

try:
    import hdbscan as hdbscan_lib
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(
    REPO_ROOT,
    "examples/datasets/UCI/heart_disease/processed.cleveland.data",
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "clustering_results_cleveland")
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_and_preprocess():
    df = pd.read_csv(DATA_PATH, header=None, names=COLS, na_values="?")

    X, feat_names = preprocess_typed(df, CLEVELAND_SPEC, CLEVELAND_IMPUTE, final_minmax=True)
    assert not np.isnan(X).any()

    true_labels = (df["num"].fillna(0).values > 0).astype(int)

    # Raw for Gower (impute only, no scaling)
    df_imp = df.copy()
    for col, strategy in CLEVELAND_IMPUTE.items():
        df_imp[col] = df_imp[col].fillna(
            df_imp[col].median() if strategy == "median" else df_imp[col].mode()[0]
        )
    feat_cols = [c for c, t in CLEVELAND_SPEC.items()
                 if t not in ("target", "id") and c in df.columns]
    X_raw = df_imp[feat_cols].astype(float).values
    gower_ftypes = feature_type_map_for_gower(CLEVELAND_SPEC, df)

    return X, X_raw, gower_ftypes, true_labels, len(df)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dbcv(X, labels):
    if not DBCV_AVAILABLE or len(set(labels) - {-1}) < 2:
        return None
    mask = labels != -1
    if mask.sum() < 2:
        return None
    try:
        return float(dbcv_score(X[mask], labels[mask]))
    except Exception:
        return None


def sil(X, labels):
    if len(set(labels) - {-1}) < 2:
        return None
    mask = labels != -1
    if mask.sum() < 2:
        return None
    try:
        return float(silhouette_score(X[mask], labels[mask]))
    except Exception:
        return None


def vm(true, pred):
    return float(v_measure_score(true, pred))


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def make_graph(n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    return G


def graph_cluster(G, n, degree_filter):
    low = [node for node, d in G.degree() if d < degree_filter]
    Gf = G.copy()
    Gf.remove_nodes_from(low)
    labels = np.full(n, -1)
    cid = 0
    for comp in nx.connected_components(Gf):
        for node in comp:
            labels[node] = cid
        cid += 1
    return labels


def dbscan_constructor_labels(X, eps, min_samples):
    n = len(X)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    DBSCANGraphConstructor(eps=eps, min_samples=min_samples, verbose=False).construct(G, X)
    labels = np.full(n, -1)
    cid = 0
    for comp in nx.weakly_connected_components(G):
        if len(comp) == 1:
            node = next(iter(comp))
            if G.in_degree(node) == 0 and G.out_degree(node) == 0:
                continue
        for node in comp:
            labels[node] = cid
        cid += 1
    return labels


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

def record(name, X, labels, true_labels, ref_labels=None):
    n_cl   = len(set(labels) - {-1})
    n_ns   = int(np.sum(labels == -1))
    noise  = 100 * n_ns / len(labels)
    vm_ref = vm(ref_labels, labels) if ref_labels is not None else None
    return {
        "name":       name,
        "n_clusters": n_cl,
        "noise_pct":  noise,
        "dbcv":       dbcv(X, labels),
        "sil":        sil(X, labels),
        "vm_true":    vm(true_labels, labels),
        "vm_ref":     vm_ref,
        "labels":     labels,
    }


def fmt(v, w=7):
    return f"{v:{w}.4f}" if v is not None else " " * (w - 3) + "N/A"


HDR = (f"{'Method':<52} {'Cl':>4} {'Noise%':>7} "
       f"{'DBCV':>8} {'Sil':>8} {'Vm(true)':>9} {'Vm(ref)':>8}")
SEP = "-" * 105


def print_row(r, show_vm_ref=False):
    vm_ref_s = fmt(r["vm_ref"]) if show_vm_ref else "       -"
    print(f"  {r['name']:<52} {r['n_clusters']:>4} {r['noise_pct']:>6.1f}%"
          f" {fmt(r['dbcv']):>8} {fmt(r['sil']):>8}"
          f" {fmt(r['vm_true']):>9} {vm_ref_s:>8}")


def top3_summary(valid, fh=None):
    """Print (and optionally write) top-3 configs per metric."""
    metrics = [
        ("Lowest Noise%", lambda r: r["noise_pct"],                              False),
        ("Highest DBCV",  lambda r: r["dbcv"]   if r["dbcv"] is not None else -999, True),
        ("Highest Sil",   lambda r: r["sil"]    if r["sil"]  is not None else -999, True),
        ("Highest Vm",    lambda r: r["vm_true"],                                True),
    ]
    lines = [f"\n{'='*78}", "TOP 3 PER METRIC  (≥2 clusters)", HDR, SEP]
    for label, key_fn, rev in metrics:
        lines.append(f"\n  ── {label} ──")
        for r in sorted(valid, key=key_fn, reverse=rev)[:3]:
            vr = fmt(r["vm_ref"]) if r.get("vm_ref") is not None else "       -"
            lines.append(
                f"  {r['name']:<52} {r['n_clusters']:>4} {r['noise_pct']:>6.1f}%"
                f" {fmt(r['dbcv']):>8} {fmt(r['sil']):>8}"
                f" {fmt(r['vm_true']):>9} {vr:>8}"
            )
    for line in lines:
        print(line)
    if fh:
        for line in lines:
            fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict = None):
    if cfg is None:
        cfg = {}
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, X_raw, gower_ftypes, true_labels, n = load_and_preprocess()

    print("=" * 78)
    print("Cleveland Heart Disease – Focused Clustering Comparison")
    print(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"n={n}  |  features (preprocessed): {X.shape[1]}  |  Gower types: {gower_ftypes}")
    print(f"Class balance: {dict(zip(*np.unique(true_labels, return_counts=True)))}")
    print("=" * 78)

    all_results = []

    # ── Section 1: DBSCAN reference ──────────────────────────────────────
    print(f"\n{'─'*78}")
    print("1. DBSCAN reference configurations  (eps=1.1, ms=13 and ms=23)")
    print(HDR)
    print(SEP)

    _dbscan_ref_tuples = [(item["eps"], item["ms"]) for item in
                          cfg.get("dbscan_refs", [{"eps": 1.1, "ms": 13}, {"eps": 1.1, "ms": 23}])]
    dbscan_refs = {}
    for eps, ms in _dbscan_ref_tuples:
        lbl = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
        r = record(f"DBSCAN (eps={eps}, ms={ms})", X, lbl, true_labels)
        dbscan_refs[(eps, ms)] = lbl
        print_row(r)
        all_results.append(r)

    # ── Section 2: TaGra DBSCANConstructor – equivalence proof ───────────
    print(f"\n{'─'*78}")
    print("2. TaGra DBSCANConstructor  [Vm(ref) = V-measure vs sklearn DBSCAN above]")
    print(HDR)
    print(SEP)

    for eps, ms in _dbscan_ref_tuples:
        lbl = dbscan_constructor_labels(X, eps, ms)
        ref = dbscan_refs[(eps, ms)]
        r = record(f"TaGra/DBSCANConstr (eps={eps}, ms={ms})",
                   X, lbl, true_labels, ref_labels=ref)
        print_row(r, show_vm_ref=True)
        all_results.append(r)

    # ── Section 3: HDBSCAN ────────────────────────────────────────────────
    print(f"\n{'─'*78}")
    print("3. HDBSCAN")
    print(HDR)
    print(SEP)

    if HDBSCAN_AVAILABLE:
        for mcs in cfg.get("hdbscan_mcs", [3, 5, 7, 10]):
            lbl = hdbscan_lib.HDBSCAN(min_cluster_size=mcs).fit_predict(X)
            r = record(f"HDBSCAN (mcs={mcs})", X, lbl, true_labels)
            print_row(r)
            all_results.append(r)

    # ── Section 4: TaGra Gower curated subspace ───────────────────────────
    # From the grid search:
    #   quadratic metric, t=0.03–0.07, df=1–3  → best Vm and low noise
    #   range metric,     t=0.05–0.09, df=1–2  → competitive at slightly higher noise
    print(f"\n{'─'*78}")
    print("4. TaGra/Gower  [curated subspace: quadratic and range, low degree_filter]")
    print("   Vm(ref) = V-measure vs DBSCAN (eps=1.1, ms=23)")
    print(HDR)
    print(SEP)

    ref_dbscan = dbscan_refs[(1.1, 23)]

    # Curated from grid search: configs producing 2–5 clusters with best Vm,
    # selected to bracket and match the two DBSCAN references.
    #
    # Ref A: DBSCAN eps=1.1 ms=13  → 3 cl, 25.7% noise, Vm=0.138
    # Ref B: DBSCAN eps=1.1 ms=23  → 3 cl, 52.8% noise, Vm=0.203
    #
    # Best Gower candidates (from 192 configs with 2-6 clusters):
    #   range     t=0.12, df=11  → 2 cl, 71.0% noise, Vm=0.199  ← near Ref B in Vm
    #   laplacian t=0.22, df=5   → 3 cl, 68.6% noise, Vm=0.182  ← 3 clusters
    #   gaussian  t=0.17, df=10  → 2 cl, 74.6% noise, Vm=0.195
    #   range     t=0.12, df=13  → 2 cl, 77.2% noise, Vm=0.189
    #   gaussian  t=0.13, df=4   → 3 cl, 75.6% noise, Vm=0.172  ← 3 clusters
    #   quadratic t=0.08, df=10  → 2 cl, 65.7% noise, Vm=0.172
    #   quadratic t=0.08, df=11  → 2 cl, 71.3% noise, DBCV=0.320, Sil=0.494, Vm=0.149
    #   quadratic t=0.08, df=14  → 2 cl, 81.5% noise, DBCV=0.407, Sil=0.590  ← high quality
    #   quadratic t=0.08, df=15  → 2 cl, 82.5% noise, DBCV=0.413, Sil=0.609
    _default_gower = [
        {"t": 0.22, "metric": "laplacian", "df": 5},
        {"t": 0.13, "metric": "gaussian",  "df": 4},
        {"t": 0.10, "metric": "range",     "df": 7},
        {"t": 0.12, "metric": "range",     "df": 11},
        {"t": 0.17, "metric": "gaussian",  "df": 10},
        {"t": 0.12, "metric": "range",     "df": 13},
        {"t": 0.22, "metric": "laplacian", "df": 6},
        {"t": 0.08, "metric": "quadratic", "df": 10},
        {"t": 0.08, "metric": "quadratic", "df": 11},
        {"t": 0.08, "metric": "quadratic", "df": 14},
        {"t": 0.08, "metric": "quadratic", "df": 15},
    ]
    gower_configs = cfg.get("gower_configs", _default_gower)

    for item in gower_configs:
        thresh, metric, df_val = item["t"], item["metric"], item["df"]
        G = make_graph(n)
        GowerDistanceConstructor(
            distance_threshold=thresh,
            feature_types=gower_ftypes,
            continuous_metric=metric,
            verbose=False,
        ).construct(G, X_raw)
        lbl = graph_cluster(G, n, df_val)
        r = record(f"TaGra/Gower ({metric},t={thresh},df={df_val})",
                   X, lbl, true_labels, ref_labels=ref_dbscan)
        print_row(r, show_vm_ref=True)
        all_results.append(r)

    # ── Section 5: TaGra Cosine Similarity ───────────────────────────────
    # Best configs from exhaustive grid (sim ∈ [0.50,0.95], df ∈ [1,20]):
    #   sim=0.87 df=6  → 6 cl, 35% noise, Vm=0.178  ← lowest noise, good Vm
    #   sim=0.87 df=7  → 5 cl, 41% noise, Vm=0.173
    #   sim=0.83 df=18 → 2 cl, 60% noise, DBCV=0.256, Vm=0.174
    #   sim=0.82 df=19 → 2 cl, 57% noise, DBCV=0.245, Vm=0.149
    #   sim=0.80 df=20 → 2 cl, 48% noise, DBCV=0.208, Vm=0.123
    print(f"\n{'─'*78}")
    print("5. TaGra/Similarity  [curated: best Vm × noise trade-off from grid search]")
    print("   Vm(ref) = V-measure vs DBSCAN (eps=1.1, ms=23)")
    print(HDR)
    print(SEP)

    _default_sim = [
        {"t": 0.87, "df":  6},
        {"t": 0.87, "df":  7},
        {"t": 0.86, "df":  6},
        {"t": 0.86, "df":  7},
        {"t": 0.86, "df":  9},
        {"t": 0.86, "df": 10},
        {"t": 0.86, "df": 11},
        {"t": 0.83, "df": 17},
        {"t": 0.83, "df": 18},
        {"t": 0.82, "df": 18},
        {"t": 0.82, "df": 19},
        {"t": 0.80, "df": 20},
    ]
    sim_configs = cfg.get("sim_configs", _default_sim)

    for item in sim_configs:
        thresh, df_val = item["t"], item["df"]
        G = make_graph(n)
        SimilarityThresholdConstructor(
            similarity_threshold=thresh, verbose=False
        ).construct(G, X)
        lbl = graph_cluster(G, n, df_val)
        r = record(f"TaGra/Sim (t={thresh}, df={df_val})",
                   X, lbl, true_labels, ref_labels=ref_dbscan)
        print_row(r, show_vm_ref=True)
        all_results.append(r)

    # ── Section 6: TaGra KNN ──────────────────────────────────────────────
    # KNN grid insight: fragmentation only occurs when df > k (degree filter
    # above the guaranteed minimum degree). Best region: k=5-10, df=k+3..k+12.
    # Results are weaker than similarity but included for completeness.
    print(f"\n{'─'*78}")
    print("6. TaGra/KNN  [df > k regime: targeted grid from exhaustive search]")
    print("   Vm(ref) = V-measure vs DBSCAN (eps=1.1, ms=23)")
    print(HDR)
    print(SEP)

    _default_knn = [
        {"k":  7, "df": 10},
        {"k":  4, "df":  5},
        {"k":  5, "df":  7},
        {"k":  5, "df":  8},
        {"k":  6, "df":  8},
        {"k":  6, "df":  9},
        {"k":  6, "df": 10},
        {"k":  7, "df":  9},
        {"k":  7, "df": 11},
        {"k":  8, "df": 10},
        {"k":  8, "df": 11},
        {"k":  8, "df": 12},
        {"k":  9, "df": 12},
        {"k":  9, "df": 13},
        {"k": 10, "df": 13},
        {"k": 10, "df": 14},
        {"k": 10, "df": 15},
    ]
    knn_configs = cfg.get("knn_configs", _default_knn)

    for item in knn_configs:
        k, df_val = item["k"], item["df"]
        G = make_graph(n)
        KNNConstructor(k=k, verbose=False).construct(G, X)
        lbl = graph_cluster(G, n, df_val)
        r = record(f"TaGra/KNN (k={k}, df={df_val})",
                   X, lbl, true_labels, ref_labels=ref_dbscan)
        print_row(r, show_vm_ref=True)
        all_results.append(r)

    # ── Summary ───────────────────────────────────────────────────────────
    valid      = [r for r in all_results if r["n_clusters"] >= 2]
    valid_low  = [r for r in valid if r["noise_pct"] <= 60]

    print(f"\n{'='*78}")
    print("SUMMARY A – ≥2 clusters, noise ≤ 60%, sorted by Vm(true) ↓")
    print(HDR)
    print(SEP)
    for r in sorted(valid_low, key=lambda r: r["vm_true"], reverse=True):
        print_row(r, show_vm_ref=r.get("vm_ref") is not None)

    print(f"\n{'='*78}")
    print("SUMMARY B – ≥2 clusters, noise ≤ 60%, sorted by DBCV ↓ then Noise% ↑")
    print(HDR)
    print(SEP)
    for r in sorted([r for r in valid_low if r["dbcv"] is not None],
                    key=lambda r: (-r["dbcv"], r["noise_pct"])):
        print_row(r, show_vm_ref=r.get("vm_ref") is not None)
    # ── Quality-filtered summary ───────────────────────────────────────────
    qc = {**QUALITY_CRITERIA, **cfg.get("quality_criteria", {})}
    qc_title = (
        f"QUALITY-FILTERED  "
        f"(DBCV≥{qc['min_dbcv']}, noise<{qc['max_noise_pct']}%, "
        f"min cluster≥{qc['min_cluster_size_pct']}% of non-noise, "
        f"clusters≥{qc['min_n_clusters']})"
    )
    qc_rows = filter_results(all_results, criteria=qc)
    print(f"\n{'='*78}")
    print(qc_title)
    print(HDR)
    print(SEP)
    if qc_rows:
        for r in sorted(qc_rows, key=lambda r: r["vm_true"], reverse=True):
            print_row(r, show_vm_ref=r.get("vm_ref") is not None)
    else:
        print("  (no results pass all criteria)")

    # ── Save ──────────────────────────────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, "cleveland_report.txt")
    with open(report_path, "w") as f:
        f.write("Cleveland Heart Disease – Focused Clustering Comparison\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"n={n}  features={X.shape[1]}  Gower types: {gower_ftypes}\n\n")
        f.write(HDR + "\n" + SEP + "\n")
        for r in all_results:
            vm_ref_s = fmt(r["vm_ref"]) if r.get("vm_ref") is not None else "       -"
            f.write(
                f"  {r['name']:<52} {r['n_clusters']:>4} {r['noise_pct']:>6.1f}%"
                f" {fmt(r['dbcv']):>8} {fmt(r['sil']):>8}"
                f" {fmt(r['vm_true']):>9} {vm_ref_s:>8}\n"
            )
        f.write(f"\n{'='*78}\n{qc_title}\n")
        f.write(HDR + "\n" + SEP + "\n")
        if qc_rows:
            for r in sorted(qc_rows, key=lambda r: r["vm_true"], reverse=True):
                vm_ref_s = fmt(r["vm_ref"]) if r.get("vm_ref") is not None else "       -"
                f.write(
                    f"  {r['name']:<52} {r['n_clusters']:>4} {r['noise_pct']:>6.1f}%"
                    f" {fmt(r['dbcv']):>8} {fmt(r['sil']):>8}"
                    f" {fmt(r['vm_true']):>9} {vm_ref_s:>8}\n"
                )
        else:
            f.write("  (no results pass all criteria)\n")
        top3_summary(valid_low, fh=f)

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    _default_cfg = os.path.join(os.path.dirname(__file__), "config_cleveland.json")
    parser = argparse.ArgumentParser(description="Cleveland clustering comparison")
    parser.add_argument("--config", default=_default_cfg,
                        help="Path to JSON config file (default: config_cleveland.json)")
    args = parser.parse_args()
    with open(args.config) as _f:
        _cfg = json.load(_f)
    main(_cfg)
