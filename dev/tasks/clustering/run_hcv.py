#!/usr/bin/env python
"""
HCV Hepatitis C – clustering comparison pipeline.

Step 1 : dataset review (distributions, class balance, missing values)
Step 2 : DBSCAN fine sweep to select reference configurations
Step 3 : exhaustive grid search – TaGra/Gower, Similarity, KNN, Distance
Step 4 : curated comparison table (best configs per method)

Target variable (binary):
    0 = Blood Donor / suspect Blood Donor  (healthy / borderline)
    1 = Hepatitis / Fibrosis / Cirrhosis   (disease)

Usage:
    cd dev/tasks/clustering
    python3 run_hcv.py
"""

import os, sys, json, argparse, numpy as np, pandas as pd, networkx as nx
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
    preprocess_typed, feature_type_map_for_gower,
    HCV_SPEC, HCV_IMPUTE,
    QUALITY_CRITERIA, filter_results,
)

try:
    from hdbscan.validity import validity_index as dbcv_score; DBCV_AVAILABLE = True
except ImportError: DBCV_AVAILABLE = False
try:
    import hdbscan as hdbscan_lib; HDBSCAN_AVAILABLE = True
except ImportError: HDBSCAN_AVAILABLE = False

DATA_PATH  = os.path.join(REPO_ROOT, "examples/datasets/UCI/hcv/hcvdat0.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "clustering_results_hcv")

# ── Data loading ─────────────────────────────────────────────────────────────

def load_hcv():
    df = pd.read_csv(DATA_PATH, index_col=0)
    # Encode Sex: m→1, f→0
    df["Sex"] = (df["Sex"] == "m").astype(float)
    # Binary target: 0=healthy (Blood Donor + suspect), 1=disease
    df["target_binary"] = (~df["Category"].str.startswith("0")).astype(int)
    return df

def prepare_raw_for_gower(df):
    df2 = df.copy()
    for col, strat in HCV_IMPUTE.items():
        if col not in df2.columns: continue
        df2[col] = df2[col].fillna(df2[col].median() if strat == "median" else df2[col].mode()[0])
    feat_cols = [c for c, t in HCV_SPEC.items() if t not in ("target","id") and c in df2.columns]
    return df2[feat_cols].astype(float).values, feat_cols

# ── Metrics ──────────────────────────────────────────────────────────────────

def dbcv(X, lbl):
    if not DBCV_AVAILABLE or len(set(lbl)-{-1}) < 2: return None
    mask = lbl != -1
    if mask.sum() < 2: return None
    try: return float(dbcv_score(X[mask], lbl[mask]))
    except: return None

def sil(X, lbl):
    if len(set(lbl)-{-1}) < 2: return None
    mask = lbl != -1
    if mask.sum() < 2: return None
    try: return float(silhouette_score(X[mask], lbl[mask]))
    except: return None

def vm(true, pred): return float(v_measure_score(true, pred))

# ── Graph helpers ─────────────────────────────────────────────────────────────

def make_graph(n):
    G = nx.Graph(); G.add_nodes_from(range(n)); return G

def graph_cluster(G, n, df_val):
    low = [v for v, d in G.degree() if d < df_val]
    Gf = G.copy(); Gf.remove_nodes_from(low)
    lbl = np.full(n, -1); cid = 0
    for comp in nx.connected_components(Gf):
        for v in comp: lbl[v] = cid
        cid += 1
    return lbl

def dbscan_constructor_labels(X, eps, min_samples):
    n = len(X)
    G = nx.DiGraph(); G.add_nodes_from(range(n))
    DBSCANGraphConstructor(eps=eps, min_samples=min_samples, verbose=False).construct(G, X)
    lbl = np.full(n, -1); cid = 0
    for comp in nx.weakly_connected_components(G):
        if len(comp) == 1:
            node = next(iter(comp))
            if G.in_degree(node) == 0 and G.out_degree(node) == 0: continue
        for node in comp: lbl[node] = cid
        cid += 1
    return lbl

# ── Record / display ──────────────────────────────────────────────────────────

def record(name, X, lbl, true_lbl, ref_lbl=None):
    n_cl = len(set(lbl) - {-1})
    n_ns = int(np.sum(lbl == -1))
    return {
        "name": name, "n_clusters": n_cl,
        "noise_pct": 100 * n_ns / len(lbl),
        "dbcv": dbcv(X, lbl), "sil": sil(X, lbl),
        "vm_true": vm(true_lbl, lbl),
        "vm_ref":  vm(ref_lbl, lbl) if ref_lbl is not None else None,
        "labels": lbl,
    }

def fmt(v, w=7): return f"{v:{w}.4f}" if v is not None else " "*(w-3)+"N/A"

HDR = (f"{'Method':<52} {'Cl':>4} {'Noise%':>7} "
       f"{'DBCV':>8} {'Sil':>8} {'Vm(true)':>9} {'Vm(ref)':>8}")
SEP = "-" * 105

def print_row(r, show_vm_ref=False):
    vr = fmt(r["vm_ref"]) if show_vm_ref and r.get("vm_ref") is not None else "       -"
    print(f"  {r['name']:<52} {r['n_clusters']:>4} {r['noise_pct']:>6.1f}%"
          f" {fmt(r['dbcv']):>8} {fmt(r['sil']):>8}"
          f" {fmt(r['vm_true']):>9} {vr:>8}")

def print_section(title, rows, show_vm_ref=False):
    print(f"\n{'─'*78}\n{title}")
    print(HDR); print(SEP)
    for r in rows: print_row(r, show_vm_ref)


def top3_summary(valid, fh=None):
    """Print (and optionally write) top-3 configs per metric."""
    metrics = [
        ("Lowest Noise%", lambda r: r["noise_pct"],                                False),
        ("Highest DBCV",  lambda r: r["dbcv"]    if r["dbcv"] is not None else -999, True),
        ("Highest Sil",   lambda r: r["sil"]     if r["sil"]  is not None else -999, True),
        ("Highest Vm",    lambda r: r["vm_true"],                                  True),
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

# ── Main ──────────────────────────────────────────────────────────────────────

def main(cfg: dict = None):
    if cfg is None:
        cfg = {}
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_hcv()
    n  = len(df)

    print("=" * 78)
    print("HCV Hepatitis C – Clustering Comparison Pipeline")
    print(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78)

    # ── Step 1: dataset review ────────────────────────────────────────────
    print("\n── Step 1: Dataset review ──")
    print(f"  Samples: {n}  |  Features: {len(HCV_SPEC)-1}")
    print(f"  Category distribution:\n{df['Category'].value_counts().to_string()}")
    print(f"\n  Binary target: 0=healthy {(df['target_binary']==0).sum()}  "
          f"1=disease {(df['target_binary']==1).sum()}")
    miss = df[list(HCV_IMPUTE.keys())].isnull().sum()
    print(f"  Missing:\n{miss[miss>0].to_string()}")

    # ── Preprocessing ─────────────────────────────────────────────────────
    X, feat_names = preprocess_typed(df, HCV_SPEC, HCV_IMPUTE, final_minmax=True)
    assert not np.isnan(X).any(), "NaN in X!"
    true_labels = df["target_binary"].values
    print(f"\n  Preprocessed X: {X.shape}  features: {feat_names}")

    X_raw, raw_cols = prepare_raw_for_gower(df)
    gower_ftypes    = feature_type_map_for_gower(HCV_SPEC, df)
    print(f"  Gower raw: {X_raw.shape}  types: {gower_ftypes}")

    all_results = []

    # ── Step 2: DBSCAN fine sweep ─────────────────────────────────────────
    print("\n── Step 2: DBSCAN fine sweep (eps=0.5–1.5, ms=3–30) ──")
    print(f"  {'eps':>5} {'ms':>4} {'Cl':>4} {'Noise%':>7} {'DBCV':>8} {'Sil':>8} {'Vm':>8}")
    print("  " + "-"*55)

    eps_grid = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    ms_grid  = [3, 5, 7, 10, 13, 15, 20, 25, 30]
    dbscan_sweep = []
    for eps in eps_grid:
        for ms in ms_grid:
            lbl = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
            ncl   = len(set(lbl) - {-1})
            noise = 100 * np.sum(lbl == -1) / n
            d, s  = dbcv(X, lbl), sil(X, lbl)
            v     = vm(true_labels, lbl)
            ds = f"{d:.4f}" if d else "     N/A"
            ss = f"{s:.4f}" if s else "     N/A"
            print(f"  {eps:>5.1f} {ms:>4}  {ncl:>4}  {noise:>6.1f}%  {ds:>8}  {ss:>8}  {v:>8.4f}")
            dbscan_sweep.append((eps, ms, ncl, noise, d, s, v, lbl))

    # ── Step 2b: fine sweep at best eps ──────────────────────────────────
    # Identify the eps with the richest structure (multiple clusters, <60% noise)
    best_eps_candidates = sorted(
        [r for r in dbscan_sweep if 2 <= r[2] <= 8 and r[3] <= 60],
        key=lambda r: (-r[6], r[3])  # max Vm, min noise
    )
    if best_eps_candidates:
        best_eps = best_eps_candidates[0][0]
    else:
        best_eps = 1.1

    print(f"\n  → Fine sweep at eps={best_eps}")
    print(f"  {'ms':>4} {'Cl':>4} {'Noise%':>7} {'DBCV':>8} {'Sil':>8} {'Vm':>8}")
    print("  " + "-"*50)
    for ms in range(3, 45):
        lbl = DBSCAN(eps=best_eps, min_samples=ms).fit_predict(X)
        ncl   = len(set(lbl) - {-1})
        noise = 100 * np.sum(lbl == -1) / n
        d, s  = dbcv(X, lbl), sil(X, lbl)
        v     = vm(true_labels, lbl)
        ds = f"{d:.4f}" if d else "     N/A"
        ss = f"{s:.4f}" if s else "     N/A"
        print(f"  {ms:>4}  {ncl:>4}  {noise:>6.1f}%  {ds:>8}  {ss:>8}  {v:>8.4f}")

    # ── Step 3: select DBSCAN references ─────────────────────────────────
    # From the sweeps above, pick reference configs manually.
    # Default seeds — will be overridden once sweep output is reviewed.
    # Strategy: pick one "low noise" ref and one "high Vm" ref.
    print("\n── Step 3: DBSCAN references + equivalence proof ──")

    # Heuristic auto-selection: best Vm at ≤40% noise, and best Vm overall
    low_noise = sorted(
        [r for r in dbscan_sweep if r[3] <= 40 and r[2] >= 2],
        key=lambda r: -r[6]
    )
    high_vm = sorted(
        [r for r in dbscan_sweep if r[2] >= 2],
        key=lambda r: -r[6]
    )
    refs = []
    seen = set()
    for row in (low_noise[:1] + high_vm[:1]):
        key = (row[0], row[1])
        if key not in seen:
            refs.append(key); seen.add(key)
    # Always include at least one 2-cluster config if available
    two_cl = sorted(
        [r for r in dbscan_sweep if r[2] == 2 and r[3] <= 60],
        key=lambda r: (-r[6], r[3])
    )
    if two_cl:
        key = (two_cl[0][0], two_cl[0][1])
        if key not in seen:
            refs.append(key); seen.add(key)

    print(f"  Auto-selected DBSCAN references: {refs}")

    dbscan_refs = {}
    dbscan_ref_rows = []
    for eps, ms in refs:
        lbl = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
        dbscan_refs[(eps, ms)] = lbl
        r = record(f"DBSCAN (eps={eps}, ms={ms})", X, lbl, true_labels)
        dbscan_ref_rows.append(r)
        all_results.append(r)
    print_section("DBSCAN references", dbscan_ref_rows)

    # TaGra equivalence (last ref config)
    eq_eps, eq_ms = refs[-1]
    lbl_eq = dbscan_constructor_labels(X, eq_eps, eq_ms)
    r_eq = record(f"TaGra/DBSCANConstr (eps={eq_eps}, ms={eq_ms})",
                  X, lbl_eq, true_labels, ref_lbl=dbscan_refs[(eq_eps, eq_ms)])
    all_results.append(r_eq)
    print_section("TaGra DBSCANConstructor equivalence", [r_eq], show_vm_ref=True)

    # Reference for Vm(ref) column in TaGra sections = highest-Vm DBSCAN ref
    best_ref_lbl = max(dbscan_ref_rows, key=lambda r: r["vm_true"])["labels"]

    # ── Step 4: HDBSCAN ──────────────────────────────────────────────────
    hdbscan_rows = []
    if HDBSCAN_AVAILABLE:
        for mcs in [3, 5, 7, 10, 15, 20]:
            lbl = hdbscan_lib.HDBSCAN(min_cluster_size=mcs).fit_predict(X)
            r = record(f"HDBSCAN (mcs={mcs})", X, lbl, true_labels)
            hdbscan_rows.append(r); all_results.append(r)
    print_section("HDBSCAN", hdbscan_rows)

    # ── Step 5: Gower grid ───────────────────────────────────────────────
    print("\n── Step 5: TaGra/Gower exhaustive grid ──")
    gower_rows = []
    for metric, thresh, df_val in product(
        ["range", "quadratic", "gaussian", "laplacian"],
        [0.03,0.05,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,
         0.17,0.18,0.20,0.22,0.25,0.28,0.30,0.35,0.40],
        list(range(1, 16)),
    ):
        G = make_graph(n)
        GowerDistanceConstructor(
            distance_threshold=thresh, feature_types=gower_ftypes,
            continuous_metric=metric, verbose=False,
        ).construct(G, X_raw)
        lbl = graph_cluster(G, n, df_val)
        if 2 <= len(set(lbl)-{-1}) <= 8:
            r = record(f"TaGra/Gower ({metric},t={thresh},df={df_val})",
                       X, lbl, true_labels, ref_lbl=best_ref_lbl)
            gower_rows.append(r); all_results.append(r)

    best_gower = sorted(
        [r for r in gower_rows if r["noise_pct"] <= 60],
        key=lambda r: (-r["vm_true"], r["noise_pct"])
    )[:20]
    print_section("TaGra/Gower  (top 20 by Vm, noise ≤ 60%)", best_gower, show_vm_ref=True)

    # ── Step 6: Similarity grid ──────────────────────────────────────────
    print("\n── Step 6: TaGra/Similarity exhaustive grid ──")
    sim_rows = []
    sim_thresholds = [round(0.50 + i*0.01, 2) for i in range(46)]
    for thresh in sim_thresholds:
        G = make_graph(n)
        SimilarityThresholdConstructor(
            similarity_threshold=thresh, verbose=False
        ).construct(G, X)
        for df_val in range(1, 21):
            lbl = graph_cluster(G, n, df_val)
            if 2 <= len(set(lbl)-{-1}) <= 8:
                r = record(f"TaGra/Sim (t={thresh},df={df_val})",
                           X, lbl, true_labels, ref_lbl=best_ref_lbl)
                sim_rows.append(r); all_results.append(r)

    best_sim = sorted(
        [r for r in sim_rows if r["noise_pct"] <= 60],
        key=lambda r: (-r["vm_true"], r["noise_pct"])
    )[:20]
    print_section("TaGra/Similarity  (top 20 by Vm, noise ≤ 60%)", best_sim, show_vm_ref=True)

    # ── Step 7: KNN grid ─────────────────────────────────────────────────
    print("\n── Step 7: TaGra/KNN exhaustive grid ──")
    knn_rows = []
    for k in range(3, 31):
        G = make_graph(n)
        KNNConstructor(k=k, verbose=False).construct(G, X)
        for df_val in range(1, 21):
            lbl = graph_cluster(G, n, df_val)
            if 2 <= len(set(lbl)-{-1}) <= 8:
                r = record(f"TaGra/KNN (k={k},df={df_val})",
                           X, lbl, true_labels, ref_lbl=best_ref_lbl)
                knn_rows.append(r); all_results.append(r)

    best_knn = sorted(
        [r for r in knn_rows if r["noise_pct"] <= 60],
        key=lambda r: (-r["vm_true"], r["noise_pct"])
    )[:20]
    print_section("TaGra/KNN  (top 20 by Vm, noise ≤ 60%)", best_knn, show_vm_ref=True)

    # ── Step 8: Distance-threshold grid ──────────────────────────────────
    print("\n── Step 8: TaGra/Distance-threshold exhaustive grid ──")
    dist_rows = []
    for thresh in [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]:
        G = make_graph(n)
        DistanceThresholdConstructor(
            distance_threshold=thresh, verbose=False
        ).construct(G, X)
        for df_val in range(1, 26):
            lbl = graph_cluster(G, n, df_val)
            if 2 <= len(set(lbl)-{-1}) <= 8:
                r = record(f"TaGra/Dist (t={thresh},df={df_val})",
                           X, lbl, true_labels, ref_lbl=best_ref_lbl)
                dist_rows.append(r); all_results.append(r)

    best_dist = sorted(
        [r for r in dist_rows if r["noise_pct"] <= 60],
        key=lambda r: (-r["vm_true"], r["noise_pct"])
    )[:20]
    print_section("TaGra/Distance  (top 20 by Vm, noise ≤ 60%)", best_dist, show_vm_ref=True)

    # ── Summary ───────────────────────────────────────────────────────────
    valid     = [r for r in all_results if r["n_clusters"] >= 2]
    valid_low = [r for r in valid if r["noise_pct"] <= 60]

    sum_a_title = "SUMMARY A – ≥2 clusters, noise ≤ 60%, sorted by Vm(true) ↓"
    sum_a_rows  = sorted(valid_low, key=lambda r: r["vm_true"], reverse=True)
    sum_b_title = "SUMMARY B – ≥2 clusters, noise ≤ 60%, sorted by DBCV ↓ then Noise% ↑"
    sum_b_rows  = sorted([r for r in valid_low if r["dbcv"] is not None],
                         key=lambda r: (-r["dbcv"], r["noise_pct"]))

    print(f"\n{'='*78}")
    print(sum_a_title)
    print(HDR); print(SEP)
    for r in sum_a_rows:
        print_row(r, show_vm_ref=r.get("vm_ref") is not None)

    print(f"\n{'='*78}")
    print(sum_b_title)
    print(HDR); print(SEP)
    for r in sum_b_rows:
        print_row(r, show_vm_ref=r.get("vm_ref") is not None)

    # ── Curated comparison (hardcoded best configs) ───────────────────────
    # Selected based on sweep results:
    #   DBSCAN Ref A : eps=0.5  ms=30  → 2cl, 3.4% noise, DBCV=0.591, Vm=0.071
    #   Gower/range  : t=0.05 df=2    → 2cl, 12.8% noise, DBCV=0.731, Vm=0.228
    #   Gower/gauss  : t=0.14 df=1    → 3cl,  10.9% noise, DBCV=-0.496, Vm=0.434
    #                  t=0.20 df=1    → 2cl,   5.9% noise, DBCV=0.002, Vm=0.290
    #   TaGra/Dist   : t=0.3  df=6   → 2cl, 12.5% noise, DBCV=0.721, Vm=0.188
    #   TaGra/Sim    : t=0.94 df=11  → 2cl,  6.0% noise, DBCV=0.678, Vm=0.142

    _default_curated = [
        ("dbscan",        {"eps": 0.5,  "ms": 30}),
        ("dbscan_constr", {"eps": 0.5,  "ms": 30}),
        ("hdbscan",       {"mcs": 20}),
        ("gower",         {"metric": "range",    "t": 0.05, "df": 2}),
        ("gower",         {"metric": "gaussian", "t": 0.20, "df": 1}),
        ("gower",         {"metric": "gaussian", "t": 0.14, "df": 1}),
        ("dist",          {"t": 0.3,  "df": 6}),
        ("sim",           {"t": 0.94, "df": 11}),
    ]
    if "curated_configs" in cfg:
        curated_configs = [
            (item["method"], {k: v for k, v in item.items() if k != "method"})
            for item in cfg["curated_configs"]
        ]
    else:
        curated_configs = _default_curated

    curated_rows = []
    ref_lbl_A = DBSCAN(eps=0.5, min_samples=30).fit_predict(X)

    for method, p in curated_configs:
        if method == "dbscan":
            lbl = DBSCAN(eps=p["eps"], min_samples=p["ms"]).fit_predict(X)
            name = f"DBSCAN (eps={p['eps']}, ms={p['ms']})"
            r = record(name, X, lbl, true_labels)
        elif method == "dbscan_constr":
            lbl = dbscan_constructor_labels(X, p["eps"], p["ms"])
            name = f"TaGra/DBSCANConstr (eps={p['eps']}, ms={p['ms']})"
            r = record(name, X, lbl, true_labels, ref_lbl=ref_lbl_A)
        elif method == "hdbscan" and HDBSCAN_AVAILABLE:
            lbl = hdbscan_lib.HDBSCAN(min_cluster_size=p["mcs"]).fit_predict(X)
            name = f"HDBSCAN (mcs={p['mcs']})"
            r = record(name, X, lbl, true_labels)
        elif method == "gower":
            G = make_graph(n)
            GowerDistanceConstructor(
                distance_threshold=p["t"], feature_types=gower_ftypes,
                continuous_metric=p["metric"], verbose=False,
            ).construct(G, X_raw)
            lbl = graph_cluster(G, n, p["df"])
            name = f"TaGra/Gower ({p['metric']},t={p['t']},df={p['df']})"
            r = record(name, X, lbl, true_labels, ref_lbl=ref_lbl_A)
        elif method == "dist":
            G = make_graph(n)
            DistanceThresholdConstructor(
                distance_threshold=p["t"], verbose=False
            ).construct(G, X)
            lbl = graph_cluster(G, n, p["df"])
            name = f"TaGra/Dist (t={p['t']},df={p['df']})"
            r = record(name, X, lbl, true_labels, ref_lbl=ref_lbl_A)
        elif method == "sim":
            G = make_graph(n)
            SimilarityThresholdConstructor(
                similarity_threshold=p["t"], verbose=False
            ).construct(G, X)
            lbl = graph_cluster(G, n, p["df"])
            name = f"TaGra/Sim (t={p['t']},df={p['df']})"
            r = record(name, X, lbl, true_labels, ref_lbl=ref_lbl_A)
        else:
            continue
        curated_rows.append(r)

    curated_title = ("CURATED COMPARISON – best selected configs\n"
                     "  Vm(ref) = Vm vs DBSCAN Ref A (eps=0.5, ms=30)")
    print_section(curated_title, curated_rows, show_vm_ref=True)

    # ── Quality-filtered summary ───────────────────────────────────────────
    qc = {**QUALITY_CRITERIA, **cfg.get("quality_criteria", {})}
    qc_title = (
        f"QUALITY-FILTERED  "
        f"(DBCV≥{qc['min_dbcv']}, noise<{qc['max_noise_pct']}%, "
        f"min cluster≥{qc['min_cluster_size_pct']}% of non-noise, "
        f"clusters≥{qc['min_n_clusters']})"
    )
    qc_rows = filter_results(all_results, criteria=qc)
    print_section(qc_title, sorted(qc_rows, key=lambda r: r["vm_true"], reverse=True),
                  show_vm_ref=True)
    if not qc_rows:
        print("  (no results pass all criteria)")

    # ── Save ──────────────────────────────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, "hcv_report.txt")
    with open(report_path, "w") as f:
        def fw(s=""):
            f.write(s + "\n")

        fw("HCV Hepatitis C – Clustering Comparison")
        fw(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
        fw(f"n={n}  features={X.shape[1]}  Gower types: {gower_ftypes}")
        fw()
        fw("=" * 78)
        fw("ALL RESULTS")
        fw(HDR); fw(SEP)
        for r in all_results:
            vr = fmt(r["vm_ref"]) if r.get("vm_ref") is not None else "       -"
            f.write(
                f"  {r['name']:<52} {r['n_clusters']:>4} {r['noise_pct']:>6.1f}%"
                f" {fmt(r['dbcv']):>8} {fmt(r['sil']):>8}"
                f" {fmt(r['vm_true']):>9} {vr:>8}\n"
            )
        fw()
        fw("=" * 78)
        fw(sum_a_title)
        fw(HDR); fw(SEP)
        for r in sum_a_rows:
            vr = fmt(r["vm_ref"]) if r.get("vm_ref") is not None else "       -"
            f.write(
                f"  {r['name']:<52} {r['n_clusters']:>4} {r['noise_pct']:>6.1f}%"
                f" {fmt(r['dbcv']):>8} {fmt(r['sil']):>8}"
                f" {fmt(r['vm_true']):>9} {vr:>8}\n"
            )
        fw()
        fw("=" * 78)
        fw(sum_b_title)
        fw(HDR); fw(SEP)
        for r in sum_b_rows:
            vr = fmt(r["vm_ref"]) if r.get("vm_ref") is not None else "       -"
            f.write(
                f"  {r['name']:<52} {r['n_clusters']:>4} {r['noise_pct']:>6.1f}%"
                f" {fmt(r['dbcv']):>8} {fmt(r['sil']):>8}"
                f" {fmt(r['vm_true']):>9} {vr:>8}\n"
            )
        fw()
        fw("=" * 78)
        fw(curated_title)
        fw(HDR); fw(SEP)
        for r in curated_rows:
            vr = fmt(r["vm_ref"]) if r.get("vm_ref") is not None else "       -"
            f.write(
                f"  {r['name']:<52} {r['n_clusters']:>4} {r['noise_pct']:>6.1f}%"
                f" {fmt(r['dbcv']):>8} {fmt(r['sil']):>8}"
                f" {fmt(r['vm_true']):>9} {vr:>8}\n"
            )
        fw("=" * 78)
        fw(qc_title)
        fw(HDR); fw(SEP)
        if qc_rows:
            for r in sorted(qc_rows, key=lambda r: r["vm_true"], reverse=True):
                vr = fmt(r["vm_ref"]) if r.get("vm_ref") is not None else "       -"
                f.write(
                    f"  {r['name']:<52} {r['n_clusters']:>4} {r['noise_pct']:>6.1f}%"
                    f" {fmt(r['dbcv']):>8} {fmt(r['sil']):>8}"
                    f" {fmt(r['vm_true']):>9} {vr:>8}\n"
                )
        else:
            fw("  (no results pass all criteria)")
        top3_summary(valid_low, fh=f)

    print(f"\nReport saved: {report_path}")
    print(f"Total configurations tested: {len(all_results)}")


if __name__ == "__main__":
    _default_cfg = os.path.join(os.path.dirname(__file__), "config_hcv.json")
    parser = argparse.ArgumentParser(description="HCV clustering comparison")
    parser.add_argument("--config", default=_default_cfg,
                        help="Path to JSON config file (default: config_hcv.json)")
    args = parser.parse_args()
    with open(args.config) as _f:
        _cfg = json.load(_f)
    main(_cfg)
