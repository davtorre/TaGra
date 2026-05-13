#!/usr/bin/env python
"""
Parameter search for DBSCAN and TaGra graph-based clustering methods,
implementing the SS-DBSCAN framework (Monko & Kimura, 2023).

For each method the approach is:
  1. Algorithm 1 (Stratified Sampling) – estimate the optimal threshold from
     the k-distance (or k-similarity) graph of the data.
     - DBSCAN / TaGra/Dist  : Euclidean k-NN distances  (ascending curve)
     - TaGra/Sim             : cosine k-NN similarities (descending curve)
     - TaGra/Gower           : Gower k-NN distances per metric (ascending)
     - TaGra/KNN             : no threshold to estimate; swept directly
  2. Grid search – sweep the threshold (or k for KNN) in a neighbourhood of
     the estimate × the degree-filter (df) range.
  3. Score every valid configuration (≥2 clusters) with Silhouette, DBCV,
     and noise %.

Output is directly paste-compatible with the project's JSON config format.

Usage:
    cd dev/experiments/clustering
    python search_params.py \\
        --bundle preprocessing/bundles/hcv_bundle.npz

    python search_params.py \\
        --bundle preprocessing/bundles/cleveland_bundle.npz \\
        --methods dbscan dist knn sim gower \\
        --k 5 --strata 5 --m 10 --n-t 10 --top 5 \\
        --df-max 20 --k-max 30
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# ── Path setup for TaGra imports ──────────────────────────────────────────────

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT    = os.path.abspath(os.path.join(_SCRIPT_DIR, "../../.."))
_PREPROC_DIR  = os.path.abspath(os.path.join(_SCRIPT_DIR, "preprocessing"))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _PREPROC_DIR)

from tagra.construction.distance   import DistanceThresholdConstructor
from tagra.construction.knn        import KNNConstructor
from tagra.construction.similarity import SimilarityThresholdConstructor
from tagra.construction.gower      import GowerDistanceConstructor

# ── Optional DBCV ─────────────────────────────────────────────────────────────

try:
    from hdbscan.validity import validity_index as _dbcv_fn
    DBCV_AVAILABLE = True
except ImportError:
    DBCV_AVAILABLE = False

GOWER_METRICS = ("range", "quadratic", "gaussian", "laplacian")


# ══════════════════════════════════════════════════════════════════════════════
# Knee / Elbow detection  (no external library required)
# ══════════════════════════════════════════════════════════════════════════════

def _elbow_from_curve(curve: np.ndarray) -> float:
    """
    Maximum perpendicular distance from the chord connecting the first and
    last points.  Works for both ascending and descending curves.
    """
    n = len(curve)
    x = np.arange(n, dtype=float)
    y = curve.astype(float)

    x_n = x / x[-1] if x[-1] > 0 else x
    y_range = y[-1] - y[0]
    y_n = (y - y[0]) / y_range if y_range != 0 else np.zeros_like(y)

    # Reference line: from (0,0) to (1,1) in normalised space → y = x
    dists = np.abs(y_n - x_n) / np.sqrt(2.0)
    return float(curve[int(np.argmax(dists))])


def _knee_poly(curve: np.ndarray, poly_degree: int = 5) -> float:
    """
    Polynomial fitting + maximum curvature κ = |y''| / (1 + y'^2)^{3/2}.
    Falls back to elbow method when the fit is degenerate.
    Works on both ascending (distances) and descending (similarities) curves.
    """
    n = len(curve)
    if n < poly_degree + 2:
        return _elbow_from_curve(curve)

    x = np.linspace(0.0, 1.0, n)
    y_min, y_max = curve.min(), curve.max()
    if y_max == y_min:
        return float(curve[0])
    y = (curve - y_min) / (y_max - y_min)

    coeffs = np.polyfit(x, y, poly_degree)
    poly   = np.poly1d(coeffs)
    d1, d2 = poly.deriv(1), poly.deriv(2)

    y1, y2    = d1(x), d2(x)
    curvature = np.abs(y2) / (1.0 + y1 ** 2) ** 1.5
    return float(curve[int(np.argmax(curvature))])


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm 1 – threshold estimation via stratified sampling
# ══════════════════════════════════════════════════════════════════════════════

def _stratified_sample(values: np.ndarray, p: int, m: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Divide *sorted* `values` into p equal-count strata, draw m from each."""
    strata = np.array_split(np.sort(values), p)
    sampled: list[float] = []
    for stratum in strata:
        sz = len(stratum)
        if sz == 0:
            continue
        idxs = rng.choice(sz, size=min(m, sz), replace=False)
        sampled.extend(stratum[idxs].tolist())
    return np.array(sampled)


def estimate_threshold_euclidean(
    X: np.ndarray, k: int = 5, p: int = 5, m: int = 10,
    poly_degree: int = 5, random_state: int = 42,
) -> tuple[float, np.ndarray]:
    """
    Algorithm 1 for Euclidean k-NN distances.
    Used for both DBSCAN eps and TaGra/Dist threshold.
    """
    rng  = np.random.default_rng(random_state)
    n    = len(X)
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(X)
    d, _ = nbrs.kneighbors(X)
    averages = d[:, 1:].mean(axis=1)               # exclude self

    sampled = np.sort(_stratified_sample(averages, p, m, rng))
    t_opt   = _knee_poly(sampled, poly_degree)
    return float(t_opt), sampled


def estimate_threshold_sim(
    X: np.ndarray, k: int = 5, p: int = 5, m: int = 10,
    poly_degree: int = 5, random_state: int = 42,
) -> tuple[float, np.ndarray]:
    """
    Algorithm 1 adapted for cosine similarity.

    For each point, average the cosine similarities to its k most-similar
    neighbours (excluding self).  Stratified-sample the averages, sort
    *descending*, and find the knee of the descending similarity curve.
    The knee gives the optimal similarity threshold.
    """
    rng  = np.random.default_rng(random_state)
    n    = len(X)
    S    = cosine_similarity(X)                     # (n, n), in [-1, 1]
    np.fill_diagonal(S, -np.inf)                    # exclude self

    k_use  = min(k, n - 1)
    # k highest similarities for each point
    top_k  = np.partition(S, -k_use, axis=1)[:, -k_use:]
    averages = np.clip(top_k.mean(axis=1), 0.0, 1.0)

    sampled = np.sort(_stratified_sample(averages, p, m, rng))[::-1]  # descending
    t_opt   = _knee_poly(sampled, poly_degree)
    return float(t_opt), sampled


def estimate_threshold_gower(
    X_raw: np.ndarray, gower_ftypes: list[str],
    metric: str = "range", k: int = 5, p: int = 5, m: int = 10,
    poly_degree: int = 5, random_state: int = 42,
) -> tuple[float, np.ndarray]:
    """
    Algorithm 1 adapted for Gower distances.

    Computes the full Gower distance matrix (O(n²)) for the given
    continuous_metric, then applies stratified sampling + knee detection.
    """
    rng  = np.random.default_rng(random_state)
    n    = len(X_raw)

    gdc = GowerDistanceConstructor(
        distance_threshold=0.5,   # placeholder; only the matrix is used
        feature_types=gower_ftypes,
        continuous_metric=metric,
        verbose=False,
    )
    D = gdc._gower_matrix(X_raw.astype(float), gower_ftypes)  # (n, n)
    np.fill_diagonal(D, np.inf)

    k_use    = min(k, n - 1)
    # k smallest distances for each point
    bot_k    = np.partition(D, k_use, axis=1)[:, :k_use]
    averages = bot_k.mean(axis=1)

    sampled = np.sort(_stratified_sample(averages, p, m, rng))
    t_opt   = _knee_poly(sampled, poly_degree)
    return float(t_opt), sampled, D    # return D so the caller can reuse it


# ══════════════════════════════════════════════════════════════════════════════
# Graph utilities  (mirrors run.py)
# ══════════════════════════════════════════════════════════════════════════════

def _make_graph(n: int) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n))
    return G


def _graph_cluster(G: nx.Graph, n: int, df_val: int) -> np.ndarray:
    """Remove nodes with degree < df_val, extract connected components."""
    low = [v for v, d in G.degree() if d < df_val]
    Gf  = G.copy()
    Gf.remove_nodes_from(low)
    lbl = np.full(n, -1)
    cid = 0
    for comp in nx.connected_components(Gf):
        for v in comp:
            lbl[v] = cid
        cid += 1
    return lbl


# ══════════════════════════════════════════════════════════════════════════════
# Scoring helpers
# ══════════════════════════════════════════════════════════════════════════════

def _silhouette(X: np.ndarray, labels: np.ndarray) -> float | None:
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return None
    try:
        return float(silhouette_score(X[mask], labels[mask]))
    except Exception:
        return None


def _dbcv(X: np.ndarray, labels: np.ndarray) -> float | None:
    if not DBCV_AVAILABLE:
        return None
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return None
    try:
        return float(_dbcv_fn(X[mask], labels[mask]))
    except Exception:
        return None


def _score(X: np.ndarray, labels: np.ndarray, n: int) -> dict:
    sil = _silhouette(X, labels)
    dcv = _dbcv(X, labels)
    return {
        "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
        "noise_pct":  round(100.0 * int(np.sum(labels == -1)) / n, 2),
        "silhouette": round(sil, 6) if sil is not None else None,
        "dbcv":       round(dcv, 6) if dcv is not None else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Grid searches
# ══════════════════════════════════════════════════════════════════════════════

def _progress(done: int, total: int) -> None:
    if total > 0 and done % max(1, total // 20) == 0:
        print(f"  {done}/{total} ({100*done//total}%)", end="\r", flush=True)


def grid_search_dbscan(
    X: np.ndarray, eps_opt: float,
    eps_scale: tuple[float, float] = (0.5, 1.5), n_eps: int = 10,
    ms_min: int = 2, ms_max: int = 100, verbose: bool = True,
) -> list[dict]:
    """Grid search for DBSCAN (eps × min_samples)."""
    n      = len(X)
    eps_vals = np.linspace(eps_opt * eps_scale[0], eps_opt * eps_scale[1], n_eps)
    total  = n_eps * (ms_max - ms_min + 1)
    done   = 0
    results: list[dict] = []

    for eps in eps_vals:
        for ms in range(ms_min, ms_max + 1):
            done += 1
            if verbose: _progress(done, total)
            labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
            if len(set(labels) - {-1}) < 2:
                continue
            r = {"method": "dbscan", "eps": round(float(eps), 4), "ms": int(ms)}
            r.update(_score(X, labels, n))
            results.append(r)

    if verbose: print()
    return results


def grid_search_dist(
    X: np.ndarray, t_opt: float,
    t_scale: tuple[float, float] = (0.5, 1.5), n_t: int = 10,
    df_min: int = 1, df_max: int = 20, verbose: bool = True,
) -> list[dict]:
    """Grid search for TaGra/Dist (threshold × df)."""
    n      = len(X)
    t_vals = np.linspace(t_opt * t_scale[0], t_opt * t_scale[1], n_t)
    total  = n_t * (df_max - df_min + 1)
    done   = 0
    results: list[dict] = []

    for t in t_vals:
        G = _make_graph(n)
        DistanceThresholdConstructor(distance_threshold=float(t), verbose=False).construct(G, X)
        for df in range(df_min, df_max + 1):
            done += 1
            if verbose: _progress(done, total)
            labels = _graph_cluster(G, n, df)
            if len(set(labels) - {-1}) < 2:
                continue
            r = {"method": "dist", "t": round(float(t), 4), "df": int(df)}
            r.update(_score(X, labels, n))
            results.append(r)

    if verbose: print()
    return results


def grid_search_knn(
    X: np.ndarray, k_min: int = 2, k_max: int = 30,
    df_min: int = 1, df_max: int = 20, verbose: bool = True,
) -> list[dict]:
    """Grid search for TaGra/KNN (k × df).  No threshold estimation needed."""
    n      = len(X)
    total  = (k_max - k_min + 1) * (df_max - df_min + 1)
    done   = 0
    results: list[dict] = []

    for k in range(k_min, k_max + 1):
        G = _make_graph(n)
        KNNConstructor(k=k, verbose=False).construct(G, X)
        for df in range(df_min, df_max + 1):
            done += 1
            if verbose: _progress(done, total)
            labels = _graph_cluster(G, n, df)
            if len(set(labels) - {-1}) < 2:
                continue
            r = {"method": "knn", "k": int(k), "df": int(df)}
            r.update(_score(X, labels, n))
            results.append(r)

    if verbose: print()
    return results


def grid_search_sim(
    X: np.ndarray, t_opt: float,
    t_scale: tuple[float, float] = (0.9, 1.1), n_t: int = 10,
    df_min: int = 1, df_max: int = 20, verbose: bool = True,
) -> list[dict]:
    """
    Grid search for TaGra/Sim (similarity threshold × df).

    For similarity, higher = more edges, so the search range is a *narrower*
    multiplicative band (default 0.9–1.1).  Values are clamped to [0, 1].
    """
    n      = len(X)
    lo     = max(0.0, t_opt * t_scale[0])
    hi     = min(1.0, t_opt * t_scale[1])
    t_vals = np.linspace(lo, hi, n_t)
    total  = n_t * (df_max - df_min + 1)
    done   = 0
    results: list[dict] = []

    for t in t_vals:
        G = _make_graph(n)
        SimilarityThresholdConstructor(
            similarity_threshold=float(t), verbose=False
        ).construct(G, X)
        for df in range(df_min, df_max + 1):
            done += 1
            if verbose: _progress(done, total)
            labels = _graph_cluster(G, n, df)
            if len(set(labels) - {-1}) < 2:
                continue
            r = {"method": "sim", "t": round(float(t), 4), "df": int(df)}
            r.update(_score(X, labels, n))
            results.append(r)

    if verbose: print()
    return results


def grid_search_gower(
    X_raw: np.ndarray, X: np.ndarray,
    gower_ftypes: list[str], gower_metrics: list[str],
    t_opts: dict[str, float], precomputed_D: dict[str, np.ndarray],
    t_scale: tuple[float, float] = (0.5, 1.5), n_t: int = 10,
    df_min: int = 1, df_max: int = 20, verbose: bool = True,
) -> list[dict]:
    """
    Grid search for TaGra/Gower (metric × threshold × df).

    The Gower distance matrix for each metric is precomputed once (passed in
    via `precomputed_D`) so the threshold sweep only rebuilds the graph mask.
    """
    n       = len(X_raw)
    total   = len(gower_metrics) * n_t * (df_max - df_min + 1)
    done    = 0
    results: list[dict] = []

    for metric in gower_metrics:
        D     = precomputed_D[metric]              # (n, n) Gower matrix
        t_opt = t_opts[metric]
        t_vals = np.linspace(t_opt * t_scale[0], t_opt * t_scale[1], n_t)
        t_vals = np.clip(t_vals, 1e-6, 1.0)       # Gower threshold ∈ (0, 1]

        for t in t_vals:
            # Build graph directly from precomputed matrix
            G    = _make_graph(n)
            mask = np.tril(D <= float(t), k=-1)
            rows_idx, cols_idx = np.where(mask)
            for i, j in zip(rows_idx, cols_idx):
                G.add_edge(int(i), int(j))

            for df in range(df_min, df_max + 1):
                done += 1
                if verbose: _progress(done, total)
                labels = _graph_cluster(G, n, df)
                if len(set(labels) - {-1}) < 2:
                    continue
                r = {
                    "method": "gower", "metric": metric,
                    "t": round(float(t), 4), "df": int(df),
                }
                r.update(_score(X, labels, n))
                results.append(r)

    if verbose: print()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Output helpers
# ══════════════════════════════════════════════════════════════════════════════

# Metric registry: (field_name, sort_descending, always_available)
_METRICS = [
    ("silhouette", True,  True),
    ("dbcv",       True,  False),
    ("noise_pct",  False, True),
]


def _result_name(r: dict) -> str:
    """Human-readable label for a result row (matches run.py naming)."""
    m = r["method"]
    if m == "dbscan":
        return f"DBSCAN (eps={r['eps']}, ms={r['ms']})"
    if m == "dist":
        return f"TaGra/Dist (t={r['t']}, df={r['df']})"
    if m == "knn":
        return f"TaGra/KNN (k={r['k']}, df={r['df']})"
    if m == "sim":
        return f"TaGra/Sim (t={r['t']}, df={r['df']})"
    if m == "gower":
        return f"TaGra/Gower ({r['metric']}, t={r['t']}, df={r['df']})"
    return str(r)


def _json_entry(r: dict) -> str:
    """One-line JSON entry for the project's curated_configs format."""
    m = r["method"]
    if m == "dbscan":
        return f'{{"method": "dbscan",        "eps": {r["eps"]}, "ms": {r["ms"]}}}'
    if m == "dist":
        return f'{{"method": "dist",           "t": {r["t"]}, "df": {r["df"]}}}'
    if m == "knn":
        return f'{{"method": "knn",            "k": {r["k"]}, "df": {r["df"]}}}'
    if m == "sim":
        return f'{{"method": "sim",            "t": {r["t"]}, "df": {r["df"]}}}'
    if m == "gower":
        return (f'{{"method": "gower",  "metric": "{r["metric"]}", '
                f'"t": {r["t"]}, "df": {r["df"]}}}')
    return ""


def _dbscan_ref_entry(r: dict) -> str | None:
    """dbscan_refs entry (only for DBSCAN results)."""
    if r["method"] == "dbscan":
        return f'{{"eps": {r["eps"]}, "ms": {r["ms"]}}}'
    return None


_FMT_V  = lambda v: f"{v:10.4f}" if v is not None else "       N/A"
_HDR    = (f"  {'Name':<46} {'n_cl':>4} {'noise%':>7} "
           f"{'silhouette':>11} {'dbcv':>10}")
_SEP    = "  " + "-" * 80


def _print_row(r: dict) -> None:
    print(
        f"  {_result_name(r):<46} {r['n_clusters']:>4} "
        f"{r['noise_pct']:>6.1f}%  {_FMT_V(r['silhouette'])} {_FMT_V(r['dbcv'])}"
    )


def _ranked(rows: list[dict], sort_key: str, reverse: bool, top: int) -> list[dict]:
    valid = [r for r in rows if r.get(sort_key) is not None]
    return sorted(valid, key=lambda r: r[sort_key], reverse=reverse)[:top]


def _print_table(rows: list[dict], sort_key: str, top: int,
                  reverse: bool = True) -> None:
    direction = "↑ highest" if reverse else "↓ lowest"
    print(f"\n{'='*82}")
    print(f"  TOP {top} by {sort_key.upper()}  ({direction})")
    print(f"{'='*82}")
    print(_HDR); print(_SEP)
    for r in _ranked(rows, sort_key, reverse, top):
        _print_row(r)


def _print_json_snippet(rows: list[dict], sort_key: str, top: int,
                         reverse: bool = True) -> None:
    ranked = _ranked(rows, sort_key, reverse, top)
    # Deduplicate while preserving order
    seen: set[tuple] = set()
    unique: list[dict] = []
    for r in ranked:
        key = (r["method"],) + tuple(
            r[k] for k in ("eps", "ms", "t", "df", "k", "metric") if k in r
        )
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print(f"\n  ── JSON snippet (sorted by {sort_key}) ──")

    # dbscan_refs block (DBSCAN rows only)
    refs = [_dbscan_ref_entry(r) for r in unique if r["method"] == "dbscan"]
    if refs:
        print('  "dbscan_refs": [')
        for e in refs:
            print(f"    {e},")
        print("  ]")

    # curated_configs block
    print('  "curated_configs" entries:')
    for r in unique:
        print(f"    {_json_entry(r)},")
        if r["method"] == "dbscan":
            print(f'    {{"method": "dbscan_constr", "eps": {r["eps"]}, "ms": {r["ms"]}}},')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SS-DBSCAN + TaGra parameter search"
    )
    parser.add_argument(
        "--bundle", required=True,
        help="Path to preprocessed bundle .npz"
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=["dbscan", "dist", "knn", "sim", "gower"],
        choices=["dbscan", "dist", "knn", "sim", "gower"],
        help="Methods to search (default: all)"
    )
    # Algorithm 1 params
    parser.add_argument("--k",      type=int, default=5,
        help="k-NN for threshold estimation  (default: 5)")
    parser.add_argument("--strata", type=int, default=5,
        help="Number of strata for stratified sampling  (default: 5)")
    parser.add_argument("--m",      type=int, default=10,
        help="Samples drawn per stratum  (default: 10)")
    # Grid search params (shared)
    parser.add_argument("--n-t",    type=int, default=10,
        help="Number of threshold values to sweep  (default: 10)")
    parser.add_argument("--t-scale", type=float, nargs=2, default=[0.5, 1.5],
        metavar=("LO", "HI"),
        help="Threshold search range as multiples of t_opt  (default: 0.5 1.5)")
    parser.add_argument("--sim-scale", type=float, nargs=2, default=[0.9, 1.1],
        metavar=("LO", "HI"),
        help="Similarity threshold range around t_opt  (default: 0.9 1.1)")
    # DBSCAN-specific
    parser.add_argument("--ms-min", type=int, default=2,
        help="DBSCAN: minimum min_samples  (default: 2)")
    parser.add_argument("--ms-max", type=int, default=None,
        help="DBSCAN: maximum min_samples  (default: 100 or 200)")
    # TaGra shared
    parser.add_argument("--df-min", type=int, default=1,
        help="TaGra: minimum degree filter  (default: 1)")
    parser.add_argument("--df-max", type=int, default=20,
        help="TaGra: maximum degree filter  (default: 20)")
    # KNN-specific
    parser.add_argument("--k-min",  type=int, default=2,
        help="KNN: minimum k  (default: 2)")
    parser.add_argument("--k-max",  type=int, default=30,
        help="KNN: maximum k  (default: 30)")
    # Gower-specific
    parser.add_argument("--gower-metrics", nargs="+", default=list(GOWER_METRICS),
        choices=list(GOWER_METRICS),
        help="Gower continuous metrics to try  (default: all 4)")
    # Output
    parser.add_argument("--top",    type=int, default=5,
        help="Top-N results to display per metric  (default: 5)")
    parser.add_argument("--seed",   type=int, default=42,
        help="Random seed for stratified sampling  (default: 42)")
    args = parser.parse_args()

    # ── Load bundle ───────────────────────────────────────────────────────
    if not os.path.exists(args.bundle):
        sys.exit(f"Bundle not found: {args.bundle}")

    bundle = np.load(args.bundle)
    X      = bundle["X"]
    X_raw  = bundle.get("X_raw", None)
    n, d   = X.shape

    meta_path    = args.bundle.replace("_bundle.npz", "_meta.json")
    dataset_name = os.path.basename(args.bundle).replace("_bundle.npz", "")
    gower_ftypes = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        dataset_name = meta.get("dataset_name", dataset_name)
        gower_ftypes = meta.get("gower_ftypes", None)

    ms_max = args.ms_max if args.ms_max is not None else (100 if n <= 3000 else 200)

    print("=" * 82)
    print(f"Parameter Search  —  {dataset_name}   (n={n}, d={d})")
    print(f"Methods : {', '.join(args.methods)}")
    print(f"Alg. 1  : k={args.k}, strata={args.strata}, m={args.m}")
    print(f"DBCV    : {'available' if DBCV_AVAILABLE else 'unavailable (install hdbscan)'}")
    print("=" * 82)

    # ── Shared Algorithm 1 for Euclidean (DBSCAN + Dist reuse) ────────────
    euc_t_opt = None
    if "dbscan" in args.methods or "dist" in args.methods:
        print("\n[Alg. 1] Euclidean threshold estimation …")
        euc_t_opt, _ = estimate_threshold_euclidean(
            X, k=args.k, p=args.strata, m=args.m, random_state=args.seed
        )
        print(f"         ✓ t_opt (Euclidean) = {euc_t_opt:.4f}")

    all_results: list[dict] = []

    # ─────────────────────────────────────────────────────────────────────
    # DBSCAN
    # ─────────────────────────────────────────────────────────────────────
    if "dbscan" in args.methods:
        lo, hi = euc_t_opt * args.t_scale[0], euc_t_opt * args.t_scale[1]
        print(f"\n[DBSCAN] Grid search: eps ∈ [{lo:.4f}, {hi:.4f}], "
              f"ms ∈ [{args.ms_min}, {ms_max}] …")
        res = grid_search_dbscan(
            X, euc_t_opt, tuple(args.t_scale), args.n_t,
            args.ms_min, ms_max, verbose=True,
        )
        print(f"         {len(res)} valid configurations.")
        all_results.extend(res)

        active = [
            (k, rev) for k, rev, always in _METRICS
            if always or (k == "dbcv" and DBCV_AVAILABLE)
        ]
        for sort_key, reverse in active:
            _print_table(res, sort_key, args.top, reverse)
            _print_json_snippet(res, sort_key, args.top, reverse)

    # ─────────────────────────────────────────────────────────────────────
    # TaGra / Distance
    # ─────────────────────────────────────────────────────────────────────
    if "dist" in args.methods:
        lo, hi = euc_t_opt * args.t_scale[0], euc_t_opt * args.t_scale[1]
        print(f"\n[Dist]   Grid search: t ∈ [{lo:.4f}, {hi:.4f}], "
              f"df ∈ [{args.df_min}, {args.df_max}] …")
        res = grid_search_dist(
            X, euc_t_opt, tuple(args.t_scale), args.n_t,
            args.df_min, args.df_max, verbose=True,
        )
        print(f"         {len(res)} valid configurations.")
        all_results.extend(res)

        active = [
            (k, rev) for k, rev, always in _METRICS
            if always or (k == "dbcv" and DBCV_AVAILABLE)
        ]
        for sort_key, reverse in active:
            _print_table(res, sort_key, args.top, reverse)
            _print_json_snippet(res, sort_key, args.top, reverse)

    # ─────────────────────────────────────────────────────────────────────
    # TaGra / KNN
    # ─────────────────────────────────────────────────────────────────────
    if "knn" in args.methods:
        print(f"\n[KNN]    Grid search: k ∈ [{args.k_min}, {args.k_max}], "
              f"df ∈ [{args.df_min}, {args.df_max}] …")
        res = grid_search_knn(
            X, args.k_min, args.k_max,
            args.df_min, args.df_max, verbose=True,
        )
        print(f"         {len(res)} valid configurations.")
        all_results.extend(res)

        active = [
            (k, rev) for k, rev, always in _METRICS
            if always or (k == "dbcv" and DBCV_AVAILABLE)
        ]
        for sort_key, reverse in active:
            _print_table(res, sort_key, args.top, reverse)
            _print_json_snippet(res, sort_key, args.top, reverse)

    # ─────────────────────────────────────────────────────────────────────
    # TaGra / Similarity
    # ─────────────────────────────────────────────────────────────────────
    if "sim" in args.methods:
        print("\n[Alg. 1] Cosine similarity threshold estimation …")
        sim_t_opt, _ = estimate_threshold_sim(
            X, k=args.k, p=args.strata, m=args.m, random_state=args.seed
        )
        print(f"         ✓ t_opt (Sim) = {sim_t_opt:.4f}")

        lo = max(0.0, sim_t_opt * args.sim_scale[0])
        hi = min(1.0, sim_t_opt * args.sim_scale[1])
        print(f"\n[Sim]    Grid search: t ∈ [{lo:.4f}, {hi:.4f}], "
              f"df ∈ [{args.df_min}, {args.df_max}] …")
        res = grid_search_sim(
            X, sim_t_opt, tuple(args.sim_scale), args.n_t,
            args.df_min, args.df_max, verbose=True,
        )
        print(f"         {len(res)} valid configurations.")
        all_results.extend(res)

        active = [
            (k, rev) for k, rev, always in _METRICS
            if always or (k == "dbcv" and DBCV_AVAILABLE)
        ]
        for sort_key, reverse in active:
            _print_table(res, sort_key, args.top, reverse)
            _print_json_snippet(res, sort_key, args.top, reverse)

    # ─────────────────────────────────────────────────────────────────────
    # TaGra / Gower
    # ─────────────────────────────────────────────────────────────────────
    if "gower" in args.methods:
        if X_raw is None or gower_ftypes is None:
            print("\n[Gower]  SKIPPED — no X_raw or gower_ftypes in bundle/meta.")
        else:
            print("\n[Alg. 1] Gower threshold estimation (per metric) …")
            t_opts_gower: dict[str, float] = {}
            D_gower:      dict[str, np.ndarray] = {}

            for metric in args.gower_metrics:
                t_opt_g, _, D_g = estimate_threshold_gower(
                    X_raw, gower_ftypes, metric=metric,
                    k=args.k, p=args.strata, m=args.m, random_state=args.seed,
                )
                t_opts_gower[metric] = t_opt_g
                D_gower[metric]      = D_g
                print(f"         ✓ t_opt (Gower/{metric}) = {t_opt_g:.4f}")

            lo_g = min(t_opts_gower.values()) * args.t_scale[0]
            hi_g = max(t_opts_gower.values()) * args.t_scale[1]
            print(f"\n[Gower]  Grid search over {args.gower_metrics}, "
                  f"t ∈ [~{lo_g:.4f}, ~{hi_g:.4f}], "
                  f"df ∈ [{args.df_min}, {args.df_max}] …")
            res = grid_search_gower(
                X_raw, X, gower_ftypes, args.gower_metrics,
                t_opts_gower, D_gower,
                tuple(args.t_scale), args.n_t,
                args.df_min, args.df_max, verbose=True,
            )
            print(f"         {len(res)} valid configurations.")
            all_results.extend(res)

            active = [
                (k, rev) for k, rev, always in _METRICS
                if always or (k == "dbcv" and DBCV_AVAILABLE)
            ]
            for sort_key, reverse in active:
                _print_table(res, sort_key, args.top, reverse)
                _print_json_snippet(res, sort_key, args.top, reverse)

    # ─────────────────────────────────────────────────────────────────────
    # Combined table across all methods
    # ─────────────────────────────────────────────────────────────────────
    if len(args.methods) > 1 and all_results:
        active = [
            (k, rev) for k, rev, always in _METRICS
            if always or (k == "dbcv" and DBCV_AVAILABLE)
        ]

        seen: set[tuple] = set()
        combined: list[dict] = []
        for sort_key, reverse in active:
            for r in _ranked(all_results, sort_key, reverse, args.top):
                key = (r["method"],) + tuple(
                    r[k] for k in ("eps", "ms", "t", "df", "k", "metric") if k in r
                )
                if key not in seen:
                    seen.add(key)
                    combined.append(r)

        # Sort combined by silhouette descending for readability
        combined.sort(key=lambda r: (r.get("silhouette") or -999), reverse=True)

        print(f"\n{'='*82}")
        print(f"  COMBINED TOP  (union of top-{args.top} across all metrics and methods)")
        print(f"{'='*82}")
        print(_HDR); print(_SEP)
        for r in combined:
            _print_row(r)

        print(f"\n{'='*82}")
        print("  COMBINED  curated_configs  (paste into JSON config)")
        print(f"{'='*82}")

        # dbscan_refs (DBSCAN only)
        dbscan_combined = [r for r in combined if r["method"] == "dbscan"]
        if dbscan_combined:
            print('  "dbscan_refs": [')
            for r in dbscan_combined:
                print(f"    {_dbscan_ref_entry(r)},")
            print("  ]")

        print('  "curated_configs": [')
        for r in combined:
            print(f"    {_json_entry(r)},")
            if r["method"] == "dbscan":
                print(f'    {{"method": "dbscan_constr", "eps": {r["eps"]}, "ms": {r["ms"]}}},')

        print("  ]")

        print(f"\n  Total valid configurations found: {len(all_results)}")


if __name__ == "__main__":
    main()
