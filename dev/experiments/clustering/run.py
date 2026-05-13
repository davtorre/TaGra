#!/usr/bin/env python
"""
Unified clustering comparison pipeline.

Loads a preprocessed bundle (.npz + _meta.json) produced by a preprocess_*.py
script, then:
  1. Runs DBSCAN references and DBSCANConstructor equivalence (pinned in config)
  2. Runs HDBSCAN (mcs values from config)
  3. If exhaustive_grid=true: full Gower / Similarity / KNN / Distance grid sweeps
  4. Runs a curated comparison table (configs from JSON, always)
  5. Prints Summary A (by Vm), Summary B (by DBCV), quality-filtered summary,
     and top-3 per metric
  6. Saves a plain-text report

Usage:
    cd dev/experiments/clustering
    python3 run.py --bundle preprocessing/bundles/hcv_bundle.npz --config config/hcv.json
    python3 run.py --bundle preprocessing/bundles/cleveland_bundle.npz --config config/cleveland.json

The --meta path defaults to <bundle_stem>_meta.json in the same directory.
"""

import os, sys, json, argparse, warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, module="hdbscan")
import networkx as nx
from datetime import datetime
from itertools import product
from sklearn.cluster import DBSCAN
from sklearn.metrics import v_measure_score, silhouette_score

REPO_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
PREPROCESSING   = os.path.abspath(os.path.join(os.path.dirname(__file__), "preprocessing"))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, PREPROCESSING)

from tagra.construction.distance import DistanceThresholdConstructor
from tagra.construction.knn import KNNConstructor
from tagra.construction.similarity import SimilarityThresholdConstructor
from tagra.construction.dbscan import DBSCANGraphConstructor
from tagra.construction.gower import GowerDistanceConstructor
from tagra.utils import track_energy

from preprocess_typed import QUALITY_CRITERIA, filter_results

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


# ── Metrics ───────────────────────────────────────────────────────────────────

def dbcv(X, lbl):
    if not DBCV_AVAILABLE or len(set(lbl) - {-1}) < 2: return None
    mask = lbl != -1
    if mask.sum() < 2: return None
    try: return float(dbcv_score(X[mask], lbl[mask]))
    except: return None

def sil(X, lbl):
    if len(set(lbl) - {-1}) < 2: return None
    mask = lbl != -1
    if mask.sum() < 2: return None
    try: return float(silhouette_score(X[mask], lbl[mask]))
    except: return None

def vm(true, pred):
    return float(v_measure_score(true, pred))


# ── Graph helpers ─────────────────────────────────────────────────────────────

def make_graph(n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    return G

def graph_cluster(G, n, df_val):
    low = [v for v, d in G.degree() if d < df_val]
    Gf  = G.copy()
    Gf.remove_nodes_from(low)
    lbl = np.full(n, -1)
    cid = 0
    for comp in nx.connected_components(Gf):
        for v in comp: lbl[v] = cid
        cid += 1
    return lbl

def dbscan_constructor_labels(X, eps, min_samples):
    n = len(X)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    DBSCANGraphConstructor(eps=eps, min_samples=min_samples, verbose=False).construct(G, X)
    lbl = np.full(n, -1)
    cid = 0
    for comp in nx.weakly_connected_components(G):
        if len(comp) == 1:
            node = next(iter(comp))
            if G.in_degree(node) == 0 and G.out_degree(node) == 0:
                continue
        for node in comp: lbl[node] = cid
        cid += 1
    return lbl


# ── Record / display ──────────────────────────────────────────────────────────

def min_cluster_pct(lbl):
    """Smallest cluster as % of non-noise points. None if fewer than 2 clusters."""
    ids = [c for c in set(lbl) if c != -1]
    if len(ids) < 2:
        return None
    n_non_noise = np.sum(lbl != -1)
    if n_non_noise == 0:
        return None
    return float(min((lbl == c).sum() for c in ids) / n_non_noise * 100)

def record(name, X, lbl, true_lbl, ref_lbl=None, energy_wh=None, emissions_grams=None):
    n_cl = len(set(lbl) - {-1})
    n_ns = int(np.sum(lbl == -1))
    return {
        "name":             name,
        "n_clusters":       n_cl,
        "noise_pct":        100 * n_ns / len(lbl),
        "min_cls_pct":      min_cluster_pct(lbl),
        "dbcv":             dbcv(X, lbl),
        "sil":              sil(X, lbl),
        "vm_true":          vm(true_lbl, lbl),
        "vm_ref":           vm(ref_lbl, lbl) if ref_lbl is not None else None,
        "energy_wh":        energy_wh,
        "emissions_grams":  emissions_grams,
        "labels":           lbl,
    }

def fmt(v, w=7):
    return f"{v:{w}.4f}" if v is not None else " " * (w - 3) + "N/A"

HDR = (f"{'Method':<52} {'Cl':>4} {'Noise%':>7} {'MinCls%':>8} "
       f"{'DBCV':>8} {'Sil':>8} {'Vm(true)':>9} {'Vm(ref)':>8} "
       f"{'Energy(Wh)':>12} {'CO2eq(g)':>12}")
SEP = "-" * 141

def _fmt_energy(v, w=12, skip_codecarbon=False):
    if v is not None:
        return f"{v:{w}.6f}"
    label = "[skipped]" if skip_codecarbon else "N/A"
    return label.rjust(w)

def print_row(r, show_vm_ref=False, skip_codecarbon=False):
    vr  = fmt(r["vm_ref"]) if show_vm_ref and r.get("vm_ref") is not None else "       -"
    mc  = f"{r['min_cls_pct']:>6.1f}%" if r.get("min_cls_pct") is not None else "     N/A"
    print(f"  {r['name']:<52} {r['n_clusters']:>4} {r['noise_pct']:>6.1f}%"
          f" {mc:>8} {fmt(r['dbcv']):>8} {fmt(r['sil']):>8}"
          f" {fmt(r['vm_true']):>9} {vr:>8}"
          f" {_fmt_energy(r.get('energy_wh'), skip_codecarbon=skip_codecarbon):>12}"
          f" {_fmt_energy(r.get('emissions_grams'), skip_codecarbon=skip_codecarbon):>12}")

def print_section(title, rows, show_vm_ref=False, skip_codecarbon=False):
    print(f"\n{'─'*78}\n{title}")
    print(HDR); print(SEP)
    for r in rows:
        print_row(r, show_vm_ref, skip_codecarbon=skip_codecarbon)

def top3_summary(valid, fh=None, skip_codecarbon=False):
    metrics = [
        ("Lowest Noise%", lambda r: r["noise_pct"],                                False),
        ("Highest DBCV",  lambda r: r["dbcv"]    if r["dbcv"] is not None else -999, True),
        ("Highest Sil",   lambda r: r["sil"]     if r["sil"]  is not None else -999, True),
        ("Highest Vm",    lambda r: r["vm_true"],                                   True),
    ]
    lines = [f"\n{'='*78}", "TOP 3 PER METRIC  (≥2 clusters, noise ≤ 60%)", HDR, SEP]
    for label, key_fn, rev in metrics:
        lines.append(f"\n  ── {label} ──")
        for r in sorted(valid, key=key_fn, reverse=rev)[:3]:
            vr = fmt(r["vm_ref"]) if r.get("vm_ref") is not None else "       -"
            mc = f"{r['min_cls_pct']:>6.1f}%" if r.get("min_cls_pct") is not None else "     N/A"
            lines.append(
                f"  {r['name']:<52} {r['n_clusters']:>4} {r['noise_pct']:>6.1f}%"
                f" {mc:>8} {fmt(r['dbcv']):>8} {fmt(r['sil']):>8}"
                f" {fmt(r['vm_true']):>9} {vr:>8}"
                f" {_fmt_energy(r.get('energy_wh'), skip_codecarbon=skip_codecarbon):>12}"
                f" {_fmt_energy(r.get('emissions_grams'), skip_codecarbon=skip_codecarbon):>12}"
            )
    for line in lines:
        print(line)
    if fh:
        for line in lines:
            fh.write(line + "\n")


# ── Curated comparison ────────────────────────────────────────────────────────

def run_curated(cfg, X, X_raw, gower_ftypes, true_labels, ref_lbl_A, n, skip_codecarbon=False):
    """Execute all entries in cfg['curated_configs'] and return result records."""
    rows = []
    for item in cfg.get("curated_configs", []):
        method = item["method"]
        p = {k: v for k, v in item.items() if k != "method"}

        if method == "dbscan":
            lbl, ew, eg = track_energy(DBSCAN(eps=p["eps"], min_samples=p["ms"]).fit_predict, X, skip=skip_codecarbon)
            name = f"DBSCAN (eps={p['eps']}, ms={p['ms']})"
            r    = record(name, X, lbl, true_labels, energy_wh=ew, emissions_grams=eg)

        elif method == "dbscan_constr":
            lbl, ew, eg = track_energy(dbscan_constructor_labels, X, p["eps"], p["ms"], skip=skip_codecarbon)
            name = f"TaGra/DBSCANConstr (eps={p['eps']}, ms={p['ms']})"
            r    = record(name, X, lbl, true_labels, ref_lbl=ref_lbl_A, energy_wh=ew, emissions_grams=eg)

        elif method == "hdbscan":
            if not HDBSCAN_AVAILABLE:
                continue
            lbl, ew, eg = track_energy(hdbscan_lib.HDBSCAN(min_cluster_size=p["mcs"]).fit_predict, X, skip=skip_codecarbon)
            name = f"HDBSCAN (mcs={p['mcs']})"
            r    = record(name, X, lbl, true_labels, energy_wh=ew, emissions_grams=eg)

        elif method == "gower":
            def _gower(p=p):
                G = make_graph(n)
                GowerDistanceConstructor(
                    distance_threshold=p["t"], feature_types=gower_ftypes,
                    continuous_metric=p["metric"], verbose=False,
                ).construct(G, X_raw)
                return graph_cluster(G, n, p["df"])
            lbl, ew, eg = track_energy(_gower, skip=skip_codecarbon)
            name = f"TaGra/Gower ({p['metric']},t={p['t']},df={p['df']})"
            r    = record(name, X, lbl, true_labels, ref_lbl=ref_lbl_A, energy_wh=ew, emissions_grams=eg)

        elif method == "dist":
            def _dist(p=p):
                G = make_graph(n)
                DistanceThresholdConstructor(distance_threshold=p["t"], verbose=False).construct(G, X)
                return graph_cluster(G, n, p["df"])
            lbl, ew, eg = track_energy(_dist, skip=skip_codecarbon)
            name = f"TaGra/Dist (t={p['t']},df={p['df']})"
            r    = record(name, X, lbl, true_labels, ref_lbl=ref_lbl_A, energy_wh=ew, emissions_grams=eg)

        elif method == "sim":
            def _sim(p=p):
                G = make_graph(n)
                SimilarityThresholdConstructor(similarity_threshold=p["t"], verbose=False).construct(G, X)
                return graph_cluster(G, n, p["df"])
            lbl, ew, eg = track_energy(_sim, skip=skip_codecarbon)
            name = f"TaGra/Sim (t={p['t']},df={p['df']})"
            r    = record(name, X, lbl, true_labels, ref_lbl=ref_lbl_A, energy_wh=ew, emissions_grams=eg)

        elif method == "knn":
            def _knn(p=p):
                G = make_graph(n)
                KNNConstructor(k=p["k"], verbose=False).construct(G, X)
                return graph_cluster(G, n, p["df"])
            lbl, ew, eg = track_energy(_knn, skip=skip_codecarbon)
            name = f"TaGra/KNN (k={p['k']},df={p['df']})"
            r    = record(name, X, lbl, true_labels, ref_lbl=ref_lbl_A, energy_wh=ew, emissions_grams=eg)

        else:
            continue

        rows.append(r)
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main(bundle_path, meta_path, cfg, show_top3=False, skip_codecarbon=False):
    # ── Load bundle ───────────────────────────────────────────────────────
    bundle      = np.load(bundle_path)
    X           = bundle["X"]
    X_raw       = bundle["X_raw"]
    true_labels = bundle["true_labels"]
    n           = len(X)

    with open(meta_path) as f:
        meta = json.load(f)
    dataset_name = meta["dataset_name"]
    gower_ftypes = meta["gower_ftypes"]
    feat_names   = meta["feat_names"]

    output_dir = os.path.join(
        os.path.dirname(__file__), "results", "reports"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Resolve quality criteria once — used in sweep filters and summaries
    qc = {**QUALITY_CRITERIA, **cfg.get("quality_criteria", {})}

    # Max clusters to admit into all_results (configurable, default 20)
    max_clusters = cfg.get("max_clusters", 20)

    print("=" * 78)
    print(f"{dataset_name} – Clustering Comparison Pipeline")
    print(f"Generated:     {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"n={n}  features={X.shape[1]}  Gower types: {gower_ftypes}")
    print(f"Class balance: {meta.get('class_balance', {})}")
    print("=" * 78)

    all_results = []

    # ── DBSCAN references (pinned) ────────────────────────────────────────
    dbscan_ref_items = cfg.get("dbscan_refs", [])
    dbscan_refs      = {}
    dbscan_ref_rows  = []
    for item in dbscan_ref_items:
        eps, ms = item["eps"], item["ms"]
        lbl, ew, eg = track_energy(DBSCAN(eps=eps, min_samples=ms).fit_predict, X, skip=skip_codecarbon)
        dbscan_refs[(eps, ms)] = lbl
        r = record(f"DBSCAN (eps={eps}, ms={ms})", X, lbl, true_labels, energy_wh=ew, emissions_grams=eg)
        dbscan_ref_rows.append(r)
        all_results.append(r)
    print_section("DBSCAN references", dbscan_ref_rows, skip_codecarbon=skip_codecarbon)

    # ── TaGra DBSCANConstructor equivalence ───────────────────────────────
    eq_rows = []
    for item in dbscan_ref_items:
        eps, ms = item["eps"], item["ms"]
        lbl_eq, ew, eg = track_energy(dbscan_constructor_labels, X, eps, ms, skip=skip_codecarbon)
        r = record(f"TaGra/DBSCANConstr (eps={eps}, ms={ms})",
                   X, lbl_eq, true_labels, ref_lbl=dbscan_refs[(eps, ms)], energy_wh=ew, emissions_grams=eg)
        eq_rows.append(r)
        all_results.append(r)
    print_section("TaGra DBSCANConstructor equivalence", eq_rows, show_vm_ref=True, skip_codecarbon=skip_codecarbon)

    # Ref A label for all "Vm(ref)" columns = first DBSCAN reference
    ref_lbl_A = dbscan_ref_rows[0]["labels"] if dbscan_ref_rows else np.zeros(n, dtype=int)
    ref_A_desc = (f"eps={dbscan_ref_items[0]['eps']}, ms={dbscan_ref_items[0]['ms']}"
                  if dbscan_ref_items else "N/A")

    # ── HDBSCAN ───────────────────────────────────────────────────────────
    hdbscan_rows = []
    if HDBSCAN_AVAILABLE:
        for mcs in cfg.get("hdbscan_mcs", []):
            lbl, ew, eg = track_energy(hdbscan_lib.HDBSCAN(min_cluster_size=mcs).fit_predict, X, skip=skip_codecarbon)
            r   = record(f"HDBSCAN (mcs={mcs})", X, lbl, true_labels, energy_wh=ew, emissions_grams=eg)
            hdbscan_rows.append(r)
            all_results.append(r)
    print_section("HDBSCAN", hdbscan_rows, skip_codecarbon=skip_codecarbon)

    # ── Exhaustive grid (optional) ────────────────────────────────────────
    if cfg.get("exhaustive_grid", False):

        # DBSCAN sweep — results fed into all_results (same as TaGra grids)
        sweep_cfg = cfg.get("dbscan_sweep", {})
        eps_grid  = sweep_cfg.get("eps_grid", [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5])
        ms_grid   = sweep_cfg.get("ms_grid",  [3,5,7,10,13,15,20,25,30])

        dbscan_rows = []
        for eps in eps_grid:
            for ms in ms_grid:
                lbl = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
                if 2 <= len(set(lbl) - {-1}) <= max_clusters:
                    r = record(f"DBSCAN (eps={eps}, ms={ms})", X, lbl, true_labels,
                               ref_lbl=ref_lbl_A)
                    dbscan_rows.append(r)
                    all_results.append(r)
        best_dbscan = sorted(filter_results(dbscan_rows, criteria=qc),
                             key=lambda r: (-r["vm_true"], r["noise_pct"]))[:20]
        print_section("DBSCAN sweep  (top 20 by Vm, quality-filtered)", best_dbscan,
                      show_vm_ref=True, skip_codecarbon=skip_codecarbon)

        # Gower grid
        print("\n── TaGra/Gower exhaustive grid ──")
        gower_rows  = []
        gcfg        = cfg.get("gower_grid", {})
        g_metrics   = gcfg.get("metrics",    ["range", "quadratic", "gaussian", "laplacian"])
        g_thresh    = gcfg.get("thresholds", [0.03,0.05,0.07,0.08,0.09,0.10,0.11,0.12,
                                              0.13,0.14,0.15,0.17,0.18,0.20,0.22,0.25,
                                              0.28,0.30,0.35,0.40])
        g_df        = gcfg.get("df_range",   [1, 15])

        for metric, thresh, df_val in product(g_metrics, g_thresh, range(g_df[0], g_df[1]+1)):
            G = make_graph(n)
            GowerDistanceConstructor(
                distance_threshold=thresh, feature_types=gower_ftypes,
                continuous_metric=metric, verbose=False,
            ).construct(G, X_raw)
            lbl = graph_cluster(G, n, df_val)
            if 2 <= len(set(lbl) - {-1}) <= max_clusters:
                r = record(f"TaGra/Gower ({metric},t={thresh},df={df_val})",
                           X, lbl, true_labels, ref_lbl=ref_lbl_A)
                gower_rows.append(r)
                all_results.append(r)
        best = sorted(filter_results(gower_rows, criteria=qc),
                      key=lambda r: (-r["vm_true"], r["noise_pct"]))[:20]
        print_section("TaGra/Gower  (top 20 by Vm, quality-filtered)", best, show_vm_ref=True, skip_codecarbon=skip_codecarbon)

        # Similarity grid
        print("\n── TaGra/Similarity exhaustive grid ──")
        sim_rows = []
        scfg     = cfg.get("sim_grid", {})
        s_thresh = scfg.get("thresholds", [round(0.50 + i*0.01, 2) for i in range(46)])
        s_df     = scfg.get("df_range",   [1, 20])

        for thresh in s_thresh:
            G = make_graph(n)
            SimilarityThresholdConstructor(
                similarity_threshold=thresh, verbose=False
            ).construct(G, X)
            for df_val in range(s_df[0], s_df[1]+1):
                lbl = graph_cluster(G, n, df_val)
                if 2 <= len(set(lbl) - {-1}) <= max_clusters:
                    r = record(f"TaGra/Sim (t={thresh},df={df_val})",
                               X, lbl, true_labels, ref_lbl=ref_lbl_A)
                    sim_rows.append(r)
                    all_results.append(r)
        best = sorted(filter_results(sim_rows, criteria=qc),
                      key=lambda r: (-r["vm_true"], r["noise_pct"]))[:20]
        print_section("TaGra/Similarity  (top 20 by Vm, quality-filtered)", best, show_vm_ref=True, skip_codecarbon=skip_codecarbon)

        # KNN grid
        print("\n── TaGra/KNN exhaustive grid ──")
        knn_rows = []
        kcfg     = cfg.get("knn_grid", {})
        k_range  = kcfg.get("k_range",  [3, 30])
        k_df     = kcfg.get("df_range", [1, 20])

        for k in range(k_range[0], k_range[1]+1):
            G = make_graph(n)
            KNNConstructor(k=k, verbose=False).construct(G, X)
            for df_val in range(k_df[0], k_df[1]+1):
                lbl = graph_cluster(G, n, df_val)
                if 2 <= len(set(lbl) - {-1}) <= max_clusters:
                    r = record(f"TaGra/KNN (k={k},df={df_val})",
                               X, lbl, true_labels, ref_lbl=ref_lbl_A)
                    knn_rows.append(r)
                    all_results.append(r)
        best = sorted(filter_results(knn_rows, criteria=qc),
                      key=lambda r: (-r["vm_true"], r["noise_pct"]))[:20]
        print_section("TaGra/KNN  (top 20 by Vm, quality-filtered)", best, show_vm_ref=True, skip_codecarbon=skip_codecarbon)

        # Distance-threshold grid
        print("\n── TaGra/Distance-threshold exhaustive grid ──")
        dist_rows = []
        dcfg      = cfg.get("dist_grid", {})
        d_thresh  = dcfg.get("thresholds", [0.3,0.4,0.5,0.6,0.7,0.8,0.9,
                                            1.0,1.1,1.2,1.3,1.4,1.5])
        d_df      = dcfg.get("df_range",   [1, 25])

        for thresh in d_thresh:
            G = make_graph(n)
            DistanceThresholdConstructor(
                distance_threshold=thresh, verbose=False
            ).construct(G, X)
            for df_val in range(d_df[0], d_df[1]+1):
                lbl = graph_cluster(G, n, df_val)
                if 2 <= len(set(lbl) - {-1}) <= max_clusters:
                    r = record(f"TaGra/Dist (t={thresh},df={df_val})",
                               X, lbl, true_labels, ref_lbl=ref_lbl_A)
                    dist_rows.append(r)
                    all_results.append(r)
        best = sorted(filter_results(dist_rows, criteria=qc),
                      key=lambda r: (-r["vm_true"], r["noise_pct"]))[:20]
        print_section("TaGra/Distance  (top 20 by Vm, quality-filtered)", best, show_vm_ref=True, skip_codecarbon=skip_codecarbon)

    # ── Curated comparison (always) ───────────────────────────────────────
    curated_rows  = run_curated(cfg, X, X_raw, gower_ftypes, true_labels, ref_lbl_A, n, skip_codecarbon=skip_codecarbon)
    curated_title = (f"CURATED COMPARISON\n"
                     f"  Vm(ref) = Vm vs first DBSCAN ref ({ref_A_desc})")
    print_section(curated_title, curated_rows, show_vm_ref=True, skip_codecarbon=skip_codecarbon)
    seen_names = {r["name"] for r in all_results}
    for r in curated_rows:
        if r["name"] not in seen_names:
            all_results.append(r)
            seen_names.add(r["name"])

    # ── Summaries (config-driven) ─────────────────────────────────────────
    valid = [r for r in all_results if r["n_clusters"] >= 2]

    def _sort_key(fields):
        """Build a sort key from a field name or list of field names.
        First field is always descending (negated), remaining ascending.
        None values are sorted last by substituting -inf for descending fields."""
        if isinstance(fields, str):
            fields = [fields]
        def key(r):
            vals = []
            for i, f in enumerate(fields):
                v = r.get(f)
                if i == 0:   # descending: negate, put None last
                    vals.append(float("inf") if v is None else -v)
                else:        # ascending: put None last
                    vals.append(float("inf") if v is None else v)
            return vals
        return key

    summary_defs = cfg.get("summaries", [])
    computed_summaries = []   # kept for the report writer

    for s in summary_defs:
        name         = s.get("name", "")
        max_noise    = s.get("max_noise_pct", 100.0)
        sort_fields  = s.get("sort", "vm_true")
        if isinstance(sort_fields, str):
            sort_label = f"{sort_fields} ↓"
        else:
            sort_label = " ↓, ".join(sort_fields[:1]) + " ↓ then " + ", ".join(sort_fields[1:]) + " ↑"

        pool  = [r for r in valid if r["noise_pct"] <= max_noise]
        rows  = sorted(pool, key=_sort_key(sort_fields))
        title = (f"SUMMARY {name} – ≥2 clusters, noise ≤ {max_noise}%, "
                 f"sorted by {sort_label}")

        print(f"\n{'='*78}\n{title}")
        print(HDR); print(SEP)
        for r in rows:
            print_row(r, show_vm_ref=r.get("vm_ref") is not None, skip_codecarbon=skip_codecarbon)

        computed_summaries.append((title, rows))

    # valid_low still needed for top-3 (uses 60% as the display baseline)
    valid_low = [r for r in valid if r["noise_pct"] <= 60]

    # ── Quality-filtered ───────────────────────────────────────────────────
    qc_title = (
        f"QUALITY-FILTERED  "
        f"(DBCV≥{qc['min_dbcv']}, noise<{qc['max_noise_pct']}%, "
        f"min cluster≥{qc['min_cluster_size_pct']}% of non-noise, "
        f"clusters≥{qc['min_n_clusters']})"
    )
    qc_rows = filter_results(all_results, criteria=qc)
    print(f"\n{'='*78}\n{qc_title}")
    print(HDR); print(SEP)
    if qc_rows:
        for r in sorted(qc_rows, key=lambda r: r["vm_true"], reverse=True):
            print_row(r, show_vm_ref=r.get("vm_ref") is not None, skip_codecarbon=skip_codecarbon)
    else:
        print("  (no results pass all criteria)")

    # ── Save report ────────────────────────────────────────────────────────
    report_path = os.path.join(output_dir, f"{dataset_name.lower()}_report.txt")

    def _write_section(fh, title, rows, show_vm_ref=False):
        fh.write("=" * 141 + "\n")
        fh.write(title + "\n")
        fh.write(HDR + "\n" + SEP + "\n")
        for r in rows:
            vr = fmt(r["vm_ref"]) if show_vm_ref and r.get("vm_ref") is not None else "       -"
            mc = f"{r['min_cls_pct']:>6.1f}%" if r.get("min_cls_pct") is not None else "     N/A"
            fh.write(
                f"  {r['name']:<52} {r['n_clusters']:>4} {r['noise_pct']:>6.1f}%"
                f" {mc:>8} {fmt(r['dbcv']):>8} {fmt(r['sil']):>8}"
                f" {fmt(r['vm_true']):>9} {vr:>8}"
                f" {_fmt_energy(r.get('energy_wh'), skip_codecarbon=skip_codecarbon):>12}"
                f" {_fmt_energy(r.get('emissions_grams'), skip_codecarbon=skip_codecarbon):>12}\n"
            )
        fh.write("\n")

    with open(report_path, "w") as f:
        f.write(f"{dataset_name} – Clustering Comparison\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"n={n}  features={X.shape[1]}  Gower types: {gower_ftypes}\n\n")

        _write_section(f, "ALL RESULTS", all_results, show_vm_ref=True)
        for title, rows in computed_summaries:
            _write_section(f, title, rows, show_vm_ref=True)
        _write_section(f, curated_title, curated_rows, show_vm_ref=True)

        f.write("=" * 141 + "\n")
        f.write(qc_title + "\n")
        f.write(HDR + "\n" + SEP + "\n")
        if qc_rows:
            for r in sorted(qc_rows, key=lambda r: r["vm_true"], reverse=True):
                vr = fmt(r["vm_ref"]) if r.get("vm_ref") is not None else "       -"
                mc = f"{r['min_cls_pct']:>6.1f}%" if r.get("min_cls_pct") is not None else "     N/A"
                f.write(
                    f"  {r['name']:<52} {r['n_clusters']:>4} {r['noise_pct']:>6.1f}%"
                    f" {mc:>8} {fmt(r['dbcv']):>8} {fmt(r['sil']):>8}"
                    f" {fmt(r['vm_true']):>9} {vr:>8}"
                    f" {_fmt_energy(r.get('energy_wh'), skip_codecarbon=skip_codecarbon):>12}"
                f" {_fmt_energy(r.get('emissions_grams'), skip_codecarbon=skip_codecarbon):>12}\n"
                )
        else:
            f.write("  (no results pass all criteria)\n")

        if show_top3:
            top3_summary(valid_low, fh=f, skip_codecarbon=skip_codecarbon)

    if show_top3:
        top3_summary(valid_low, skip_codecarbon=skip_codecarbon)

    print(f"\nReport saved: {report_path}")
    print(f"Total configurations tested: {len(all_results)}")


if __name__ == "__main__":
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Unified clustering comparison pipeline")
    parser.add_argument("--bundle", required=True,
                        help="Path to bundle .npz (e.g. bundles/hcv_bundle.npz)")
    parser.add_argument("--meta",
                        help="Path to meta JSON (default: inferred from --bundle path)")
    parser.add_argument("--config", required=True,
                        help="Path to dataset JSON config file")
    parser.add_argument("--top", action="store_true",
                        help="Print top-3 results per metric at the end")
    parser.add_argument("--skip-codecarbon", action="store_true",
                        help="Skip CodeCarbon energy/emissions tracking (shows [skipped] in reports)")
    args = parser.parse_args()

    meta_path = args.meta or args.bundle.replace("_bundle.npz", "_meta.json")

    with open(args.config) as _f:
        _cfg = json.load(_f)

    main(args.bundle, meta_path, _cfg, show_top3=args.top, skip_codecarbon=args.skip_codecarbon)
