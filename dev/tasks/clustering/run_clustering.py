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
    cd dev/tasks/clustering
    python3 run_clustering.py --bundle bundles/hcv_bundle.npz --config config_hcv.json
    python3 run_clustering.py --bundle bundles/cleveland_bundle.npz --config config_cleveland.json

The --meta path defaults to <bundle_stem>_meta.json in the same directory.
"""

import os, sys, json, argparse
import numpy as np
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

def record(name, X, lbl, true_lbl, ref_lbl=None):
    n_cl = len(set(lbl) - {-1})
    n_ns = int(np.sum(lbl == -1))
    return {
        "name":       name,
        "n_clusters": n_cl,
        "noise_pct":  100 * n_ns / len(lbl),
        "dbcv":       dbcv(X, lbl),
        "sil":        sil(X, lbl),
        "vm_true":    vm(true_lbl, lbl),
        "vm_ref":     vm(ref_lbl, lbl) if ref_lbl is not None else None,
        "labels":     lbl,
    }

def fmt(v, w=7):
    return f"{v:{w}.4f}" if v is not None else " " * (w - 3) + "N/A"

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
    for r in rows:
        print_row(r, show_vm_ref)

def top3_summary(valid, fh=None):
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


# ── Curated comparison ────────────────────────────────────────────────────────

def run_curated(cfg, X, X_raw, gower_ftypes, true_labels, ref_lbl_A, n):
    """Execute all entries in cfg['curated_configs'] and return result records."""
    rows = []
    for item in cfg.get("curated_configs", []):
        method = item["method"]
        p = {k: v for k, v in item.items() if k != "method"}

        if method == "dbscan":
            lbl  = DBSCAN(eps=p["eps"], min_samples=p["ms"]).fit_predict(X)
            name = f"DBSCAN (eps={p['eps']}, ms={p['ms']})"
            r    = record(name, X, lbl, true_labels)

        elif method == "dbscan_constr":
            lbl  = dbscan_constructor_labels(X, p["eps"], p["ms"])
            name = f"TaGra/DBSCANConstr (eps={p['eps']}, ms={p['ms']})"
            r    = record(name, X, lbl, true_labels, ref_lbl=ref_lbl_A)

        elif method == "hdbscan":
            if not HDBSCAN_AVAILABLE:
                continue
            lbl  = hdbscan_lib.HDBSCAN(min_cluster_size=p["mcs"]).fit_predict(X)
            name = f"HDBSCAN (mcs={p['mcs']})"
            r    = record(name, X, lbl, true_labels)

        elif method == "gower":
            G = make_graph(n)
            GowerDistanceConstructor(
                distance_threshold=p["t"], feature_types=gower_ftypes,
                continuous_metric=p["metric"], verbose=False,
            ).construct(G, X_raw)
            lbl  = graph_cluster(G, n, p["df"])
            name = f"TaGra/Gower ({p['metric']},t={p['t']},df={p['df']})"
            r    = record(name, X, lbl, true_labels, ref_lbl=ref_lbl_A)

        elif method == "dist":
            G = make_graph(n)
            DistanceThresholdConstructor(
                distance_threshold=p["t"], verbose=False
            ).construct(G, X)
            lbl  = graph_cluster(G, n, p["df"])
            name = f"TaGra/Dist (t={p['t']},df={p['df']})"
            r    = record(name, X, lbl, true_labels, ref_lbl=ref_lbl_A)

        elif method == "sim":
            G = make_graph(n)
            SimilarityThresholdConstructor(
                similarity_threshold=p["t"], verbose=False
            ).construct(G, X)
            lbl  = graph_cluster(G, n, p["df"])
            name = f"TaGra/Sim (t={p['t']},df={p['df']})"
            r    = record(name, X, lbl, true_labels, ref_lbl=ref_lbl_A)

        elif method == "knn":
            G = make_graph(n)
            KNNConstructor(k=p["k"], verbose=False).construct(G, X)
            lbl  = graph_cluster(G, n, p["df"])
            name = f"TaGra/KNN (k={p['k']},df={p['df']})"
            r    = record(name, X, lbl, true_labels, ref_lbl=ref_lbl_A)

        else:
            continue

        rows.append(r)
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main(bundle_path, meta_path, cfg):
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
        os.path.dirname(__file__), f"clustering_results_{dataset_name.lower()}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Resolve quality criteria once — used in sweep filters and summaries
    qc = {**QUALITY_CRITERIA, **cfg.get("quality_criteria", {})}

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
        lbl = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
        dbscan_refs[(eps, ms)] = lbl
        r = record(f"DBSCAN (eps={eps}, ms={ms})", X, lbl, true_labels)
        dbscan_ref_rows.append(r)
        all_results.append(r)
    print_section("DBSCAN references", dbscan_ref_rows)

    # ── TaGra DBSCANConstructor equivalence ───────────────────────────────
    eq_rows = []
    for item in dbscan_ref_items:
        eps, ms = item["eps"], item["ms"]
        lbl_eq = dbscan_constructor_labels(X, eps, ms)
        r = record(f"TaGra/DBSCANConstr (eps={eps}, ms={ms})",
                   X, lbl_eq, true_labels, ref_lbl=dbscan_refs[(eps, ms)])
        eq_rows.append(r)
        all_results.append(r)
    print_section("TaGra DBSCANConstructor equivalence", eq_rows, show_vm_ref=True)

    # Ref A label for all "Vm(ref)" columns = first DBSCAN reference
    ref_lbl_A = dbscan_ref_rows[0]["labels"] if dbscan_ref_rows else np.zeros(n, dtype=int)
    ref_A_desc = (f"eps={dbscan_ref_items[0]['eps']}, ms={dbscan_ref_items[0]['ms']}"
                  if dbscan_ref_items else "N/A")

    # ── HDBSCAN ───────────────────────────────────────────────────────────
    hdbscan_rows = []
    if HDBSCAN_AVAILABLE:
        for mcs in cfg.get("hdbscan_mcs", []):
            lbl = hdbscan_lib.HDBSCAN(min_cluster_size=mcs).fit_predict(X)
            r   = record(f"HDBSCAN (mcs={mcs})", X, lbl, true_labels)
            hdbscan_rows.append(r)
            all_results.append(r)
    print_section("HDBSCAN", hdbscan_rows)

    # ── Exhaustive grid (optional) ────────────────────────────────────────
    if cfg.get("exhaustive_grid", False):

        # DBSCAN sweep (informational — refs are pinned, not auto-selected)
        sweep_cfg = cfg.get("dbscan_sweep", {})
        eps_grid  = sweep_cfg.get("eps_grid", [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5])
        ms_grid   = sweep_cfg.get("ms_grid",  [3,5,7,10,13,15,20,25,30])

        print(f"\n── DBSCAN sweep (eps × ms grid — quality-filtered) ──")
        print(f"  {'eps':>5} {'ms':>4} {'Cl':>4} {'Noise%':>7} {'DBCV':>8} {'Sil':>8} {'Vm':>8}")
        print("  " + "-" * 55)
        for eps in eps_grid:
            for ms in ms_grid:
                lbl = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
                r   = record(f"DBSCAN (eps={eps}, ms={ms})", X, lbl, true_labels)
                if not filter_results([r], criteria=qc):
                    continue
                ds = f"{r['dbcv']:.4f}" if r['dbcv'] is not None else "     N/A"
                ss = f"{r['sil']:.4f}"  if r['sil']  is not None else "     N/A"
                print(f"  {eps:>5.1f} {ms:>4}  {r['n_clusters']:>4}  {r['noise_pct']:>6.1f}%"
                      f"  {ds:>8}  {ss:>8}  {r['vm_true']:>8.4f}")

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
            if 2 <= len(set(lbl) - {-1}) <= 8:
                r = record(f"TaGra/Gower ({metric},t={thresh},df={df_val})",
                           X, lbl, true_labels, ref_lbl=ref_lbl_A)
                gower_rows.append(r)
                all_results.append(r)
        best = sorted(filter_results(gower_rows, criteria=qc),
                      key=lambda r: (-r["vm_true"], r["noise_pct"]))[:20]
        print_section("TaGra/Gower  (top 20 by Vm, quality-filtered)", best, show_vm_ref=True)

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
                if 2 <= len(set(lbl) - {-1}) <= 8:
                    r = record(f"TaGra/Sim (t={thresh},df={df_val})",
                               X, lbl, true_labels, ref_lbl=ref_lbl_A)
                    sim_rows.append(r)
                    all_results.append(r)
        best = sorted(filter_results(sim_rows, criteria=qc),
                      key=lambda r: (-r["vm_true"], r["noise_pct"]))[:20]
        print_section("TaGra/Similarity  (top 20 by Vm, quality-filtered)", best, show_vm_ref=True)

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
                if 2 <= len(set(lbl) - {-1}) <= 8:
                    r = record(f"TaGra/KNN (k={k},df={df_val})",
                               X, lbl, true_labels, ref_lbl=ref_lbl_A)
                    knn_rows.append(r)
                    all_results.append(r)
        best = sorted(filter_results(knn_rows, criteria=qc),
                      key=lambda r: (-r["vm_true"], r["noise_pct"]))[:20]
        print_section("TaGra/KNN  (top 20 by Vm, quality-filtered)", best, show_vm_ref=True)

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
                if 2 <= len(set(lbl) - {-1}) <= 8:
                    r = record(f"TaGra/Dist (t={thresh},df={df_val})",
                               X, lbl, true_labels, ref_lbl=ref_lbl_A)
                    dist_rows.append(r)
                    all_results.append(r)
        best = sorted(filter_results(dist_rows, criteria=qc),
                      key=lambda r: (-r["vm_true"], r["noise_pct"]))[:20]
        print_section("TaGra/Distance  (top 20 by Vm, quality-filtered)", best, show_vm_ref=True)

    # ── Curated comparison (always) ───────────────────────────────────────
    curated_rows  = run_curated(cfg, X, X_raw, gower_ftypes, true_labels, ref_lbl_A, n)
    curated_title = (f"CURATED COMPARISON\n"
                     f"  Vm(ref) = Vm vs first DBSCAN ref ({ref_A_desc})")
    print_section(curated_title, curated_rows, show_vm_ref=True)

    # ── Summaries ──────────────────────────────────────────────────────────
    valid     = [r for r in all_results if r["n_clusters"] >= 2]
    valid_low = [r for r in valid if r["noise_pct"] <= 60]

    sum_a_title = "SUMMARY A – ≥2 clusters, noise ≤ 60%, sorted by Vm(true) ↓"
    sum_a_rows  = sorted(valid_low, key=lambda r: r["vm_true"], reverse=True)
    sum_b_title = "SUMMARY B – ≥2 clusters, noise ≤ 60%, sorted by DBCV ↓ then Noise% ↑"
    sum_b_rows  = sorted([r for r in valid_low if r["dbcv"] is not None],
                         key=lambda r: (-r["dbcv"], r["noise_pct"]))

    print(f"\n{'='*78}\n{sum_a_title}")
    print(HDR); print(SEP)
    for r in sum_a_rows:
        print_row(r, show_vm_ref=r.get("vm_ref") is not None)

    print(f"\n{'='*78}\n{sum_b_title}")
    print(HDR); print(SEP)
    for r in sum_b_rows:
        print_row(r, show_vm_ref=r.get("vm_ref") is not None)

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
            print_row(r, show_vm_ref=r.get("vm_ref") is not None)
    else:
        print("  (no results pass all criteria)")

    # ── Save report ────────────────────────────────────────────────────────
    report_path = os.path.join(output_dir, f"{dataset_name.lower()}_report.txt")

    def _write_section(fh, title, rows, show_vm_ref=False):
        fh.write("=" * 78 + "\n")
        fh.write(title + "\n")
        fh.write(HDR + "\n" + SEP + "\n")
        for r in rows:
            vr = fmt(r["vm_ref"]) if show_vm_ref and r.get("vm_ref") is not None else "       -"
            fh.write(
                f"  {r['name']:<52} {r['n_clusters']:>4} {r['noise_pct']:>6.1f}%"
                f" {fmt(r['dbcv']):>8} {fmt(r['sil']):>8}"
                f" {fmt(r['vm_true']):>9} {vr:>8}\n"
            )
        fh.write("\n")

    with open(report_path, "w") as f:
        f.write(f"{dataset_name} – Clustering Comparison\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"n={n}  features={X.shape[1]}  Gower types: {gower_ftypes}\n\n")

        _write_section(f, "ALL RESULTS", all_results, show_vm_ref=True)
        _write_section(f, sum_a_title, sum_a_rows,    show_vm_ref=True)
        _write_section(f, sum_b_title, sum_b_rows,    show_vm_ref=True)
        _write_section(f, curated_title, curated_rows, show_vm_ref=True)

        f.write("=" * 78 + "\n")
        f.write(qc_title + "\n")
        f.write(HDR + "\n" + SEP + "\n")
        if qc_rows:
            for r in sorted(qc_rows, key=lambda r: r["vm_true"], reverse=True):
                vr = fmt(r["vm_ref"]) if r.get("vm_ref") is not None else "       -"
                f.write(
                    f"  {r['name']:<52} {r['n_clusters']:>4} {r['noise_pct']:>6.1f}%"
                    f" {fmt(r['dbcv']):>8} {fmt(r['sil']):>8}"
                    f" {fmt(r['vm_true']):>9} {vr:>8}\n"
                )
        else:
            f.write("  (no results pass all criteria)\n")

        top3_summary(valid_low, fh=f)

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
    args = parser.parse_args()

    meta_path = args.meta or args.bundle.replace("_bundle.npz", "_meta.json")

    with open(args.config) as _f:
        _cfg = json.load(_f)

    main(args.bundle, meta_path, _cfg)
