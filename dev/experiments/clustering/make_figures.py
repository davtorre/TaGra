#!/usr/bin/env python
"""
Generate static PNG figures for the TaGra paper.

Reads:
  dev/experiments/clustering/results/all_results.csv
  dev/experiments/clustering/results/outcome_tables.csv

Writes 6 PNGs to docs/paper/figs/ at 300 dpi.
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# ── Constants ─────────────────────────────────────────────────────────────────

REPO_ROOT   = "/Users/davide.torre/Research/Projects/tagra"
RESULTS_DIR = os.path.join(REPO_ROOT, "dev/experiments/clustering/results")
FIGS_DIR    = os.path.join(REPO_ROOT, "docs/paper/figs")
os.makedirs(FIGS_DIR, exist_ok=True)

METHOD_LABELS = {
    "dbscan":        "DBSCAN",
    "hdbscan":       "HDBSCAN",
    "dbscan_constr": "TaGra/DBSCANConstr",
    "dist":          "TaGra/Dist",
    "knn":           "TaGra/KNN",
    "sim":           "TaGra/Sim",
    "gower":         "TaGra/Gower",
}

DATASET_LABELS = {
    "cleveland":      "Cleveland",
    "hcv":            "HCV",
    "ckd":            "CKD",
    "diabetes_t1":    "DiabetesT1",
    "sepsis":         "Sepsis",
    "cardiac_arrest": "CardiacArrest",
    "depression_hf":  "DepressionHF",
    "neuroblastoma":  "Neuroblastoma",
}

METHOD_COLORS = {
    "dbscan":        "#4C72B0",
    "hdbscan":       "#DD8452",
    "dbscan_constr": "#55A868",
    "dist":          "#C44E52",
    "knn":           "#8172B2",
    "sim":           "#937860",
    "gower":         "#DA8BC3",
}

METHODS_COMPARE = ["dbscan", "hdbscan", "dist", "knn", "sim", "gower"]
DATASETS_ORDER  = list(DATASET_LABELS.keys())

# Quality criteria matching config_universal.json
QUALITY = dict(min_dbcv=0.2, max_noise_pct=30.0, min_cls_pct=5.0, min_n_clusters=2)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    df_all = pd.read_csv(os.path.join(RESULTS_DIR, "all_results.csv"))
    oc     = pd.read_csv(os.path.join(RESULTS_DIR, "outcome_tables.csv"))

    q  = QUALITY
    df = df_all[
        (df_all["dbcv"]        >= q["min_dbcv"])      &
        (df_all["noise_pct"]   <  q["max_noise_pct"]) &
        (df_all["min_cls_pct"] >= q["min_cls_pct"])   &
        (df_all["n_clusters"]  >= q["min_n_clusters"])
    ].copy()

    assert df["noise_pct"].max() < 30.0,  f"noise filter failed: max={df['noise_pct'].max()}"
    assert df["dbcv"].min()      >= 0.2,  f"dbcv filter failed:  min={df['dbcv'].min()}"

    return df_all, df, oc


def apply_quality(df):
    """Filter all_results rows to quality-passing configs."""
    q = QUALITY
    mask = (
        df["dbcv"].notna() &
        (df["dbcv"] >= q["min_dbcv"]) &
        (df["noise_pct"] < q["max_noise_pct"]) &
        (df["n_clusters"] >= q["min_n_clusters"]) &
        df["min_cls_pct"].notna() &
        (df["min_cls_pct"] >= q["min_cls_pct"])
    )
    return df[mask].copy()


def best_dbcv_matrix(df):
    """Return (methods × datasets) DataFrame of max DBCV. Expects quality-filtered df."""
    sub = df[df["method"].isin(METHODS_COMPARE)]
    pivot = (
        sub.groupby(["method", "dataset"])["dbcv"]
           .max()
           .unstack(level="dataset")
           .reindex(index=METHODS_COMPARE, columns=DATASETS_ORDER)
    )
    return pivot


# ── Figure 1: Pareto scatter ──────────────────────────────────────────────────

def fig_pareto_scatter(df_all):
    sub = df_all[df_all["method"].isin(METHODS_COMPARE) & df_all["dbcv"].notna()].copy()

    x_max = max(sub["noise_pct"].max() * 1.05, 35)
    y_min = min(sub["dbcv"].min() - 0.05, -0.1)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Shade the quality-passing region
    ax.axhspan(QUALITY["min_dbcv"], 1.05, xmin=0,
               xmax=QUALITY["max_noise_pct"] / x_max,
               color="#e8f4e8", zorder=0, label="Quality zone")

    for method in METHODS_COMPARE:
        m_df = sub[sub["method"] == method]
        ax.scatter(
            m_df["noise_pct"], m_df["dbcv"],
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            s=40, alpha=0.6, edgecolors="none",
        )

    ax.axvline(QUALITY["max_noise_pct"], color="#888888", linestyle="--",
               linewidth=1, label=f"noise = {QUALITY['max_noise_pct']:.0f}%")
    ax.axhline(QUALITY["min_dbcv"],      color="#888888", linestyle=":",
               linewidth=1, label=f"DBCV = {QUALITY['min_dbcv']}")

    ax.set_xlabel("Noise %")
    ax.set_ylabel("DBCV")
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, 1.02)
    ax.set_title("All configurations — noise% vs DBCV")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGS_DIR, "fig_pareto_scatter.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 2: Best DBCV heatmap ───────────────────────────────────────────────

def fig_best_dbcv_heatmap(df):
    pivot = best_dbcv_matrix(df)

    fig, ax = plt.subplots(figsize=(12, 4))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#cccccc")

    data = pivot.values.astype(float)
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0.2, vmax=1.0)

    ax.set_xticks(range(len(DATASETS_ORDER)))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS_ORDER], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(METHODS_COMPARE)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS_COMPARE], fontsize=9)

    for i in range(len(METHODS_COMPARE)):
        for j in range(len(DATASETS_ORDER)):
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="#999999")
            else:
                color = "white" if v < 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="DBCV", fraction=0.03, pad=0.02)
    ax.set_title("Best DBCV — method × dataset")
    fig.tight_layout()
    path = os.path.join(FIGS_DIR, "fig_best_dbcv_heatmap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 3: Equivalence check ───────────────────────────────────────────────

def fig_equivalence_check(df):
    sub = df[df["method"] == "dbscan_constr"].copy()
    if sub.empty:
        print("  [skip] no dbscan_constr results — Fig 3 will show empty bars")
    # Best config per dataset: highest vm_ref (DBSCANConstr vs DBSCAN labels)
    sub_valid = sub[sub["vm_ref"].notna() & (sub["n_clusters"] >= 1)]
    best = (
        sub_valid.sort_values("vm_ref", ascending=False)
                 .groupby("dataset", as_index=False)
                 .first()
    )

    vm_vals = []
    for ds in DATASETS_ORDER:
        row = best[best["dataset"] == ds]
        vm_vals.append(float(row["vm_ref"].iloc[0]) if len(row) else float("nan"))

    colors = [METHOD_COLORS["dbscan_constr"] if not math.isnan(v) else "#cccccc"
              for v in vm_vals]
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_heights = [v if not math.isnan(v) else 0.0 for v in vm_vals]
    ax.bar(range(len(DATASETS_ORDER)), bar_heights, color=colors)

    for i, (v, ds) in enumerate(zip(vm_vals, DATASETS_ORDER)):
        if math.isnan(v):
            ax.text(i, 0.02, "n/a", ha="center", va="bottom", fontsize=8, color="#888888")

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="V-measure = 1.0")
    ax.set_xticks(range(len(DATASETS_ORDER)))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS_ORDER], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("V-measure (DBSCANConstr vs DBSCAN)")
    ax.set_title("TaGra/DBSCANConstr ≡ DBSCAN: V-measure by dataset")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(FIGS_DIR, "fig_equivalence_check.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 4: Noise% boxplot ──────────────────────────────────────────────────

def fig_noise_boxplot(df_all):
    sub = df_all[df_all["method"].isin(METHODS_COMPARE) & (df_all["n_clusters"] >= 2)].copy()

    y_max = max(sub["noise_pct"].max() * 1.05, 35)

    fig, ax = plt.subplots(figsize=(9, 5))
    data_by_method = [sub[sub["method"] == m]["noise_pct"].dropna().values for m in METHODS_COMPARE]
    bp = ax.boxplot(
        data_by_method,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, method in zip(bp["boxes"], METHODS_COMPARE):
        patch.set_facecolor(METHOD_COLORS[method])

    ax.axhline(QUALITY["max_noise_pct"], color="#888888", linestyle="--",
               linewidth=1.2, label=f"noise threshold ({QUALITY['max_noise_pct']:.0f}%)")
    ax.legend(fontsize=9)

    ax.set_xticks(range(1, len(METHODS_COMPARE) + 1))
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS_COMPARE], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Noise %")
    ax.set_title("Noise% distribution per method (all configurations)")
    fig.tight_layout()
    path = os.path.join(FIGS_DIR, "fig_noise_boxplot.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 5: Outcome rate per cluster ────────────────────────────────────────

def fig_outcome_per_cluster(df, oc):
    pivot = best_dbcv_matrix(df)

    # Best method per dataset: highest DBCV; tie-break on lowest chi2_pval
    chi2_first = (
        oc[oc["method"].isin(METHODS_COMPARE)]
          .groupby(["dataset", "method"], as_index=False)["chi2_pval"]
          .first()
    )

    best_method_per_ds = {}
    for ds in DATASETS_ORDER:
        row_series = pivot.loc[:, ds] if ds in pivot.columns else pd.Series(dtype=float)
        row_series = row_series.dropna()
        if row_series.empty:
            best_method_per_ds[ds] = None
            continue
        max_dbcv = row_series.max()
        candidates = row_series[row_series == max_dbcv].index.tolist()
        if len(candidates) == 1:
            best_method_per_ds[ds] = candidates[0]
        else:
            # tie-break on chi2_pval (lower is better)
            sub_chi2 = chi2_first[
                (chi2_first["dataset"] == ds) & (chi2_first["method"].isin(candidates))
            ]
            if sub_chi2.empty:
                best_method_per_ds[ds] = candidates[0]
            else:
                best_method_per_ds[ds] = sub_chi2.sort_values("chi2_pval").iloc[0]["method"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for ax_idx, ds in enumerate(DATASETS_ORDER):
        ax = axes[ax_idx]
        method = best_method_per_ds.get(ds)

        if method is None:
            ax.text(0.5, 0.5, "No valid config", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{DATASET_LABELS[ds]}\nn/a")
            continue

        rows = oc[(oc["dataset"] == ds) & (oc["method"] == method)].copy()
        if rows.empty:
            ax.text(0.5, 0.5, "No outcome data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{DATASET_LABELS[ds]}\n{METHOD_LABELS.get(method, method)}")
            continue

        pval = rows["chi2_pval"].iloc[0]
        if isinstance(pval, float) and not math.isnan(pval):
            pval_str = f"p < 0.001" if pval < 0.001 else f"p = {pval:.3g}"
        else:
            pval_str = "p = n/a"

        rows = rows.sort_values("cluster")
        ns = rows["n"].values.astype(float)
        max_n = ns.max() if ns.max() > 0 else 1.0
        heights = np.clip(ns / max_n * 0.8, 0.15, 0.8)

        non_noise = rows[rows["cluster"] != -1]
        overall_rate = (
            non_noise["n_outcome_positive"].sum() / non_noise["n"].sum()
            if non_noise["n"].sum() > 0 else np.nan
        )

        ys = np.arange(len(rows))
        for i, (_, row) in enumerate(rows.iterrows()):
            clr = "#aaaaaa" if row["cluster"] == -1 else METHOD_COLORS[method]
            ax.barh(i, row["outcome_rate"], height=heights[i], color=clr, align="center")
            if row["n"] >= 10:
                ax.text(
                    min(row["outcome_rate"] - 0.01, 0.97), i,
                    f"n={row['n']}", ha="right", va="center",
                    fontsize=6, color="white", fontweight="bold",
                )

        if not math.isnan(overall_rate):
            ax.axvline(overall_rate, color="#555555", linestyle="--", linewidth=1, alpha=0.7)
            ax.text(
                overall_rate + 0.01, len(rows) - 0.5, "overall",
                fontsize=6, color="#555555", va="top",
            )

        labels = ["Noise" if r["cluster"] == -1 else str(r["cluster"]) for _, r in rows.iterrows()]
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0, 1.05)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x*100)}%"))
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(f"{DATASET_LABELS[ds]}\n{METHOD_LABELS.get(method, method)}, {pval_str}", fontsize=8)
        if ax_idx >= 4:
            ax.set_xlabel("Outcome rate", fontsize=7)

    fig.suptitle("Outcome rate per cluster — best method per dataset", fontsize=12)
    fig.tight_layout(h_pad=2.5, w_pad=1.5)
    path = os.path.join(FIGS_DIR, "fig_outcome_per_cluster.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 6: Chi-square p-value heatmap ──────────────────────────────────────

def fig_chi2_heatmap(oc):
    chi2_first = (
        oc[oc["method"].isin(METHODS_COMPARE)]
          .groupby(["dataset", "method"], as_index=False)["chi2_pval"]
          .first()
    )
    pivot = (
        chi2_first.pivot(index="method", columns="dataset", values="chi2_pval")
                  .reindex(index=METHODS_COMPARE, columns=DATASETS_ORDER)
    )

    log_data = -np.log10(pivot.values.astype(float))
    log_data = np.where(np.isnan(log_data), np.nan, log_data)

    fig, ax = plt.subplots(figsize=(12, 4))
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("#cccccc")

    im = ax.imshow(log_data, cmap=cmap, aspect="auto", vmin=0, vmax=4)

    ax.set_xticks(range(len(DATASETS_ORDER)))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS_ORDER], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(METHODS_COMPARE)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS_COMPARE], fontsize=9)

    for i, m in enumerate(METHODS_COMPARE):
        for j, ds in enumerate(DATASETS_ORDER):
            try:
                v = float(pivot.loc[m, ds])
            except (KeyError, TypeError, ValueError):
                v = float("nan")
            if math.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="#888888")
            else:
                ann = "< 0.001" if v < 0.001 else f"{v:.2f}"
                lv = -math.log10(v) if v > 0 else 4
                txt_color = "white" if lv > 2.5 else "black"
                ax.text(j, i, ann, ha="center", va="center", fontsize=7, color=txt_color)
                if v < 0.05:
                    rect = mpatches.FancyBboxPatch(
                        (j - 0.48, i - 0.48), 0.96, 0.96,
                        boxstyle="square,pad=0",
                        linewidth=1.5, edgecolor="black", facecolor="none",
                        transform=ax.transData,
                    )
                    ax.add_patch(rect)

    fig.colorbar(im, ax=ax, label="−log₁₀(p)", fraction=0.03, pad=0.02)
    ax.set_title("Clinical significance — chi-square p-value (outcome × cluster)")
    fig.tight_layout()
    path = os.path.join(FIGS_DIR, "fig_chi2_heatmap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Step 3b: Per-cluster feature profiles ─────────────────────────────────────

def build_cluster_profiles(results_dir, repo_root):
    """Write results/cluster_profiles.csv from saved label arrays."""
    labels_dir  = os.path.join(results_dir, "labels")
    bundle_dir  = os.path.join(repo_root, "dev/experiments/clustering/preprocessing/bundles")

    outcome_df = pd.read_csv(os.path.join(results_dir, "outcome_tables.csv"))
    best_df    = pd.read_csv(os.path.join(results_dir, "all_results.csv"))
    import json as _json

    best_df = best_df[best_df["method"].isin(METHODS_COMPARE)]
    best_per_ds = (
        best_df.sort_values("dbcv", ascending=False)
               .groupby("dataset", as_index=False)
               .first()
    )

    all_rows = []
    for _, row in best_per_ds.iterrows():
        ds     = row["dataset"]
        method = row["method"]

        label_path  = os.path.join(labels_dir, f"{ds}_{method}.npy")
        bundle_path = os.path.join(bundle_dir, f"{ds}_bundle.npz")
        meta_path   = os.path.join(bundle_dir, f"{ds}_meta.json")

        if not os.path.exists(label_path):
            print(f"  [skip] no saved labels for {ds}/{method}")
            continue
        if not os.path.exists(bundle_path):
            print(f"  [skip] bundle not found: {bundle_path}")
            continue

        cluster_labels = np.load(label_path)
        bundle         = np.load(bundle_path)
        X_raw          = bundle["X_raw"]
        with open(meta_path) as f:
            meta = _json.load(f)
        feat_names   = meta["feat_names"][:X_raw.shape[1]]
        gower_ftypes = meta["gower_ftypes"][:X_raw.shape[1]]

        df_feat            = pd.DataFrame(X_raw, columns=feat_names)
        df_feat["cluster"] = cluster_labels

        for c, grp in df_feat.groupby("cluster"):
            for i, feat in enumerate(feat_names):
                col_vals = grp[feat].dropna()
                ftype    = gower_ftypes[i] if i < len(gower_ftypes) else "continuous"
                is_cont  = ftype in ("continuous", "ordinal")
                all_rows.append({
                    "dataset":     ds,
                    "method":      METHOD_LABELS.get(method, method),
                    "cluster":     int(c),
                    "feature":     feat,
                    "ftype":       ftype,
                    "n_obs":       len(col_vals),
                    "pct_missing": round(100 * grp[feat].isna().mean(), 1),
                    "mean":        round(float(col_vals.mean()), 4) if is_cont and len(col_vals) > 0 else None,
                    "std":         round(float(col_vals.std()),  4) if is_cont and len(col_vals) > 0 else None,
                    "median":      round(float(col_vals.median()), 4) if is_cont and len(col_vals) > 0 else None,
                    "mode":        col_vals.mode().iloc[0] if len(col_vals) > 0 else None,
                })

    if all_rows:
        out_path = os.path.join(results_dir, "cluster_profiles.csv")
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
    else:
        print("cluster_profiles.csv not written — no label files found.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    df_all, df, oc = load_data()
    print(f"All configs  : {len(df_all):>5}")
    print(f"Valid configs: {len(df):>5}  ({100*len(df)/len(df_all):.1f}%)")
    print(f"  methods: {sorted(df['method'].unique())}")
    print(f"  outcome_tables: {len(oc)} rows")

    print("\n── Fig 1: Pareto scatter ──")
    fig_pareto_scatter(df_all)

    print("\n── Fig 2: Best DBCV heatmap ──")
    fig_best_dbcv_heatmap(df)

    print("\n── Fig 3: Equivalence check ──")
    fig_equivalence_check(df_all)

    print("\n── Fig 4: Noise% boxplot ──")
    fig_noise_boxplot(df_all)

    print("\n── Fig 5: Outcome rate per cluster ──")
    fig_outcome_per_cluster(df, oc)

    print("\n── Fig 6: Chi-square heatmap ──")
    fig_chi2_heatmap(oc)

    print("\n── Step 3b: Cluster profiles ──")
    build_cluster_profiles(RESULTS_DIR, REPO_ROOT)

    print("\nDone.")
