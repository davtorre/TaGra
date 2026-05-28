#!/usr/bin/env python
"""
CKD Chronic Kidney Disease – preprocessing pipeline.

Loads raw ARFF data (manual parser to handle whitespace-padded values),
applies column-typed preprocessing, and saves a bundle to bundles/.

Output:
    bundles/ckd_bundle.npz   – arrays X, X_raw, true_labels
    bundles/ckd_meta.json    – feat_names, gower_ftypes, n, class_balance,
                               dataset_name

Usage:
    cd dev/experiments/clustering
    python3 preprocess_ckd.py
"""

import os, sys, json
import numpy as np
import pandas as pd

REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
BUNDLE_DIR = os.path.join(os.path.dirname(__file__), "bundles")
DATA_PATH  = os.path.join(
    REPO_ROOT,
    "examples/datasets/UCI/Chronic_Kidney_Disease/chronic_kidney_disease_full.arff",
)

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from preprocess_typed import (
    preprocess_typed, feature_type_map_for_gower,
    CKD_SPEC, CKD_IMPUTE,
)


def load_ckd() -> pd.DataFrame:
    """Manual ARFF parser — scipy.io.arff fails on whitespace-padded values."""
    col_names, rows = [], []
    in_data = False
    with open(DATA_PATH, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.lower().startswith("@attribute"):
                parts = line.split()
                col_names.append(parts[1].strip("'"))
            elif line.lower() == "@data":
                in_data = True
            elif in_data:
                vals = [v.strip() or np.nan for v in line.rstrip(",").split(",")]
                if len(vals) == len(col_names):
                    rows.append(vals)

    df = pd.DataFrame(rows, columns=col_names)
    df.replace("?", np.nan, inplace=True)

    num_cols = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
                "hemo", "pcv", "wbcc", "rbcc"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["rbc"]   = (df["rbc"]   == "normal").astype(float)
    df["pc"]    = (df["pc"]    == "normal").astype(float)
    df["pcc"]   = (df["pcc"]   == "present").astype(float)
    df["ba"]    = (df["ba"]    == "present").astype(float)
    df["htn"]   = (df["htn"]   == "yes").astype(float)
    df["dm"]    = (df["dm"]    == "yes").astype(float)
    df["cad"]   = (df["cad"]   == "yes").astype(float)
    df["appet"] = (df["appet"] == "good").astype(float)
    df["pe"]    = (df["pe"]    == "yes").astype(float)
    df["ane"]   = (df["ane"]   == "yes").astype(float)

    df["target_binary"] = (df["class"] == "ckd").astype(int)
    return df


def prepare_raw_for_gower(df: pd.DataFrame):
    """Return the raw feature matrix (NaN preserved) for GowerDistanceConstructor."""
    feat_cols = [c for c, t in CKD_SPEC.items()
                 if t not in ("target", "id") and c in df.columns]
    return df[feat_cols].astype(float).values, feat_cols


def main():
    os.makedirs(BUNDLE_DIR, exist_ok=True)

    df = load_ckd()
    n  = len(df)

    X, feat_names = preprocess_typed(df, CKD_SPEC, CKD_IMPUTE, final_minmax=True)
    assert not np.isnan(X).any(), "NaN in X after preprocessing!"

    true_labels     = df["target_binary"].values
    X_raw, raw_cols = prepare_raw_for_gower(df)
    gower_ftypes    = feature_type_map_for_gower(CKD_SPEC, df)
    class_balance   = {str(k): int(v)
                       for k, v in zip(*np.unique(true_labels, return_counts=True))}

    bundle_path = os.path.join(BUNDLE_DIR, "ckd_bundle.npz")
    meta_path   = os.path.join(BUNDLE_DIR, "ckd_meta.json")

    np.savez(bundle_path, X=X, X_raw=X_raw, true_labels=true_labels)
    meta = {
        "dataset_name":  "CKD",
        "n":             n,
        "feat_names":    list(feat_names),
        "gower_ftypes":  gower_ftypes,
        "class_balance": class_balance,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {bundle_path}")
    print(f"Saved: {meta_path}")
    print(f"n={n}  X={X.shape}  X_raw={X_raw.shape}")
    print(f"Class balance: {class_balance}")
    print(f"Gower types:   {gower_ftypes}")


if __name__ == "__main__":
    main()
