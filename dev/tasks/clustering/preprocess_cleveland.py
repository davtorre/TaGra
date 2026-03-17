#!/usr/bin/env python
"""
Cleveland Heart Disease – preprocessing pipeline.

Loads the processed.cleveland.data CSV, applies column-typed preprocessing,
and saves a bundle to bundles/.

Output:
    bundles/cleveland_bundle.npz  – arrays X, X_raw, true_labels
    bundles/cleveland_meta.json   – feat_names, gower_ftypes, n, class_balance,
                                    dataset_name

Usage:
    cd dev/tasks/clustering
    python3 preprocess_cleveland.py
"""

import os, sys, json
import numpy as np
import pandas as pd

REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
BUNDLE_DIR = os.path.join(os.path.dirname(__file__), "bundles")
DATA_PATH  = os.path.join(
    REPO_ROOT, "examples/datasets/UCI/heart_disease/processed.cleveland.data"
)
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from preprocess_typed import (
    preprocess_typed, feature_type_map_for_gower,
    CLEVELAND_SPEC, CLEVELAND_IMPUTE,
)


def load_cleveland() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, header=None, names=COLS, na_values="?")


def prepare_raw_for_gower(df: pd.DataFrame):
    df2 = df.copy()
    for col, strat in CLEVELAND_IMPUTE.items():
        if col not in df2.columns:
            continue
        df2[col] = df2[col].fillna(
            df2[col].median() if strat == "median" else df2[col].mode()[0]
        )
    feat_cols = [c for c, t in CLEVELAND_SPEC.items()
                 if t not in ("target", "id") and c in df.columns]
    return df2[feat_cols].astype(float).values, feat_cols


def main():
    os.makedirs(BUNDLE_DIR, exist_ok=True)

    df = load_cleveland()
    n  = len(df)

    X, feat_names = preprocess_typed(df, CLEVELAND_SPEC, CLEVELAND_IMPUTE, final_minmax=True)
    assert not np.isnan(X).any(), "NaN in X after preprocessing!"

    true_labels     = (df["num"].fillna(0).values > 0).astype(int)
    X_raw, raw_cols = prepare_raw_for_gower(df)
    gower_ftypes    = feature_type_map_for_gower(CLEVELAND_SPEC, df)
    class_balance   = {str(k): int(v)
                       for k, v in zip(*np.unique(true_labels, return_counts=True))}

    bundle_path = os.path.join(BUNDLE_DIR, "cleveland_bundle.npz")
    meta_path   = os.path.join(BUNDLE_DIR, "cleveland_meta.json")

    np.savez(bundle_path, X=X, X_raw=X_raw, true_labels=true_labels)
    meta = {
        "dataset_name":  "Cleveland",
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
