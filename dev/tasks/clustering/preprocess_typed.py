"""
Column-typed preprocessing pipeline for mixed EHR datasets.

Applies feature-specific transformations before Euclidean-based clustering
(DBSCAN, HDBSCAN, TaGra graph methods):

  continuous → StandardScaler
  binary     → keep as 0/1
  ordinal    → divide by observed max → [0, 1]
  nominal    → one-hot encode (drop first dummy to avoid collinearity)
  id         → drop

Returns a float64 numpy array with all features in comparable ranges.

Usage
-----
from preprocess_typed import preprocess_typed, CLEVELAND_SPEC

X, feature_names = preprocess_typed(df, CLEVELAND_SPEC)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Dataset-specific feature specifications
# ---------------------------------------------------------------------------

# Cleveland Heart Disease (processed.cleveland.data, 14 cols)
CLEVELAND_SPEC = {
    # continuous
    "age":      "continuous",
    "trestbps": "continuous",
    "chol":     "continuous",
    "thalach":  "continuous",
    "oldpeak":  "continuous",
    # binary
    "sex":      "binary",
    "fbs":      "binary",
    "exang":    "binary",
    # ordinal (0-based or 1-based, scaled to [0,1] by dividing by max)
    "restecg":  "ordinal",    # 0, 1, 2
    "slope":    "ordinal",    # 1, 2, 3
    "ca":       "ordinal",    # 0, 1, 2, 3  (2 missing)
    # nominal (unordered categories — indicator distance in Gower,
    #          one-hot encoding for Euclidean methods)
    "cp":       "nominal",    # 1, 2, 3, 4  (chest pain type)
    "thal":     "nominal",    # 3, 6, 7     (thalassemia code)
    # target — excluded from X
    "num":      "target",
    "target":   "target",
}


# HCV Hepatitis C Virus (hcvdat0.csv, 13 cols)
# Target: Category (0=Blood Donor, 0s=suspect, 1=Hepatitis, 2=Fibrosis, 3=Cirrhosis)
# Sex must be encoded as 0/1 before calling preprocess_typed (m→1, f→0).
HCV_SPEC = {
    "Age":  "continuous",
    "Sex":  "binary",       # pre-encoded: m=1, f=0
    "ALB":  "continuous",   # albumin
    "ALP":  "continuous",   # alkaline phosphatase
    "ALT":  "continuous",   # alanine aminotransferase
    "AST":  "continuous",   # aspartate aminotransferase
    "BIL":  "continuous",   # bilirubin
    "CHE":  "continuous",   # cholinesterase
    "CHOL": "continuous",   # cholesterol
    "CREA": "continuous",   # creatinine
    "GGT":  "continuous",   # gamma-glutamyl transferase
    "PROT": "continuous",   # total protein
    "Category": "target",
}

# ---------------------------------------------------------------------------
# Imputation defaults per dataset
# ---------------------------------------------------------------------------

CLEVELAND_IMPUTE = {
    "ca":   "median",   # 2 missing
    "thal": "mode",     # 2 missing
}

HCV_IMPUTE = {
    "ALB":  "median",   # 1 missing
    "ALP":  "median",   # 18 missing
    "ALT":  "median",   # 1 missing
    "CHOL": "median",   # 10 missing
    "PROT": "median",   # 1 missing
}


# Chronic Kidney Disease (chronic_kidney_disease_full.arff, 25 cols)
# Target: class (ckd=1, notckd=0)
# All binary cols must be pre-encoded as 0/1 before calling preprocess_typed.
CKD_SPEC = {
    # continuous
    "age":   "continuous",
    "bp":    "continuous",
    "bgr":   "continuous",
    "bu":    "continuous",
    "sc":    "continuous",
    "sod":   "continuous",
    "pot":   "continuous",
    "hemo":  "continuous",
    "pcv":   "continuous",
    "wbcc":  "continuous",
    "rbcc":  "continuous",
    # ordinal (numeric scale, divided by max → [0,1])
    "sg":    "ordinal",   # 1.005, 1.010, 1.015, 1.020, 1.025
    "al":    "ordinal",   # 0–5
    "su":    "ordinal",   # 0–5
    # binary (pre-encoded: yes/normal/present/good → 1, no/abnormal/notpresent/poor → 0)
    "rbc":   "binary",
    "pc":    "binary",
    "pcc":   "binary",
    "ba":    "binary",
    "htn":   "binary",
    "dm":    "binary",
    "cad":   "binary",
    "appet": "binary",
    "pe":    "binary",
    "ane":   "binary",
    # target
    "class": "target",
}

CKD_IMPUTE = {
    # continuous (median)
    "age":   "median",   # 0.5%
    "bp":    "median",   # 4.5%
    "bgr":   "median",   # 22.3%
    "bu":    "median",   # 11.0%
    "sc":    "median",   # 11.5%
    "sod":   "median",   # 26.5%
    "pot":   "median",   # 34.5%
    "hemo":  "median",   # 21.8%
    "pcv":   "median",   # 22.0%
    "wbcc":  "median",   # 26.3%
    "rbcc":  "median",   # 28.0%
    # ordinal (median)
    "sg":    "median",   # 13.3%
    "al":    "median",   # 11.3%
    "su":    "median",   # 11.3%
    # binary (mode)
    "rbc":   "mode",     # 38.3%
    "pc":    "mode",     # 15.3%
    "pcc":   "mode",     # 4.5%
    "ba":    "mode",     # 4.5%
    "htn":   "mode",     # 2.3%
    "dm":    "mode",     # 2.3%
    "cad":   "mode",     # 2.5%
    "appet": "mode",     # 2.3%
    "pe":    "mode",     # 2.3%
    "ane":   "mode",     # 2.3%
}


# ---------------------------------------------------------------------------
# New EHR datasets
# ---------------------------------------------------------------------------

# Neuroblastoma (10_7717_peerj_5665_dataYM2018, 169 rows)
# Target: outcome (0=alive, 1=dead)
# Note: CSV column "MYCN_status " has a trailing space — strip in loader.
NEUROBLASTOMA_SPEC = {
    "age":                                  "continuous",
    "sex":                                  "binary",
    "site":                                 "nominal",    # 0, 1, 2
    "stage":                                "binary",
    "risk":                                 "binary",
    "time_months":                          "continuous",
    "autologous_stem_cell_transplantation": "binary",
    "radiation":                            "binary",
    "degree_of_differentiation":            "ordinal",    # 0, 1, 2
    "UH_or_FH":                             "binary",
    "MYCN_status":                          "binary",     # trailing space stripped in loader
    "surgical_methods":                     "binary",
    "outcome":                              "target",
}

NEUROBLASTOMA_IMPUTE = {}  # no missing values


# Sepsis / SIRS (journal.pone.0148699, 1257 rows)
# Target: Mortality (0=survived, 1=died)
SEPSIS_SPEC = {
    "Age":                    "continuous",
    "sex_woman":              "binary",
    "diagnosis_0EC_1M_2_AC":  "nominal",    # 0=EC, 1=M, 2=AC
    "APACHE II":              "continuous",
    "SOFA":                   "continuous",
    "CRP":                    "continuous",
    "WBCC":                   "continuous",
    "NeuC":                   "continuous",
    "LymC":                   "continuous",
    "EOC":                    "continuous",
    "NLCR":                   "continuous",
    "PLTC":                   "continuous",
    "MPV":                    "continuous",
    "Group":                  "binary",     # 0=SIRS, 1=Sepsis
    "LOS-ICU":                "continuous",
    "Mortality":              "target",
}

SEPSIS_IMPUTE = {}  # no missing values


# Depression + Heart Failure (journal.pone.0158570, 425 rows)
# Target: Death (1=Yes, 0=No)
DEPRESSION_HF_SPEC = {
    "Age (years)":                              "continuous",
    "Male (1=Yes, 0=No)":                       "binary",
    "PHQ-9":                                    "ordinal",   # 0–25 depression scale
    "Systolic BP (mm Hg)":                      "continuous",
    "Estimated glomerular filtration rate":     "continuous",
    "Ejection fraction (%)":                    "continuous",
    "Serum sodium (mmol/l)":                    "continuous",
    "Blood urea nitrogen (mg/dl)":              "continuous",
    "Etiology HF(1=Yes, 0=No)":                 "binary",
    "Prior diabetes mellitus":                  "binary",
    "Elevated level of BNP/NT-BNP (1=Yes, 0=No)": "binary",
    "Time from HF to Death (days)":             "continuous",
    "Death (1=Yes, 0=No)":                      "target",
    "Time from HF to hospitalization (days)":   "continuous",
    "Hospitalized (1=Yes, 0=No)":               "binary",
}

DEPRESSION_HF_IMPUTE = {}  # no missing values


# Cardiac Arrest – Spain (journal.pone.0175818, 422 rows, 7 missing)
# Target: Exitus (0=survived, 1=died)
CARDIAC_ARREST_SPEC = {
    "Exitus":                    "target",
    "sex_woman":                 "binary",
    "Age_years":                 "continuous",
    "Endotracheal_intubation":   "binary",
    "Functional_status":         "ordinal",   # 0, 1, 2, 3
    "Asystole":                  "binary",
    "Bystander":                 "binary",
    "Time_min":                  "continuous",
    "Cardiogenic":               "binary",
    "Cardiac_arrest_at_home":    "binary",
}

CARDIAC_ARREST_IMPUTE = {
    "Age_years":        "median",   # 4 missing
    "Functional_status": "median",  # 3 missing
}


# Diabetes Type 1 – Takashi 2019 (67 rows)
# Target: insulin_regimen_binary (0=MDI, 1=CSII)
DIABETES_T1_SPEC = {
    "age":                      "continuous",
    "duration.of.diabetes":     "continuous",
    "body_mass_index":          "continuous",
    "TDD":                      "continuous",
    "basal":                    "continuous",
    "bolus":                    "continuous",
    "HbA1c":                    "continuous",
    "eGFR":                     "continuous",
    "perc.body.fat":            "continuous",
    "adiponectin":              "continuous",
    "free.testosterone":        "continuous",
    "SMI":                      "continuous",
    "grip.strength":            "continuous",
    "knee.extension.strength":  "continuous",
    "gait.speed":               "continuous",
    "ucOC":                     "continuous",
    "OC":                       "continuous",
    "weight_kg":                "continuous",
    "sex_0man_1woman":          "binary",
    "insulin_regimen_binary":   "target",
}

DIABETES_T1_IMPUTE = {}  # no missing values


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def preprocess_typed(
    df: pd.DataFrame,
    spec: dict,
    impute: dict = None,
    final_minmax: bool = True,
) -> tuple:
    """
    Apply column-typed preprocessing to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data.
    spec : dict
        Mapping column_name → type string.
        Types: 'continuous', 'binary', 'ordinal', 'nominal', 'id', 'target'.
        Columns not in spec are dropped with a warning.
    impute : dict, optional
        Mapping column_name → 'median' | 'mode'.
        Applied before any transformation.
    final_minmax : bool, default True
        If True, apply MinMaxScaler to the full output matrix so every
        feature lands in [0, 1].  Recommended for HDBSCAN; optional for DBSCAN.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features), dtype float64
    feature_names : list of str
        Column names in the same order as X columns.
    """
    df = df.copy()
    impute = impute or {}

    # --- 1. Imputation ---
    for col, strategy in impute.items():
        if col not in df.columns:
            continue
        if strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            df[col] = df[col].fillna(df[col].mode()[0])

    # --- 2. Split columns by type ---
    continuous, binary, ordinal, nominal = [], [], [], []
    for col, dtype in spec.items():
        if col not in df.columns:
            continue
        if dtype == "continuous":
            continuous.append(col)
        elif dtype == "binary":
            binary.append(col)
        elif dtype == "ordinal":
            ordinal.append(col)
        elif dtype == "nominal":
            nominal.append(col)
        # 'id' and 'target' are skipped

    parts = []
    names = []

    # --- 3. Continuous: StandardScaler ---
    if continuous:
        X_cont = StandardScaler().fit_transform(
            df[continuous].astype(float)
        )
        parts.append(X_cont)
        names.extend(continuous)

    # --- 4. Binary: keep as float ---
    if binary:
        parts.append(df[binary].astype(float).values)
        names.extend(binary)

    # --- 5. Ordinal: divide by max → [0, 1] ---
    if ordinal:
        X_ord = df[ordinal].astype(float)
        X_ord = X_ord / X_ord.max()
        parts.append(X_ord.values)
        names.extend(ordinal)

    # --- 6. Nominal: one-hot encode (drop first) ---
    for col in nominal:
        dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True)
        parts.append(dummies.values.astype(float))
        names.extend(dummies.columns.tolist())

    if not parts:
        raise ValueError("No features selected — check the spec dict.")

    X = np.hstack(parts).astype(np.float64)

    # --- 7. Optional final MinMaxScaler ---
    if final_minmax:
        from sklearn.preprocessing import MinMaxScaler
        X = MinMaxScaler().fit_transform(X)

    return X, names


# ---------------------------------------------------------------------------
# Quality criteria for clustering results
# ---------------------------------------------------------------------------

QUALITY_CRITERIA = {
    "min_dbcv":              0.2,   # DBCV >= 0.2
    "max_noise_pct":        30.0,   # noise < 30 %
    "min_cluster_size_pct": 20.0,   # smallest cluster >= 20 % of non-noise points
    "min_n_clusters":        2,     # at least 2 real clusters
}


def filter_results(results: list, criteria: dict = None) -> list:
    """
    Return only results that satisfy all quality criteria.

    Each result dict must contain the keys produced by record():
        n_clusters, noise_pct, dbcv, labels (np.ndarray).

    Parameters
    ----------
    results  : list of result dicts (as returned by record())
    criteria : dict of thresholds; defaults to QUALITY_CRITERIA

    Returns
    -------
    list of result dicts passing all filters
    """
    import numpy as np
    if criteria is None:
        criteria = QUALITY_CRITERIA

    passed = []
    for r in results:
        # 1. Minimum number of clusters
        if r["n_clusters"] < criteria["min_n_clusters"]:
            continue
        # 2. Maximum noise
        if r["noise_pct"] >= criteria["max_noise_pct"]:
            continue
        # 3. Minimum DBCV (None counts as failing)
        if r["dbcv"] is None or r["dbcv"] < criteria["min_dbcv"]:
            continue
        # 4. Minimum cluster size (% of non-noise points)
        lbl = np.asarray(r["labels"])
        non_noise_mask = lbl != -1
        n_non_noise = non_noise_mask.sum()
        if n_non_noise == 0:
            continue
        cluster_ids = [c for c in set(lbl) if c != -1]
        min_frac = min((lbl == c).sum() / n_non_noise * 100 for c in cluster_ids)
        if min_frac < criteria["min_cluster_size_pct"]:
            continue
        passed.append(r)
    return passed


def feature_type_map_for_gower(spec: dict, df: pd.DataFrame) -> list:
    """
    Return a feature_types list aligned with preprocess_typed output columns,
    for use with GowerDistanceConstructor (skips one-hot; keeps nominals as-is).

    Note: this returns types for the RAW columns (before one-hot), so it is
    intended for direct use with GowerDistanceConstructor on the original df,
    not on the preprocess_typed output.
    """
    types = []
    for col, dtype in spec.items():
        if col not in df.columns:
            continue
        if dtype in ("id", "target"):
            continue
        if dtype == "nominal":
            types.append("nominal")
        elif dtype == "ordinal":
            types.append("ordinal")
        elif dtype == "binary":
            types.append("binary")
        else:
            types.append("continuous")
    return types
