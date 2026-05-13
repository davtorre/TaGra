#!/usr/bin/env python
"""
Parse all grid-search report files into a single CSV.

Reads every *_report.txt under a results root directory, extracts the
ALL RESULTS section, and writes one row per configuration with:
  dataset, method, params (JSON), n_clusters, noise_pct, min_cls_pct,
  dbcv, sil, vm_true, vm_ref

The 'method' and 'params' columns match the curated_configs format in
the JSON config files, so rows can be pasted back into a config directly.

Usage:
    cd dev/experiments/clustering
    python3 parse_results.py                          # default: ./results/reports
    python3 parse_results.py --results path/to/dir   # custom results root
    python3 parse_results.py --out results.csv        # custom output path
"""

import os, sys, re, json, argparse
import pandas as pd

# ---------------------------------------------------------------------------
# Method name → (method_key, params_dict)
# ---------------------------------------------------------------------------

_PATTERNS = [
    (re.compile(r'^DBSCAN \(eps=([\d.]+), ms=(\d+)\)$'),
     lambda m: ("dbscan",        {"eps": float(m.group(1)), "ms": int(m.group(2))})),

    (re.compile(r'^TaGra/DBSCANConstr \(eps=([\d.]+), ms=(\d+)\)$'),
     lambda m: ("dbscan_constr", {"eps": float(m.group(1)), "ms": int(m.group(2))})),

    (re.compile(r'^HDBSCAN \(mcs=(\d+)\)$'),
     lambda m: ("hdbscan",       {"mcs": int(m.group(1))})),

    (re.compile(r'^TaGra/Gower \((\w+),t=([\d.]+),df=(\d+)\)$'),
     lambda m: ("gower",         {"metric": m.group(1),
                                  "t": float(m.group(2)),
                                  "df": int(m.group(3))})),

    (re.compile(r'^TaGra/Sim \(t=([\d.]+),df=(\d+)\)$'),
     lambda m: ("sim",           {"t": float(m.group(1)), "df": int(m.group(2))})),

    (re.compile(r'^TaGra/KNN \(k=(\d+),df=(\d+)\)$'),
     lambda m: ("knn",           {"k": int(m.group(1)), "df": int(m.group(2))})),

    (re.compile(r'^TaGra/Dist \(t=([\d.]+),df=(\d+)\)$'),
     lambda m: ("dist",          {"t": float(m.group(1)), "df": int(m.group(2))})),
]


def parse_method(name: str):
    """Return (method_key, params_dict) or (None, None) if unrecognised."""
    for pattern, extractor in _PATTERNS:
        m = pattern.match(name)
        if m:
            return extractor(m)
    return None, None


# ---------------------------------------------------------------------------
# Row parser
# ---------------------------------------------------------------------------

def _float_or_none(s: str):
    s = s.strip()
    return None if s in ("N/A", "-", "") else float(s)


def parse_row(line: str):
    """
    Parse a single data row from the fixed-width report table.

    Layout (from run.py):
      2 spaces | name (52 chars) | n_clusters (4) | noise_pct (6.1f)% |
      min_cls_pct (8) | dbcv (8) | sil (8) | vm_true (9) | vm_ref (8)

    Returns a dict or None if the line is not a data row.
    """
    if len(line) < 56 or not line.startswith("  "):
        return None

    name = line[2:54].rstrip()
    if not name or name.startswith("-") or name.startswith("=") or name.startswith("Method"):
        return None

    # Everything after the name field, split on whitespace
    tokens = line[54:].split()
    # Expected: [n_clusters, noise_pct%, min_cls_pct%|N/A, dbcv|N/A, sil|N/A, vm_true, vm_ref|-]
    if len(tokens) < 7:
        return None

    try:
        n_clusters  = int(tokens[0])
        noise_pct   = float(tokens[1].rstrip("%"))
        min_cls_raw = tokens[2]
        min_cls_pct = None if min_cls_raw == "N/A" else float(min_cls_raw.rstrip("%"))
        dbcv        = _float_or_none(tokens[3])
        sil         = _float_or_none(tokens[4])
        vm_true     = _float_or_none(tokens[5])
        vm_ref      = _float_or_none(tokens[6])
    except (ValueError, IndexError):
        return None

    method, params = parse_method(name)
    if method is None:
        return None

    return {
        "method":      method,
        "params":      json.dumps(params, separators=(",", ":")),
        "n_clusters":  n_clusters,
        "noise_pct":   noise_pct,
        "min_cls_pct": min_cls_pct,
        "dbcv":        dbcv,
        "sil":         sil,
        "vm_true":     vm_true,
        "vm_ref":      vm_ref,
    }


# ---------------------------------------------------------------------------
# Report file parser — reads only the ALL RESULTS section
# ---------------------------------------------------------------------------

def parse_report(path: str):
    """Return list of row dicts from the ALL RESULTS section of a report."""
    rows = []
    in_all_results = False

    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")

            # Detect section boundaries
            if line.startswith("=" * 10):
                if in_all_results:
                    break           # end of ALL RESULTS — stop
                continue

            if line.strip() == "ALL RESULTS":
                in_all_results = True
                continue

            if not in_all_results:
                continue

            row = parse_row(line)
            if row:
                rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parse grid-search reports into CSV")
    parser.add_argument(
        "--results",
        default=os.path.join(os.path.dirname(__file__), "results", "reports"),
        help="Root directory containing clustering_results_*/ subdirectories",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "results", "all_results.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    records = []

    for entry in sorted(os.listdir(args.results)):
        subdir = os.path.join(args.results, entry)
        if not (os.path.isdir(subdir) and entry.startswith("clustering_results_")):
            continue

        # Find the report file
        txt_files = [f for f in os.listdir(subdir) if f.endswith("_report.txt")]
        if not txt_files:
            print(f"  [skip] no report in {entry}")
            continue
        report_path = os.path.join(subdir, txt_files[0])

        # Extract dataset name from first line: "DatasetName – Clustering Comparison"
        with open(report_path) as fh:
            first_line = fh.readline().strip()
        dataset = first_line.split("–")[0].strip()

        rows = parse_report(report_path)
        for row in rows:
            row["dataset"] = dataset
        records.extend(rows)
        print(f"  {dataset:<20} {len(rows):>4} rows  ({report_path})")

    if not records:
        print("No records found — check --results path.")
        sys.exit(1)

    df = pd.DataFrame(records, columns=[
        "dataset", "method", "params",
        "n_clusters", "noise_pct", "min_cls_pct",
        "dbcv", "sil", "vm_true", "vm_ref",
    ])
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows → {args.out}")


if __name__ == "__main__":
    main()
