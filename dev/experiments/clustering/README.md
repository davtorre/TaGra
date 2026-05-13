# Clustering Experiments: TaGra Generalizes DBSCAN

This directory contains the experimental validation for the claim that TaGra's
graph construction framework generalizes DBSCAN to support multiple types of
neighborhood structures (distance-threshold, KNN, cosine similarity, Gower).

## Pipeline

```
raw data  ──►  preprocessing/  ──►  bundles/  ──►  run.py  ──►  results/
                    │                                              │
                    └── preprocess_typed.py                        ├── reports/{dataset}_report.txt
                    + 8 dataset-specific scripts                   ├── all_results.csv
                                                                   └── clustering_radar_report.html
```

## Reproducing

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Preprocess data (one per dataset)
cd dev/experiments/clustering
python3 preprocessing/preprocess_hcv.py

# 3. Run clustering comparison
python3 run.py --bundle preprocessing/bundles/hcv_bundle.npz --config config/hcv.json --top

# 4. Batch run all datasets
bash run_all.sh

# 5. Parse results into CSV
python3 parse_results.py

# 6. Generate radar report
python3 radar_report.py
```

## Parameter search

```bash
python3 search_params.py --bundle preprocessing/bundles/hcv_bundle.npz
```

## Structure

| Path | Purpose |
|------|---------|
| `preprocessing/` | Preprocessing pipeline (typed column specs, 8 datasets) |
| `preprocessing/bundles/` | Preprocessed `.npz` + metadata |
| `config/` | JSON configs: parameters, grid sweeps, quality criteria |
| `run.py` | Main clustering comparison pipeline |
| `search_params.py` | Parameter estimation (SS-DBSCAN method) |
| `parse_results.py` | Parse report files into CSV |
| `radar_report.py` | Generate interactive Bokeh radar report |
| `results/` | All generated outputs (reports, CSV, HTML) |
