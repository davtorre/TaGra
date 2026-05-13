# CLAUDE.md

Context file for Claude Code sessions working on the TaGra project.

## Project Overview

TaGra (Table to Graph) v0.3.0 is a Python library for preprocessing tabular data, constructing graphs from it, and performing machine learning on graph-structured data. Four main capabilities:

1. **Data Preprocessing**: Missing values, scaling, encoding, manifold learning (Isomap, TSNE, UMAP)
2. **Graph Creation**: KNN, distance threshold, similarity threshold
3. **Graph Analysis**: Degree distribution, community detection, neighborhood probability, homophily
4. **Machine Learning on Graphs**: Label propagation, graph-aware augmentation, anomaly detection, imputation

## Repository Layout

```
tagra/                          # Main package (pip-installable)
├── __init__.py                 # Public API: from_dataframe(), analyze(), visualize()
├── core/                       # TaGraGraph, GraphMetadata, enums
│   ├── graph.py
│   └── types.py
├── preprocessing/              # preprocess() function, pipeline, scalers, encoders
│   ├── __init__.py
│   ├── pipeline.py
│   ├── scaling.py
│   ├── encoding.py
│   ├── missing.py
│   ├── manifold.py
│   ├── inference.py
│   └── base.py
├── construction/               # build_graph() function, constructor classes
│   ├── __init__.py             # build_graph() lives here
│   ├── base.py                 # Abstract GraphConstructor
│   ├── knn.py                  # KNNConstructor
│   ├── distance.py             # DistanceThresholdConstructor
│   ├── similarity.py           # SimilarityThresholdConstructor
│   └── registry.py             # ConstructorRegistry
├── analysis/                   # analyze() function, metrics, community detection
│   ├── __init__.py
│   ├── metrics.py
│   ├── neighborhood.py
│   ├── community.py
│   └── report.py
├── visualization/              # Plotting (matplotlib, heatmaps, distributions)
│   ├── __init__.py
│   ├── graph_plot.py
│   ├── heatmap.py
│   ├── distribution.py
│   └── community_plot.py
├── ml/                         # ML capabilities (v0.3.0)
│   ├── __init__.py
│   ├── label_propagation.py    # LabelPropagator
│   ├── augmentation.py         # GraphAwareAugmenter
│   ├── anomaly.py              # GraphAnomalyDetector, AnomalyExplanation
│   └── imputation.py           # GraphImputer
├── io/                         # File I/O (readers, writers, Cytoscape/GraphML export)
│   ├── readers.py
│   ├── writers.py
│   └── exporters.py
├── compat/                     # Legacy API wrappers with deprecation warnings
│   └── legacy.py
├── exceptions.py               # Custom exceptions
├── config.py                   # JSON config loading
└── cytoscape_vis.py            # Optional Cytoscape visualization

go.py                           # CLI entry point
dev/                            # Development workspace (not in git)
├── docs/                       # This file, roadmap, paper notes
└── tasks/                      # Test scripts and experiments
    ├── clustering/             # DBSCAN vs TaGra comparison pipeline
    │   ├── preprocess_{hcv,ckd,cleveland}.py  # Stage 2: produce bundles
    │   ├── run_clustering.py                  # Stage 3: unified clustering
    │   ├── preprocess_typed.py                # Typed preprocessing + quality filter
    │   ├── config_{hcv,ckd,cleveland}.json    # Per-dataset model parameters
    │   └── bundles/                           # .npz + .json bundles (not in git)
    └── anomaly/                # Anomaly detection comparison
```

## Build & Development Commands

```bash
pip install -r requirements.txt
pip install -e .                                         # Editable install
python -m pytest tests/                                  # Run tests
python go.py -c examples/article/takashi_similarity.json # Full pipeline via config
python go.py -d path/to/dataframe.csv -a target_column   # Direct usage
```

## Key APIs

### Constructor pattern (used in test scripts)

Constructors modify a pre-existing NetworkX graph in-place by adding edges:

```python
from tagra.construction.knn import KNNConstructor
from tagra.construction.distance import DistanceThresholdConstructor
from tagra.construction.similarity import SimilarityThresholdConstructor

G = nx.Graph()
for i in range(n): G.add_node(i)

constructor = KNNConstructor(k=5, verbose=False)
constructor.construct(G, values)  # values: np.ndarray (n_samples, n_features)
```

### High-level API

```python
import tagra

graph = tagra.from_dataframe(df, target='label', method='knn', k=5)
metrics = tagra.analyze(graph, target_attribute='label')
tagra.visualize(graph, output='graph.png')
```

### Modular API

```python
from tagra.preprocessing import preprocess
from tagra.construction import build_graph
from tagra.analysis import analyze

df_processed, positions = preprocess(df, target_columns=['label'])
graph = build_graph(df, df_processed, method='knn', k=5)
metrics = analyze(graph, target_attributes='label')
```

### ML module

```python
from tagra.ml import LabelPropagator, GraphAwareAugmenter, GraphAnomalyDetector, GraphImputer
```

### Legacy API (deprecated, still works)

```python
from tagra import preprocess_dataframe, create_graph, analyze_graph
```

## Key Dependencies

- pandas, numpy, scikit-learn, networkx, scipy, matplotlib
- umap-learn (optional, lazy-imported)
- ipycytoscape (optional)
- hdbscan (optional, for DBCV metric in test scripts)

## Architecture Decisions

### Why modular subpackages (v0.3.0)

The v0.2.x codebase had monolithic functions (15-25 parameters each) in flat modules. This made it impossible to add ML capabilities without modifying existing code. The restructuring (Feb 6, 2026) introduced:

- **Constructor registry**: New graph construction methods can be added without modifying existing code
- **Abstract base classes**: `GraphConstructor`, `Transformer` provide extension points
- **Backward compatibility layer**: `tagra/compat/legacy.py` wraps old API with deprecation warnings
- **TaGraGraph class**: Wraps NetworkX graph + metadata, provides save/load

### Constructor API convention

Constructors take `(G: nx.Graph, values: np.ndarray)` and modify G in-place. They do NOT return a new graph. The `build_graph()` convenience function in `construction/__init__.py` handles node creation, file I/O, and metadata.

### Graph construction methods

| Method | Class | Key param | Algorithm |
|--------|-------|-----------|-----------|
| KNN | `KNNConstructor` | `k` | scipy cKDTree, k+1 query, skip self |
| Distance | `DistanceThresholdConstructor` | `distance_threshold` | cKDTree.query_pairs |
| Similarity | `SimilarityThresholdConstructor` | `similarity_threshold` | sklearn cosine_similarity matrix |

## Current Development Focus

### Clustering pipeline (active)

Full pipeline comparing DBSCAN vs TaGra across three UCI datasets (Cleveland Heart Disease, HCV Hepatitis C, Chronic Kidney Disease). The pipeline is split into four stages:

#### 1. EDA
Scripts in `examples/EDA/generate_eda.py`. Produces distributions, class balance, missing value summaries. Run independently — not part of preprocessing or clustering.

#### 2. Preprocessing (`dev/tasks/clustering/preprocess_*.py`)
One script per dataset. Each: loads raw data → applies column-typed preprocessing → saves a **bundle** to `dev/tasks/clustering/bundles/`.

- `preprocess_hcv.py` → `bundles/hcv_bundle.npz` + `bundles/hcv_meta.json`
- `preprocess_ckd.py` → `bundles/ckd_bundle.npz` + `bundles/ckd_meta.json`
- `preprocess_cleveland.py` → `bundles/cleveland_bundle.npz` + `bundles/cleveland_meta.json`

Bundle format: `.npz` for arrays (`X`, `X_raw`, `true_labels`), `.json` for metadata (`feat_names`, `gower_ftypes`, `dataset_name`, `class_balance`). No pickles.

Column-typed preprocessing lives in `preprocess_typed.py`:
- `continuous` → StandardScaler
- `ordinal` → divide by max
- `binary` → 0/1 as-is
- `nominal` → one-hot
- Optional final MinMaxScaler (`final_minmax=True`)
- Gower distance uses `X_raw` (imputed but unscaled); `feature_type_map_for_gower()` returns the type list

#### 3. Clustering (`dev/tasks/clustering/run_clustering.py`)
Single unified script, config-driven. Usage:

```bash
python3 run_clustering.py --bundle bundles/hcv_bundle.npz --config config_hcv.json
```

Pipeline inside the script:
1. **DBSCAN references** — pinned `(eps, ms)` pairs from config, always run
2. **TaGra DBSCANConstructor equivalence** — proves TaGra/distance ≡ DBSCAN (Vm(ref)=1.0)
3. **HDBSCAN** — `mcs` values from config
4. **Exhaustive grid** (if `"exhaustive_grid": true` in config):
   - DBSCAN sweep: full eps × ms grid, results **quality-filtered** before display
   - TaGra/Gower, Similarity, KNN, Distance grids — top 20 by Vm, **quality-filtered**
5. **Curated comparison** — explicit best configs from `cfg["curated_configs"]`, always run
6. **Summary A** (by Vm), **Summary B** (by DBCV), **Quality-filtered summary**, **Top-3 per metric**
7. Saves plain-text report to `clustering_results_{name}/`

#### 4. Per-dataset JSON configs (`config_*.json`)
All model parameters live in the config — no hardcoded values in the script:

| Key | Purpose |
|-----|---------|
| `exhaustive_grid` | `true`/`false` — enables full grid sweeps |
| `dbscan_refs` | Pinned `[{eps, ms}]` reference configs |
| `hdbscan_mcs` | List of min_cluster_size values |
| `dbscan_sweep` | `eps_grid`, `ms_grid` for informational sweep |
| `gower_grid` / `sim_grid` / `knn_grid` / `dist_grid` | Grid bounds per method |
| `curated_configs` | List of `{method, ...params}` for the comparison table |
| `quality_criteria` | Overrides for `QUALITY_CRITERIA` defaults |
| `summaries` | List of summary table definitions (optional, see below) |

Cleveland uses `"exhaustive_grid": false` (curated comparison only). HCV and CKD use `"exhaustive_grid": true`. A sweep-only config (e.g. `config_cleveland_dbscan_sweep.json`) can set empty grids for all methods except DBSCAN.

#### Summaries (config-driven, optional)
Defined as a list in the config — omitting the key skips all summary tables:

```json
"summaries": [
  {"name": "A", "max_noise_pct": 60.0, "sort": "vm_true"},
  {"name": "B", "max_noise_pct": 60.0, "sort": ["dbcv", "noise_pct"]}
]
```

- `sort` as a string → sort by that field descending
- `sort` as a list → first field descending, remaining fields ascending
- Any result field is valid: `vm_true`, `vm_ref`, `dbcv`, `sil`, `noise_pct`, `min_cls_pct`
- `--top` CLI flag prints top-3 per metric at the end (also written to report)

#### Quality criteria (`preprocess_typed.py`)
`QUALITY_CRITERIA` defaults + `filter_results()` function:
- `min_dbcv ≥ 0.2`
- `max_noise_pct < 30%`
- `min_cluster_size_pct ≥ 20%` of non-noise points
- `min_n_clusters ≥ 2`

Applied consistently in: DBSCAN sweep display, all exhaustive grid top-20 tables, and the final quality-filtered summary section. Curated configs are merged into `all_results` (deduplicated by name) so they also appear in the quality-filtered section.

#### Key metrics (all printed as table columns)
| Metric | Key | Description |
|--------|-----|-------------|
| Clusters | `n_clusters` | Number of non-noise clusters |
| Noise% | `noise_pct` | % of points labelled as noise |
| MinCls% | `min_cls_pct` | Smallest cluster as % of non-noise points |
| DBCV | `dbcv` | Density-Based Clustering Validation (`hdbscan.validity`) |
| Sil | `sil` | Silhouette score (sklearn) |
| Vm(true) | `vm_true` | V-measure vs ground-truth binary clinical label |
| Vm(ref) | `vm_ref` | V-measure vs DBSCAN Ref A (structural similarity) |

### Anomaly detection comparison (implemented)

Systematic comparison of sklearn anomaly detectors vs TaGra's `GraphAnomalyDetector`.

Script: `dev/tasks/anomaly/anomaly_task_test.py`

## Configuration

JSON config files. Key parameters:
- `input_dataframe`, `method`, `k`, `distance_threshold`, `similarity_threshold`
- `manifold_method`: "Isomap", "TSNE", "UMAP", or null
- `target_columns`, `unknown_column_action`: "infer" or "ignore"

See `tagra/config.py` for defaults and `examples/` for samples.
