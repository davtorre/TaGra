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
    ├── clustering/             # DBSCAN vs TaGra comparison
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

### Clustering task (active)

Testing the claim that TaGra generalizes DBSCAN for clustering. Key insight: DBSCAN = distance-threshold graph + node degree filter + connected components. TaGra generalizes by allowing KNN/similarity graphs instead of distance-threshold.

Script: `dev/tasks/clustering/clustering_comparison.py`
Config: `dev/tasks/clustering/clustering_config.json`

### Anomaly detection comparison (implemented)

Systematic comparison of sklearn anomaly detectors vs TaGra's `GraphAnomalyDetector`.

Script: `dev/tasks/anomaly/anomaly_task_test.py`

## Configuration

JSON config files. Key parameters:
- `input_dataframe`, `method`, `k`, `distance_threshold`, `similarity_threshold`
- `manifold_method`: "Isomap", "TSNE", "UMAP", or null
- `target_columns`, `unknown_column_action`: "infer" or "ignore"

See `tagra/config.py` for defaults and `examples/` for samples.
