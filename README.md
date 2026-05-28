# TaGra (Table to Graph)

TaGra is an open Python library for building and analysing proximity graphs from
tabular data.  It provides:

1. **Automatic preprocessing** — missing-value handling, scaling, encoding,
   manifold learning.
2. **Pluggable graph constructors** — KNN, distance threshold, cosine similarity,
   and Gower distance for mixed-type data (no preprocessing required).
3. **Density clustering** — a three-stage pipeline (constructor → degree filter →
   connected components) that formally subsumes DBSCAN and extends density
   clustering to non-Euclidean proximity structures.
4. **Graph analysis** — degree distribution, community composition,
   neighbour-class probability.
5. **Machine learning on graphs** — label propagation, graph-aware augmentation,
   anomaly detection, graph-based imputation.

---

## Installation

```sh
pip install tagra
```

---

## Quick start

### Standard graph-building workflow (v0.3.0 API)

```python
import tagra
import pandas as pd

df = pd.read_csv('data.csv')

# Build a k-NN graph and analyse it
graph   = tagra.from_dataframe(df, target='label', method='knn', k=5)
metrics = tagra.analyze(graph, target_attribute='label')
print(f"Homophily: {metrics['homophily_score']:.4f}")
tagra.visualize(graph, output='graph.png')
```

### Density clustering on mixed-type data (Gower constructor)

```python
import networkx as nx
import numpy as np
from tagra.construction import GowerDistanceConstructor

# Raw mixed-type array — no preprocessing needed
X_raw         = ...                                          # shape (n, p)
feature_types = ['continuous', 'binary', 'nominal', ...]    # one entry per column

# Stage 1 — proximity graph
G = nx.Graph()
G.add_nodes_from(range(len(X_raw)))
GowerDistanceConstructor(
    distance_threshold=0.3,
    feature_types=feature_types,
    continuous_metric='gaussian',   # 'range' | 'quadratic' | 'gaussian' | 'laplacian'
).construct(G, X_raw)

# Stage 2 — degree filter δ (generalises DBSCAN's core-point criterion)
delta  = 5
G_filt = G.copy()
G_filt.remove_nodes_from([v for v, d in G.degree() if d < delta])

# Stage 3 — connected components → cluster labels  (−1 = noise)
labels = np.full(len(X_raw), -1)
for cid, comp in enumerate(nx.connected_components(G_filt)):
    for v in comp:
        labels[v] = cid
```

---

## Density clustering

TaGra decouples the two decisions that DBSCAN conflates into a single (ε, m) pair:

| Decision | DBSCAN | TaGra |
|---|---|---|
| Proximity structure | Euclidean ε-ball | Any constructor (KNN, Gower, …) |
| Density threshold | `min_samples` | Degree filter δ |

The three-stage pipeline — **constructor → degree filter → connected components**
— formally subsumes DBSCAN (exact equivalence via `DBSCANGraphConstructor` +
weakly connected components) while generalising it to arbitrary proximity
structures.

### Why Gower distance for EHR data?

Electronic Health Records contain continuous measurements, binary flags, ordinal
scores, and nominal categories.  Euclidean distance is inappropriate because:

- One-hot encoding inflates the weight of high-cardinality nominal features
  (a feature with *c* categories contributes *c* − 1 binary columns, far
  outweighing a single binary flag).
- Standardisation treats ordinal and continuous features identically.

`GowerDistanceConstructor` operates on the **raw feature matrix** without any
preprocessing.  It computes a per-feature partial distance normalised to [0, 1],
averaged across all features, yielding a final distance in [0, 1].
Four kernels for continuous features are supported:
`'range'`, `'quadratic'`, `'gaussian'`, `'laplacian'`.

### Available constructors

| Class | Edge criterion | Notes |
|---|---|---|
| `KNNConstructor` | *k* nearest neighbours (Euclidean) | Adapts to local density |
| `DistanceThresholdConstructor` | Euclidean distance < ε | Equivalent to DBSCAN ε-ball |
| `SimilarityThresholdConstructor` | Cosine similarity > σ | Text / embedding data |
| `GowerDistanceConstructor` | Gower distance ≤ τ | Mixed-type data; no preprocessing |
| `DBSCANGraphConstructor` | DBSCAN directed graph | Formal equivalence check |

```python
from tagra.construction import (
    KNNConstructor,
    DistanceThresholdConstructor,
    SimilarityThresholdConstructor,
    GowerDistanceConstructor,
    DBSCANGraphConstructor,
)
```

---

## API reference

### Convenience functions (recommended)

#### `tagra.from_dataframe()`

```python
import tagra

graph = tagra.from_dataframe(
    df,
    target='label',         # target column name
    method='knn',           # 'knn' | 'distance' | 'similarity'
    k=5,
    scaling='standard',     # 'standard' | 'minmax'
    encoding='one-hot',     # 'one-hot' | 'label'
    manifold_method='UMAP', # 'UMAP' | 'TSNE' | 'Isomap' | None
    manifold_dim=2,
    verbose=True,
)
```

#### `tagra.analyze()`

```python
metrics = tagra.analyze(graph, target_attribute='label')
print(f"Homophily: {metrics['homophily_score']:.4f}")
```

#### `tagra.visualize()`

```python
tagra.visualize(graph, output='graph.png')          # Matplotlib
tagra.visualize(graph, method='cytoscape')           # Jupyter / Cytoscape
```

### Modular API

```python
from tagra.preprocessing import preprocess
from tagra.construction   import build_graph
from tagra.analysis       import analyze

df_processed, positions = preprocess(
    input_dataframe=df,
    target_columns=['label'],
    numeric_scaling='standard',
    categorical_encoding='one-hot',
    manifold_method='UMAP',
    manifold_dim=2,
)
graph   = build_graph(df, df_processed, method='knn', k=5)
metrics = analyze(graph, target_attributes='label')
```

### Machine learning module (`tagra.ml`)

| Class | Purpose |
|---|---|
| `LabelPropagator` | Semi-supervised learning via graph connectivity |
| `GraphAwareAugmenter` | Minority-class augmentation (graph-constrained SMOTE) |
| `GraphAnomalyDetector` | Structural + attribute anomaly detection with explanations |
| `GraphImputer` | Missing-value imputation using neighbour information |

See [Machine learning examples](#machine-learning-examples) below for code.

### Core types (`tagra.core`)

```python
from tagra.core import TaGraGraph, GraphMetadata

graph.number_of_nodes()
graph.number_of_edges()
graph.save('graph.pkl')
loaded = TaGraGraph.load('graph.pkl')
positions = graph.get_positions_dict()
```

### I/O module (`tagra.io`)

```python
from tagra.io import save_graph, load_graph

save_graph(graph, 'graph.graphml')   # GraphML
save_graph(graph, 'graph.pkl')       # Pickle
save_graph(graph, 'graph.cyjs')      # Cytoscape JSON
graph = load_graph('graph.graphml')
```

---

## Machine learning examples

### Semi-supervised learning

```python
from tagra.ml import LabelPropagator

propagator  = LabelPropagator(max_iter=100, alpha=0.8)
predictions = propagator.fit_predict(graph, labels)   # None / −1 for unlabeled
```

### Graph-aware augmentation

```python
from tagra.ml import GraphAwareAugmenter

augmenter        = GraphAwareAugmenter(n_samples=3)
df_aug, y_aug    = augmenter.augment(
    graph,
    X=df[feature_cols],
    y=df['class'],
    target_class='minority',
)
```

### Anomaly detection

```python
from tagra.ml import GraphAnomalyDetector

detector = GraphAnomalyDetector(method='combined', contamination=0.05)
detector.fit(graph, attribute_columns=feature_cols)
for a in detector.get_anomalies(top_k=10):
    print(f"Node {a.node_id}: score={a.anomaly_score:.4f}, reasons={a.reasons}")
```

### Graph-based imputation

```python
from tagra.ml import GraphImputer

imputer    = GraphImputer(strategy='weighted_mean', n_neighbors=5)
df_imputed = imputer.fit_transform(graph, df_with_missing)
```

---

## CLI and configuration file

`go.py` accepts a JSON configuration file:

```sh
python3 go.py -c examples/config.json
python3 go.py -d path/to/data.csv -a class_name
```

Key settings:

| Key | Description | Default |
|---|---|---|
| `input_dataframe` | Path to input data (csv/xlsx/pickle/json/parquet/hdf/h5) | required |
| `output_directory` | Results folder | current dir |
| `target_columns` | Target column(s) for graph colouring and statistics | `[]` |
| `numeric_columns` | Columns to treat as numeric | auto-inferred |
| `categorical_columns` | Columns to treat as categorical | auto-inferred |
| `ignore_columns` | Columns to exclude from processing | `[]` |
| `unknown_column_action` | `'infer'` or `'ignore'` | `'infer'` |
| `numeric_scaling` | `'standard'` or `'minmax'` | `'standard'` |
| `categorical_encoding` | `'one-hot'` or `'label'` | `'one-hot'` |
| `nan_action` | `'drop row'`, `'drop column'`, or `'infer'` | `'infer'` |
| `method` | `'knn'`, `'distance'`, or `'similarity'` | `'knn'` |
| `k` | Neighbours for KNN | `5` |
| `distance_threshold` | Threshold for distance method | — |
| `similarity_threshold` | Threshold for similarity method | — |
| `manifold_method` | `'Isomap'`, `'TSNE'`, `'UMAP'`, or `null` | `null` |
| `graph_visualization_filename` | Path for graph plot (`null` to skip) | `null` |
| `verbose` | Print detailed output | `true` |

---

## Migration from v0.2.x to v0.3.0

The v0.2.x API still works with deprecation warnings; no immediate changes are
required.  Remove deprecated calls before v1.0.0.

**v0.2.x (deprecated — still works)**

```python
from tagra import preprocess_dataframe, create_graph, analyze_graph

df_prep, pos = preprocess_dataframe(df, target_columns=['label'])
G            = create_graph(df_prep, method='knn', k=5)
metrics      = analyze_graph(G, target_attributes='label')
```

**v0.3.0 (recommended)**

```python
import tagra

graph   = tagra.from_dataframe(df, target='label', method='knn', k=5)
metrics = tagra.analyze(graph, target_attribute='label')
tagra.visualize(graph, output='graph.png')
```

---

## Reference

Davide Torre, Davide Chicco, "TaGra: an open Python package for easily generating
graphs from data tables through manifold learning", *PeerJ Computer Science*
11:e2986, 2025. <https://doi.org/10.7717/peerj-cs.2986>

## Contributing

We welcome contributions from the community. Please read our Contributing Guide
for more information on how to get started.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

Open an issue on our GitHub repository if you have questions or need help.

---

Thank you for using TaGra!
