# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TaGra (Table to Graph) v0.3.0 is a Python library for preprocessing tabular data, constructing graphs from it, and performing machine learning on graph-structured data. The library provides four main capabilities:

1. **Data Preprocessing**: Handling missing values, scaling numeric features, encoding categorical variables, and manifold learning (Isomap, TSNE, UMAP)
2. **Graph Creation**: Three methods - KNN (k-nearest neighbors), distance threshold, and similarity threshold
3. **Graph Analysis**: Degree distribution, community detection (Girvan-Newman), neighborhood probability analysis, homophily metrics, and various visualizations
4. **Machine Learning on Graphs**: Semi-supervised learning (label propagation), graph-aware data augmentation, explainable anomaly detection, and graph-based imputation

## Build & Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Run tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_preprocessing.py

# Run the main pipeline with a config file
python go.py -c examples/article/takashi_similarity.json

# Run with a dataframe directly (uses default config)
python go.py -d path/to/dataframe.csv -a target_column_name
```

## Architecture

TaGra v0.3.0 uses a modular architecture with separate subpackages for different functionality.

### Package Structure (`tagra/`)

```
tagra/
├── __init__.py           # Main API: from_dataframe(), analyze(), visualize()
├── core/                 # Core types and classes
│   ├── graph.py         # TaGraGraph class
│   ├── metadata.py      # GraphMetadata dataclass
│   └── enums.py         # ScalingMethod, EncodingMethod, etc.
├── preprocessing/        # Data preprocessing
│   ├── pipeline.py      # PreprocessingPipeline class
│   └── preprocess.py    # preprocess() function
├── construction/         # Graph construction
│   ├── builders.py      # build_graph() function
│   ├── constructors.py  # KNNConstructor, DistanceThresholdConstructor, etc.
│   └── registry.py      # ConstructorRegistry for extensibility
├── analysis/            # Graph analysis
│   ├── analyze.py       # analyze() function
│   ├── metrics.py       # Metric computation
│   └── community.py     # Community detection
├── visualization/       # Plotting and visualization
│   ├── matplotlib_vis.py
│   └── plotly_vis.py
├── ml/                  # Machine learning (NEW in v0.3.0)
│   ├── label_propagation.py  # LabelPropagator class
│   ├── augmentation.py       # GraphAwareAugmenter class
│   ├── anomaly.py           # GraphAnomalyDetector, AnomalyExplanation
│   └── imputation.py        # GraphImputer class
├── io/                  # File I/O
│   ├── readers.py
│   └── writers.py
├── compat/              # Legacy API compatibility layer
│   └── legacy.py        # preprocess_dataframe(), create_graph(), analyze_graph()
├── exceptions.py        # Custom exceptions
├── config.py            # Configuration handling
└── cytoscape_vis.py     # Optional Cytoscape support

go.py                     # CLI entry point
```

### Core Modules

#### 1. Core (`tagra.core`)
- **TaGraGraph**: Wrapper around NetworkX graphs with metadata
  - Stores construction parameters, preprocessing info
  - Provides `.save()` and `.load()` methods
  - Maintains manifold positions for visualization
- **GraphMetadata**: Dataclass for graph metadata
- **Enums**: Type-safe enumerations for methods

#### 2. Preprocessing (`tagra.preprocessing`)
- **preprocess()**: Main preprocessing function
  - File loading (CSV, Excel, pickle, JSON, parquet, HDF5)
  - NaN handling (drop, infer, fill)
  - Column type inference
  - Scaling (StandardScaler/MinMaxScaler)
  - Encoding (one-hot/label)
  - Manifold learning (Isomap, TSNE, UMAP)
- **PreprocessingPipeline**: Reusable preprocessing pipeline

#### 3. Construction (`tagra.construction`)
- **build_graph()**: Creates TaGraGraph from data
- **Constructors**:
  - `KNNConstructor`: Uses scipy's cKDTree for k-nearest neighbors
  - `DistanceThresholdConstructor`: Uses cKDTree.query_pairs
  - `SimilarityThresholdConstructor`: Uses sklearn's cosine_similarity
- **ConstructorRegistry**: Register custom graph construction methods

#### 4. Analysis (`tagra.analysis`)
- **analyze()**: Comprehensive graph analysis
  - Density, clustering coefficient, connected components
  - Assortativity, community detection (Girvan-Newman)
  - Chi-square tests, homophily analysis (with permutation testing)
  - Generates visualizations

#### 5. Visualization (`tagra.visualization`)
- `matplotlib_graph_visualization()`: Static plots
- `plotly_graph_visualization()`: Interactive plots
- `heat_map_prob()`: Neighborhood probability heatmaps
- `plot_distribution()`: Degree/betweenness distributions
- `plot_community_composition()`: Community analysis plots

#### 6. Machine Learning (`tagra.ml`) - NEW in v0.3.0

- **LabelPropagator**: Semi-supervised learning
  - Propagates labels from labeled to unlabeled nodes
  - Uses graph structure for better predictions
  - Includes `propagate_labels()` convenience function

- **GraphAwareAugmenter**: Data augmentation
  - Generates synthetic samples using graph structure
  - Interpolates between connected nodes
  - Better than SMOTE for graph-structured data

- **GraphAnomalyDetector**: Anomaly detection
  - Detects structural and attribute anomalies
  - Provides `AnomalyExplanation` with human-readable reasons
  - Combines multiple anomaly signals

- **GraphImputer**: Missing value imputation
  - Fills missing values using neighbor information
  - Weighted by graph proximity
  - Includes `impute_with_graph()` convenience function

#### 7. I/O (`tagra.io`)
- Reading/writing graphs in multiple formats
- Export to Cytoscape (CYJS), GraphML, pickle
- DataFrame I/O utilities

#### 8. Compatibility Layer (`tagra.compat`)
- **Legacy API** (deprecated but functional):
  - `preprocess_dataframe()` → delegates to new `preprocess()`
  - `create_graph()` → delegates to new `build_graph()`
  - `analyze_graph()` → delegates to new `analyze()`
- Shows deprecation warnings
- Maintains full backward compatibility

### Entry Points

#### Python API (Recommended)
```python
import tagra

# One-line graph creation
graph = tagra.from_dataframe(df, target='label', method='knn', k=5)

# Analysis and visualization
metrics = tagra.analyze(graph, target_attribute='label')
tagra.visualize(graph, output='graph.png')
```

#### CLI
- **go.py**: Orchestrates full pipeline (preprocessing → graph creation → analysis → visualization)
  - `-c config.json`: Use configuration file
  - `-d dataframe.csv -a target`: Direct usage with defaults

### Data Flow

#### New API (v0.3.0)
1. Raw DataFrame → `preprocess()` → preprocessed DataFrame + manifold positions
2. DataFrames → `build_graph()` → TaGraGraph (with metadata)
3. TaGraGraph → `analyze()` → metrics dict
4. TaGraGraph → `visualize()` → plots and visualizations

#### Legacy API (v0.2.x - deprecated)
1. Raw data → `preprocess_dataframe()` → preprocessed DataFrame + positions
2. Preprocessed DataFrame → `create_graph()` → NetworkX Graph
3. NetworkX Graph → `analyze_graph()` → metrics dict + visualization files

## Configuration

Configuration is via JSON files. Key parameters:
- `input_dataframe`: Path to input data (required)
- `method`: Graph method - "knn", "distance", or "similarity"
- `k`, `distance_threshold`, `similarity_threshold`: Method-specific parameters
- `manifold_method`: "Isomap", "TSNE", "UMAP", or null
- `target_columns`: Column(s) for coloring and neighborhood analysis
- `unknown_column_action`: "infer" or "ignore" for unspecified columns

See `tagra/config.py` for all defaults and `examples/` for sample configurations.

## Key Dependencies

- **pandas, numpy**: Data handling
- **scikit-learn**: Preprocessing (scalers, encoders), manifold learning, and ML utilities
- **networkx**: Graph construction and analysis
- **scipy**: Spatial operations (cKDTree for KNN, cosine_similarity)
- **matplotlib, plotly**: Visualization
- **umap-learn**: UMAP manifold learning (optional, lazy-imported)
- **ipycytoscape**: Interactive Cytoscape visualization (optional)

## Quick API Examples

### New API (v0.3.0)

```python
import tagra
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Create graph
graph = tagra.from_dataframe(df, target='label', method='knn', k=5)

# Analyze
metrics = tagra.analyze(graph, target_attribute='label')
print(f"Homophily: {metrics['homophily_score']:.4f}")

# Visualize
tagra.visualize(graph, output='graph.png')

# Machine Learning
from tagra.ml import LabelPropagator, GraphAwareAugmenter

# Semi-supervised learning
propagator = LabelPropagator(alpha=0.8)
predictions = propagator.fit_predict(graph, labels)

# Data augmentation
augmenter = GraphAwareAugmenter(n_samples=2)
X_aug, y_aug = augmenter.augment(graph, X, y, target_class='minority')
```

### Modular API

```python
from tagra.preprocessing import preprocess
from tagra.construction import build_graph
from tagra.analysis import analyze

# Step by step
df_processed, positions = preprocess(df, target_columns=['label'])
graph = build_graph(df, df_processed, method='knn', k=5)
metrics = analyze(graph, target_attributes='label')
```

### Legacy API (deprecated)

```python
from tagra import preprocess_dataframe, create_graph, analyze_graph

# Old workflow (still works, shows deprecation warnings)
df_prep, pos = preprocess_dataframe(df, target_columns=['label'])
G = create_graph(df_prep, method='knn', k=5)
metrics = analyze_graph(G, target_attributes='label')
```
