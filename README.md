# TaGra (Table to Graph)

TaGra is a comprehensive Python library designed to simplify the preprocessing of data, the construction of graphs from data tables, and the analysis of those graphs. It provides automated tools for handling missing data, scaling, encoding, manifold learning techniques, graph construction, and machine learning on graph-structured data.

## Scope of TaGra

TaGra achieves four primary objectives:

1. **Automatic Data Preprocessing**: TaGra automates the preprocessing of tabular data, handling missing values, scaling numeric features, encoding categorical variables and manifold learning techniques based on user-defined configurations.

2. **Graph Creation**: TaGra offers three distinct methods to create graphs from the data:
   - **K-Nearest Neighbors (KNN)**: Constructs a graph by connecting each node to its k-nearest neighbors based on Euclidean distance.
   - **Distance Threshold (Radius Graph)**: Connects nodes if the Euclidean distance between them is less than a specified threshold.
   - **Similarity Graph**: Adds an edge between nodes if their cosine similarity exceeds a given threshold.

When creating the graph, each row together with all its features is mapped to a node and an edge between two rows is created using the methods described above.

3. **Graph Analysis**: TaGra provides functions to analyze the generated graphs, including degree distribution, community composition analysis, homophily metrics, and various visualization options.

4. **Machine Learning on Graphs**: TaGra includes powerful ML capabilities that leverage graph structure for semi-supervised learning, data augmentation, anomaly detection, and missing value imputation.

## Migration from v0.2.x to v0.3.0

TaGra v0.3.0 introduces a new modular API while maintaining backward compatibility.

### What Changed

**v0.2.x (Old API - Still Works)**
```python
from tagra import preprocess_dataframe, create_graph, analyze_graph

# Old workflow
df_prep, pos = preprocess_dataframe(df, target_columns=['label'])
G = create_graph(df_prep, method='knn', k=5)
metrics = analyze_graph(G, target_attributes='label')
```

**v0.3.0 (New API - Recommended)**
```python
import tagra

# New workflow - simpler and more intuitive
graph = tagra.from_dataframe(df, target='label', method='knn', k=5)
metrics = tagra.analyze(graph, target_attribute='label')
tagra.visualize(graph, output='graph.png')
```

### New Features in v0.3.0

1. **TaGraGraph Class**: Graphs now use a custom wrapper with metadata
2. **Machine Learning Module**: Label propagation, augmentation, anomaly detection, imputation
3. **Modular Structure**: Organized into subpackages (`tagra.ml`, `tagra.preprocessing`, etc.)
4. **Convenience Functions**: `from_dataframe()`, `analyze()`, `visualize()`

### Breaking Changes

**None!** The old API continues to work with deprecation warnings. You can update at your own pace.

### Migration Strategy

1. **Immediate**: Start using `tagra.from_dataframe()` for new projects
2. **Short-term**: Update existing code to use new convenience functions
3. **Before v1.0.0**: Remove all usage of deprecated functions

## Configuration File

The configuration file is a JSON file that contains all the settings required for preprocessing and graph creation. It's primarily used with the `go.py` CLI script. Below are the key settings:

- `input_dataframe`: Path to the input DataFrame. Supported extensions are csv, xlsx, pickle, json, parquet, hdf, h5.
- `output_directory`: Path to the folder where the results will be collected. If not specified, the current directory is used.
- `preprocessed_filename`: Filename of the preprocessed DataFrame. If not specified, a default name pattern is used.
- `graph_filename`: Filename of the graph file. If not specified, a default name pattern is used.
- `numeric_columns`: List of numeric columns.
- `categorical_columns`: List of categorical columns.
- `target_columns`: List of target columns used for graph coloring and neighborhood statistics.
- `ignore_columns`: List of columns to ignore during preprocessing.
- `unknown_column_action`: Action for unspecified columns. Options: 'infer' or 'ignore'.
- `numeric_threshold`: Threshold for inferring numeric columns.
- `numeric_scaling`: Scaling mode for numeric columns. Options: 'standard' or 'minmax'.
- `categorical_encoding`: Encoding for categorical columns. Options: 'one-hot' or 'label'.
- `nan_action`: Action for NaN values. Options: 'drop row', 'drop column', or 'infer'.
- `nan_threshold`: Threshold for dropping columns based on NaN ratio.
- `verbose`: Flag for detailed output.
- `manifold_method`: Method for manifold learning. Options: 'Isomap' or null.
- `manifold_dimension`: Number of dimensions for manifold learning output.
- `method`: Method to infer the graph. Options: 'knn', 'distance_threshold', or 'similarity'.
- `k`: Number of neighbors for 'knn' method.
- `distance_threshold`: Distance threshold for 'distance_threshold' method.
- `similarity_threshold`: Similarity threshold for 'similarity' method.
- `clustering_method`: Method for clustering analysis. (TODO)
- `inconsistency_threshold`: Threshold for inconsistency in clustering. (TODO)
- `neigh_prob_path`: Filename for neighborhood statistics.
- `prob_heatmap_filename`: Filename for heatmap of neighborhood statistics.
- `degree_distribution_filename`: Filename for degree distribution plot.
- `betweenness_distribution_filename`: Filename for betweenness centrality distribution plot.
- `community_composition_filename`: Filename for community composition histogram.
- `graph_visualization_filename`: Filename for graph visualization. If null, graph is not plotted.

## Functions

### Preprocessing

TaGra provides automatic data preprocessing that includes:

- Handling missing values based on user-defined settings.
- Scaling numeric features using standard or min-max scaling.
- Encoding categorical variables using one-hot or label encoding.
- Inferring the type of unspecified columns based on a threshold.

### Graph Creation

TaGra supports three methods for creating graphs from preprocessed data:

1. **K-Nearest Neighbors (KNN)**:
   - Connects each node to its k-nearest neighbors.
   - Requires the parameter `k` to specify the number of neighbors.

2. **Distance Threshold (Radius Graph)**:
   - Connects nodes if their Euclidean distance is below a specified threshold.
   - Requires the parameter `distance_threshold`.

3. **Similarity Graph**:
   - Adds an edge between nodes if their cosine similarity is above a specified threshold.
   - Requires the parameter `similarity_threshold`.

### Graph Analysis

TaGra includes basic graph analysis functions:

- **Degree Distribution**: Plots the degree distribution of the graph.
- **Community Composition**: Analyzes and plots the composition of communities within the graph.
- **Neighbor class probability**: Evaluates the probability of extracting a node of class $j$ in the neighborhood of a node of class $i$.
## Installation

To install TaGra, simply use pip:

```sh
pip install tagra
```

## Quickstart

### New API (v0.3.0 - Recommended)

```python
import tagra
import pandas as pd

# Load your data
df = pd.read_csv('data.csv')

# Create graph with one line
graph = tagra.from_dataframe(df, target='label', method='knn', k=5)

# Analyze the graph
metrics = tagra.analyze(graph, target_attribute='label')
print(f"Homophily: {metrics['homophily_score']:.4f}")

# Visualize
tagra.visualize(graph, output='graph.png')
```

### Using the CLI

```sh
# With config file
python3 go.py -c examples/config.json

# Direct usage
python3 go.py -d path/to/dataframe.csv -a class_name
```

The CLI will preprocess the data, create a KNN graph, and perform analysis. You can customize options in `examples/config.json`.
# Usage

## Settings

The settings can be specified in a configuration file. It must be a JSON file that contains the settings required for preprocessing and graph creation. Below are the key settings:

- `input_dataframe`: DataFrame path. Supported extensions are: csv, xlsx, pickle, json, parquet, hdf, h5. In the case of a .csv file, the presence of the header will be deduced in the preprocessing part. It is the only mandatory argument.
- `output_directory`: Path to the folder where the results will be collected. If not specified, the path from where the executable was launched will be used. If the folder does not exist, it will be created.
- `preprocessed_filename`: Filename of the preprocessed dataframe. If not specified, a name with this pattern is created: `{basename}_{timestamp}.{ext}` where `{basename}` is the name of the `input_dataframe`, `{timestamp}` is a string in the format ‘%Y%m%d%H%M’ and `{ext}` is the file extension. The supported extensions are the same as for `input_dataframe`.
- `graph_filename`: Filename of the graph file. If not specified, a name with the same pattern as before is created. Supported extension: .graphml.
- `inferred_columns_filename`: Filename for saving the inferred column types. If not specified, it will not be created. Supported extension: .pickle.
- `numeric_columns`: A list containing the numeric columns.
- `categorical_columns`: A list containing the categorical columns.
- `target_columns`: A list containing the "target" variable, used only to color the graph and to evaluate the statistics on the neighborhood in the resulting graph.
- `ignore_columns`: A list containing the columns to be ignored in the preprocessing.
- `unknown_column_action`: An action to deal with columns that have not been specified. Available options: 'infer' (infer how to deal with those columns) or 'ignore' (ignore the columns).
- `numeric_threshold`: Threshold to determine if a column is numeric when `unknown_column_action` is `infer`. If the ratio of unique instances to total rows exceeds this threshold, the column is added to `numeric_columns`; otherwise, to `categorical_columns`.
- `numeric_scaling`: Scaling mode for `numeric_columns`. Available options: 'standard' (Standard Scaler) or 'minmax' (MinMax Scaler). Notice that if a numerical columns must be ignored, it should be added to the list in `ignore_columns`.
- `categorical_encoding`: Encoding for `categorical_columns`. Available options: 'one-hot' (One-Hot-Encoding) or 'label' (Label Encoding).
- `nan_action`: An action to deal with NaN values. Options: 'drop row', 'drop column' or 'infer' (fills with the average).
- `nan_threshold`: If `nan_action` is 'drop column', the column will be dropped if the ratio of NaNs in the column to the total number of rows is greater than this value.
- `verbose`: A flag to print detailed output.
- `manifold_method`: Method for applying manifold learning on `numeric_columns`. Options are `Isomap`, `TSNE`, or None (to avoid manifold learning). The output dimension is always 2 and will be used to visualize the output graph.
- `method`: Method to infer the graph. Available options: 'knn' (make a graph with the k-nearest neighbors based on Euclidean distance), 'distance' (put an edge between nodes if their Euclidean distance is less than `distance_threshold`), 'similarity' (add an edge between two nodes if their cosine similarity is more than `similarity_threshold`).
- `k`: Number of neighbors if method is 'knn'.
- `distance_threshold`: Distance threshold; if the Euclidean distance between two rows is less than the threshold, add an edge between the rows.
- `similarity_threshold`: Similarity threshold; if the cosine similarity between two rows is greater than the threshold, add an edge between the rows.
- `neigh_prob_path`: Filename containing the statistics on the neighbors.
- `degree_distribution_filename`: Filename with the log-log degree distribution plot.
- `community_filename`: Filename with the community distribution histogram.
- `graph_visualization_filename`: Path to the file where the graph visualization will be saved. If null, the graph will not be plotted.
- `prob_heatmap_filename`: Filename of the heatmap containing the statistics on the neighbors.
- `overwrite`: A flag indicating whether to overwrite the results of experiments or not. If set to False, all output filenames are equipped with a timestamp, otherwise outputs are overwritten.
# TaGra API Reference

## New API (v0.3.0 - Recommended)

TaGra v0.3.0 introduces a new modular API with convenience functions for common workflows.

### Convenience Functions

#### `tagra.from_dataframe()`
Main entry point for creating graphs from DataFrames.

```python
import tagra

graph = tagra.from_dataframe(
    df,
    target='label',           # Target column for analysis
    method='knn',             # 'knn', 'distance', or 'similarity'
    k=5,                      # Number of neighbors (for KNN)
    scaling='standard',       # 'standard' or 'minmax'
    encoding='one-hot',       # 'one-hot' or 'label'
    manifold_method='UMAP',   # 'UMAP', 'TSNE', 'Isomap', or None
    manifold_dim=2,
    verbose=True
)
```

#### `tagra.analyze()`
Analyze a TaGra graph.

```python
metrics = tagra.analyze(
    graph,
    target_attribute='label',
    output_directory='results/',
    graph_visualization_filename='graph.png'
)
```

#### `tagra.visualize()`
Visualize a graph.

```python
# Matplotlib visualization
tagra.visualize(graph, output='graph.png', target_attribute='label')

# Cytoscape visualization (in Jupyter)
tagra.visualize(graph, method='cytoscape')
```

### Modular API

For more control, use the modular API directly:

#### Preprocessing Module (`tagra.preprocessing`)

```python
from tagra.preprocessing import preprocess

df_processed, positions = preprocess(
    input_dataframe=df,
    target_columns=['label'],
    numeric_scaling='standard',
    categorical_encoding='one-hot',
    manifold_method='UMAP',
    manifold_dim=2,
    verbose=True
)
```

#### Construction Module (`tagra.construction`)

```python
from tagra.construction import build_graph

graph = build_graph(
    input_dataframe=df,
    preprocessed_dataframe=df_processed,
    method='knn',
    k=5,
    verbose=True
)
```

#### Analysis Module (`tagra.analysis`)

```python
from tagra.analysis import analyze

metrics = analyze(
    graph,
    target_attributes='label',
    verbose=True
)
```

#### Visualization Module (`tagra.visualization`)

```python
from tagra.visualization import matplotlib_graph_visualization

matplotlib_graph_visualization(
    graph,
    target_attribute='label',
    output_path='graph.png',
    pos=positions
)
```

### Machine Learning Module (`tagra.ml`)

TaGra v0.3.0 introduces powerful ML capabilities that leverage graph structure.

#### Semi-Supervised Learning (Label Propagation)

Propagate labels from labeled to unlabeled nodes using graph connectivity.

```python
from tagra.ml import LabelPropagator

# Create a propagator
propagator = LabelPropagator(max_iter=100, alpha=0.8)

# Fit on graph with partial labels
# Use None or -1 for unlabeled nodes
labels = ['A', 'B', None, 'A', None, 'B', ...]
predictions = propagator.fit_predict(graph, labels)

# Or use convenience function
from tagra.ml import propagate_labels
predictions = propagate_labels(graph, target_column='label', unlabeled_value=None)
```

**Use cases:**
- Learning from partially labeled datasets
- Reducing labeling costs
- Exploiting graph structure for better predictions

#### Graph-Aware Data Augmentation

Generate synthetic samples by interpolating between connected nodes, respecting graph structure.

```python
from tagra.ml import GraphAwareAugmenter

augmenter = GraphAwareAugmenter(n_samples=2, alpha_range=(0.3, 0.7))

# Augment specific class
df_augmented, labels_augmented = augmenter.augment(
    graph,
    X=df[['feature1', 'feature2']],
    y=df['label'],
    target_class='minority_class'
)
```

**Advantages over SMOTE:**
- Respects graph connectivity
- Generates samples between truly similar points
- Preserves local structure

#### Explainable Anomaly Detection

Detect outliers based on graph structure with interpretable explanations.

```python
from tagra.ml import GraphAnomalyDetector

detector = GraphAnomalyDetector(
    method='combined',        # 'structural', 'attribute', or 'combined'
    contamination=0.1
)

detector.fit(graph, attribute_columns=['feature1', 'feature2'])
anomalies = detector.get_anomalies()

# Get explanation for specific node
explanation = detector.explain(node_id=42)
print(f"Node {explanation.node_id}: Score = {explanation.anomaly_score:.4f}")
print("Reasons:", explanation.reasons)
```

**Features:**
- Structural anomalies (unusual degree, clustering)
- Attribute anomalies (different from neighbors)
- Combined detection
- Human-readable explanations

#### Graph-Based Imputation

Fill missing values using neighbor information.

```python
from tagra.ml import GraphImputer

imputer = GraphImputer(
    strategy='weighted_mean',  # 'mean', 'weighted_mean', 'median', 'mode'
    n_neighbors=5
)

df_imputed = imputer.fit_transform(graph, df_with_missing)

# Or use convenience function
from tagra.ml import impute_with_graph
df_imputed = impute_with_graph(graph, df_with_missing, strategy='weighted_mean')
```

**Advantages:**
- Uses local neighborhood information
- Better than global statistics
- Respects graph structure

### Core Types (`tagra.core`)

```python
from tagra.core import TaGraGraph, GraphMetadata

# TaGraGraph wraps NetworkX graphs with metadata
graph = TaGraGraph(nx_graph, metadata)

# Access NetworkX functionality
print(graph.number_of_nodes())
print(graph.number_of_edges())

# Save and load
graph.save('graph.pkl')
loaded = TaGraGraph.load('graph.pkl')

# Get positions for visualization
positions = graph.get_positions_dict()
```

### I/O Module (`tagra.io`)

```python
from tagra.io import save_graph, load_graph

# Save in various formats
save_graph(graph, 'graph.graphml')  # GraphML
save_graph(graph, 'graph.pkl')      # Pickle
save_graph(graph, 'graph.cyjs')     # Cytoscape JSON

# Load graphs
graph = load_graph('graph.graphml')
```

## Legacy API (Deprecated)

The old API still works but shows deprecation warnings. It will be removed in v1.0.0.

### 1. Data Preprocessing (Legacy)

```python
from tagra import preprocessing

# Process data with all configuration options
preprocessed_df, manifold_positions = preprocessing.preprocess_dataframe(
    input_dataframe="path/to/dataset.csv",    # Path to file or pandas DataFrame
    output_directory="results/",              # Where to save outputs
    preprocessed_filename=None,               # Custom filename for processed data
    inferred_columns_filename=None,           # Save column types in pickle file
    numeric_columns=[],                       # Columns to treat as numeric
    categorical_columns=[],                   # Columns to treat as categorical
    target_columns=["target"],                # Target variable(s) for analysis
    unknown_column_action='infer',            # How to handle unspecified columns
    ignore_columns=[],                        # Columns to exclude from processing
    numeric_threshold=0.05,                   # Threshold for numeric inference
    numeric_scaling='standard',               # 'standard' or 'minmax' scaling
    categorical_encoding='one-hot',           # 'one-hot' or 'label' encoding
    nan_action='infer',                       # How to handle missing values
    nan_threshold=0.5,                        # Threshold for column removal
    verbose=True,                             # Print processing details
    manifold_method=None,                     # 'Isomap', 'TSNE', or 'UMAP' 
    manifold_dim=2,                           # Dimensions for manifold learning
    overwrite=False                           # Overwrite existing files
)

# Minimal usage with auto-inference
preprocessed_df, _ = preprocessing.preprocess_dataframe(
    input_dataframe="path/to/dataset.csv",
    target_columns=["target"]
)

```

Returns:

- ```preprocessed_df```: Processed pandas DataFrame with encoded/scaled features
- ```manifold_positions```: Coordinates from manifold learning (if applied) for visualization

### List of optional arguments and their default values
```python
output_directory = "results/", 
preprocessed_filename = None,
inferred_columns_filename = None, 
numeric_columns = [],
categorical_columns = [], 
target_columns = [], 
unknown_column_action = 'infer',
ignore_columns = [], 
numeric_threshold = 0.05,
numeric_scaling = 'standard', 
categorical_encoding = 'one-hot',
nan_action = 'infer', 
nan_threshold = 0.5,
verbose = True, 
manifold_method = None, 
manifold_dim = None,
overwrite = False
```

## 2.Graph Creation Module

```python
from tagra import graph

# Create a graph with default KNN method
G = graph.create_graph(
    preprocessed_dataframe="path/to/preprocessed.csv",  # Processed data from preprocessing step
    output_directory="results/",                        # Where to save the graph
    method="knn",                                       # Graph creation method
    k=5                                                 # Number of neighbors for KNN
)

# Create a distance-based graph
G = graph.create_graph(
    input_dataframe="dataset.csv",                     # Raw input data
    numeric_columns=["feature1", "feature2"],          # Columns to use for graph building
    output_directory="results/",
    method="distance",                                 # Use distance threshold method
    distance_threshold=0.5,                            # Maximum distance for edge creation
    graph_filename="distance_graph.graphml"            # Custom filename
)

# Create a similarity-based graph
G = graph.create_graph(
    preprocessed_dataframe=df_preprocessed,            # Pass DataFrame directly
    method="similarity",                               # Use similarity threshold method
    similarity_threshold=0.7,                          # Minimum similarity for edge creation
    verbose=True                                       # Print detailed progress
)
```
Returns a NetworkX graph object with:
- Nodes representing data points
- Node attributes containing original data values
- Edges representing relationships based on the chosen method


### List of optional arguments and their default values
```python
preprocessed_dataframe=None,
inferred_columns_filename=None,
numeric_columns=None,
output_directory=None,
graph_filename=None,
method='knn',
k=5,
distance_threshold=None,
similarity_threshold=None,
verbose=True,
overwrite=False
```
## 3. Graph Analysis Module

```python
from tagra import analysis

# Analyze graph with all visualization outputs
analysis.analyze_graph(
    graph="path/to/graph.graphml",                   # Path to saved graph or NetworkX graph object
    target_attributes="target_column",               # Target variable for coloring and analysis
    verbose=True,                                    # Print analysis details
    pos=None,                                        # Optional node positions for visualization
    output_directory="results/",                     # Where to save analysis outputs
    neigh_prob_filename="neigh_prob.txt",            # Save neighborhood probabilities as text
    degree_distribution_filename="degree.png",       # Save degree distribution plot
    prob_heatmap_filename="prob_heatmap.png",        # Save probability heatmap visualization
    community_filename="communities.png",            # Save community composition histogram
    graph_visualization_filename="graph.png",        # Save graph visualization
    overwrite=False                                  # Whether to overwrite existing files
)

# Basic usage with a NetworkX graph
import networkx as nx
G = nx.Graph()  # A graph from previous steps
analysis.analyze_graph(G, target_attributes="class")

# Use node positions from manifold learning
from tagra import preprocessing
df, manifold_positions = preprocessing.preprocess_dataframe(
    input_dataframe="data.csv", 
    manifold_method="Isomap"
)
analysis.analyze_graph(
    graph=G,
    pos=manifold_positions,  # Use positions from manifold learning
    graph_visualization_filename="manifold_graph.png"
)
```
**Generated outputs**:

- Neighborhood analysis: Probability matrix showing the likelihood of finding neighbors with different target attributes

- Text output with P(j|i) values indicating probability of neighbors having class j given node class i
Heatmap visualization where diagonal elements approaching 1 indicate strong class separation

- Degree distribution plot: Log-log plot showing the number of connections per node. Helps identify potential outliers (nodes with few connections)
Reveals central nodes (nodes with many connections)

- Community composition visualization: Histogram showing how target attributes are distributed across detected communities. Uses the Girvan-Newman algorithm for community detection
Bars colored according to the class distribution within each community

- Graph visualization: 2D visualization of the graph with nodes colored by target attribute

- Uses manifold learning coordinates or force-directed layout
Reveals clusters, isolated nodes, and connectivity patterns




### List of optional arguments and their default values
```python
target_attributes=None, 
verbose=True,
pos=None,
output_directory=None,
neigh_prob_filename = None,
degree_distribution_filename = None,
prob_heatmap_filename = None,
community_filename = None,
graph_visualization_filename = None,
overwrite = False
```
# Machine Learning Examples

TaGra v0.3.0's ML module enables powerful graph-based machine learning workflows.

## Example 1: Semi-Supervised Learning

Learn from a small labeled dataset using graph structure.

```python
import tagra
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('partially_labeled_data.csv')

# Create graph
graph = tagra.from_dataframe(df, target='label', method='knn', k=5)

# Prepare labels (None for unlabeled)
labels = df['label'].values  # Contains None for unlabeled samples

# Propagate labels
from tagra.ml import LabelPropagator
propagator = LabelPropagator(max_iter=100, alpha=0.8, verbose=True)
predictions = propagator.fit_predict(graph, labels)

# Evaluate on unlabeled data
unlabeled_mask = df['label'].isna()
print(f"Predicted {unlabeled_mask.sum()} labels")
```

## Example 2: Handling Imbalanced Data

Augment minority class using graph structure.

```python
import tagra
from tagra.ml import GraphAwareAugmenter

# Create graph
graph = tagra.from_dataframe(df, target='class', method='knn', k=5)

# Augment minority class
augmenter = GraphAwareAugmenter(n_samples=3, verbose=True)
df_aug, labels_aug = augmenter.augment(
    graph,
    X=df[['feature1', 'feature2', 'feature3']],
    y=df['class'],
    target_class='minority'
)

print(f"Original samples: {len(df)}")
print(f"Augmented samples: {len(df_aug)}")
```

## Example 3: Anomaly Detection

Detect and explain anomalies in your data.

```python
import tagra
from tagra.ml import GraphAnomalyDetector

# Create graph
graph = tagra.from_dataframe(df, target='label', method='knn', k=5)

# Detect anomalies
detector = GraphAnomalyDetector(
    method='combined',
    contamination=0.05,  # Expect 5% anomalies
    verbose=True
)

detector.fit(graph, attribute_columns=['feature1', 'feature2'])

# Get top anomalies
anomalies = detector.get_anomalies(top_k=10)

for anomaly in anomalies:
    print(f"\nNode {anomaly.node_id}:")
    print(f"  Score: {anomaly.anomaly_score:.4f}")
    print(f"  Reasons: {', '.join(anomaly.reasons)}")
```

## Example 4: Missing Value Imputation

Fill missing values using neighbor information.

```python
import tagra
from tagra.ml import GraphImputer

# Create graph from data with missing values
df_with_missing = pd.read_csv('data_with_nans.csv')

# Build graph using available features
graph = tagra.from_dataframe(
    df_with_missing.dropna(),  # Use complete cases for graph
    method='knn',
    k=5
)

# Impute missing values
imputer = GraphImputer(strategy='weighted_mean', n_neighbors=5, verbose=True)
df_imputed = imputer.fit_transform(graph, df_with_missing)

print(f"Missing values before: {df_with_missing.isna().sum().sum()}")
print(f"Missing values after: {df_imputed.isna().sum().sum()}")
```

## Example 5: Complete ML Workflow

Combine multiple ML features for a complete workflow.

```python
import tagra
from tagra.ml import GraphImputer, LabelPropagator, GraphAwareAugmenter
import pandas as pd

# 1. Load data with missing values and partial labels
df = pd.read_csv('messy_data.csv')

# 2. Impute missing values
graph = tagra.from_dataframe(df, method='knn', k=5)
imputer = GraphImputer(strategy='weighted_mean')
df_clean = imputer.fit_transform(graph, df)

# 3. Rebuild graph with clean data
graph = tagra.from_dataframe(df_clean, target='label', method='knn', k=5)

# 4. Propagate labels
propagator = LabelPropagator(max_iter=100, alpha=0.8)
df_clean['label_predicted'] = propagator.fit_predict(graph, df_clean['label'])

# 5. Augment minority class
augmenter = GraphAwareAugmenter(n_samples=2)
df_balanced, labels_balanced = augmenter.augment(
    graph,
    X=df_clean.drop(columns=['label', 'label_predicted']),
    y=df_clean['label_predicted'],
    target_class='minority'
)

# 6. Analyze final graph
final_graph = tagra.from_dataframe(df_balanced, target='label', method='knn', k=5)
metrics = tagra.analyze(final_graph, target_attribute='label')
tagra.visualize(final_graph, output='final_graph.png')

print(f"Final dataset: {len(df_balanced)} samples")
print(f"Homophily: {metrics['homophily_score']:.4f}")
```

# Reference
Davide Torre, Davide Chicco, "TaGra: an open Python package for easily generating graphs from data tables through manifold learning", PeerJ Computer Science 11:e2986, 2025. https://doi.org/10.7717/peerj-cs.2986

# Contributing
We welcome contributions from the community. If you would like to contribute, please read our Contributing Guide for more information on how to get started.

# License
This project is licensed under the MIT License. See the LICENSE file for more details.

# Support
If you have any questions or need help, please feel free to open an issue on our GitHub repository.

---
Thank you for using TaGra! We hope it makes your data preprocessing and graph analysis tasks easier and more efficient.
