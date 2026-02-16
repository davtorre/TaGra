# TaGra Code Restructuring Plan

## Executive Summary

This document outlines a comprehensive plan to restructure TaGra's codebase from its current monolithic design to a modular, extensible architecture. The restructuring will facilitate the addition of new ML capabilities while maintaining backward compatibility with existing users.

---

## Part 1: Current State Analysis

### 1.1 Current Directory Structure

```
tagra/
├── __init__.py          # Empty - no public API defined
├── config.py            # Configuration loading (~100 lines)
├── preprocessing.py     # Single function (~400 lines)
├── graph.py             # Single function (~200 lines)
├── analysis.py          # Single function (~400 lines)
└── utils.py             # Mixed utilities (~300 lines)
```

### 1.2 Problems with Current Structure

#### Problem 1: Monolithic Functions

Each module contains one giant function with 15-25 parameters:

```python
# Current: preprocessing.py
def preprocess_dataframe(
    input_dataframe=None,
    output_directory="results/",
    preprocessed_filename=None,
    inferred_columns_filename=None,
    numeric_columns=[],
    categorical_columns=[],
    target_columns=[],
    unknown_column_action='infer',
    ignore_columns=[],
    numeric_threshold=0.05,
    numeric_scaling='standard',
    categorical_encoding='one-hot',
    nan_action='infer',
    nan_threshold=0.5,
    verbose=True,
    manifold_method='UMAP',
    manifold_dim=2,
    overwrite=False
):
    # 400 lines of mixed concerns...
```

**Consequences:**
- Difficult to test individual components
- Cannot reuse parts of the logic
- Adding new functionality requires modifying existing code
- High cognitive load for contributors

#### Problem 2: `utils.py` as Catch-All

`utils.py` currently contains:
- Neighborhood analysis functions
- Probability printing/formatting
- Heatmap visualization
- Distribution plotting
- Community composition plotting
- Graph visualization

These have different responsibilities and should be separated.

#### Problem 3: No Extensibility Points

Adding a new graph construction method currently requires:
1. Modifying `graph.py`
2. Adding new parameters to the function signature
3. Adding conditional logic inside the function
4. Updating all callers

This violates the Open/Closed Principle.

#### Problem 4: No Public API

The empty `__init__.py` means users must know internal structure:

```python
# Current usage - exposes internal structure
from tagra.preprocessing import preprocess_dataframe
from tagra.graph import create_graph
from tagra.analysis import analyze_graph
```

#### Problem 5: No Home for Future Modules

Where would these go?
- `label_propagation.py` - Not preprocessing, not graph creation, not analysis
- `augmentation.py` - Creates new data, doesn't fit existing categories
- `longitudinal.py` - Requires multiple graphs over time

---

## Part 2: Proposed Architecture

### 2.1 New Directory Structure

```
tagra/
├── __init__.py                    # Clean public API
├── config.py                      # Configuration (keep as-is, minor refactor)
├── exceptions.py                  # Custom exceptions
│
├── core/                          # Core data structures and types
│   ├── __init__.py
│   ├── graph.py                   # TaGraGraph class
│   └── types.py                   # Type definitions, enums
│
├── preprocessing/                 # Data preprocessing
│   ├── __init__.py                # Exports preprocess() function
│   ├── pipeline.py                # PreprocessingPipeline class
│   ├── scaling.py                 # NumericScaler classes
│   ├── encoding.py                # CategoricalEncoder classes
│   ├── missing.py                 # MissingValueHandler classes
│   └── manifold.py                # ManifoldReducer classes
│
├── construction/                  # Graph construction
│   ├── __init__.py                # Exports build_graph() function
│   ├── base.py                    # Abstract GraphConstructor
│   ├── knn.py                     # KNNConstructor
│   ├── distance.py                # DistanceThresholdConstructor
│   ├── similarity.py              # SimilarityThresholdConstructor
│   └── registry.py                # Constructor registry for extensibility
│
├── analysis/                      # Graph analysis
│   ├── __init__.py                # Exports analyze() function
│   ├── metrics.py                 # Structural metrics
│   ├── neighborhood.py            # Neighborhood analysis
│   ├── community.py               # Community detection
│   └── report.py                  # Report generation
│
├── visualization/                 # Visualization (separated)
│   ├── __init__.py
│   ├── graph_plot.py              # Graph visualization
│   ├── heatmap.py                 # Probability heatmaps
│   ├── distribution.py            # Degree distribution
│   └── community_plot.py          # Community composition
│
├── ml/                            # Machine learning applications
│   ├── __init__.py
│   ├── label_propagation.py       # Semi-supervised learning
│   ├── augmentation.py            # Data augmentation
│   ├── anomaly.py                 # Anomaly detection
│   └── imputation.py              # Missing value imputation
│
├── longitudinal/                  # Longitudinal analysis
│   ├── __init__.py
│   ├── tracker.py                 # Multi-timepoint tracking
│   ├── metrics.py                 # Stability, drift metrics
│   └── transitions.py             # Transition probability estimation
│
├── io/                            # Input/Output operations
│   ├── __init__.py
│   ├── readers.py                 # DataFrame readers
│   ├── writers.py                 # Graph/results writers
│   └── exporters.py               # Cytoscape, Gephi export
│
└── compat/                        # Backward compatibility layer
    ├── __init__.py
    └── legacy.py                  # Old function signatures
```

### 2.2 Core Classes and Interfaces

#### 2.2.1 TaGraGraph (core/graph.py)

A wrapper around NetworkX that stores metadata and provides convenience methods:

```python
# core/graph.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import networkx as nx
import pandas as pd


@dataclass
class GraphMetadata:
    """Metadata about how the graph was constructed."""
    construction_method: str
    construction_params: Dict[str, Any]
    source_dataframe_shape: tuple
    numeric_columns: List[str]
    target_columns: List[str]
    created_at: str
    manifold_positions: Optional[Dict[int, tuple]] = None


class TaGraGraph:
    """
    A graph representation of tabular data with associated metadata.

    Wraps a NetworkX graph and provides convenience methods for
    TaGra-specific operations.
    """

    def __init__(self, graph: nx.Graph, metadata: GraphMetadata):
        self._graph = graph
        self._metadata = metadata

    @property
    def graph(self) -> nx.Graph:
        """Access the underlying NetworkX graph."""
        return self._graph

    @property
    def metadata(self) -> GraphMetadata:
        """Access graph metadata."""
        return self._metadata

    @property
    def n_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self._graph.number_of_edges()

    def get_node_features(self, node_id: int) -> Dict[str, Any]:
        """Get all features for a specific node."""
        return dict(self._graph.nodes[node_id])

    def get_neighbors(self, node_id: int) -> List[int]:
        """Get neighbor node IDs."""
        return list(self._graph.neighbors(node_id))

    def get_neighborhood_labels(self, node_id: int, target_col: str) -> List[Any]:
        """Get target labels of all neighbors."""
        return [
            self._graph.nodes[n].get(target_col)
            for n in self.get_neighbors(node_id)
        ]

    def subgraph(self, node_ids: List[int]) -> 'TaGraGraph':
        """Extract a subgraph containing only specified nodes."""
        subg = self._graph.subgraph(node_ids).copy()
        return TaGraGraph(subg, self._metadata)

    def to_networkx(self) -> nx.Graph:
        """Export as plain NetworkX graph."""
        return self._graph.copy()

    def save(self, path: str) -> None:
        """Save graph to file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({'graph': self._graph, 'metadata': self._metadata}, f)

    @classmethod
    def load(cls, path: str) -> 'TaGraGraph':
        """Load graph from file."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(data['graph'], data['metadata'])
```

#### 2.2.2 Abstract Base Classes

```python
# construction/base.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
import networkx as nx


class GraphConstructor(ABC):
    """Abstract base class for graph construction methods."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the method name (e.g., 'knn', 'distance', 'similarity')."""
        pass

    @abstractmethod
    def construct(
        self,
        data: pd.DataFrame,
        numeric_columns: list
    ) -> nx.Graph:
        """
        Construct a graph from preprocessed data.

        Args:
            data: Preprocessed DataFrame
            numeric_columns: Columns to use for distance/similarity calculation

        Returns:
            NetworkX graph with nodes containing all original attributes
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """Return construction parameters for metadata."""
        return {}


# preprocessing/base.py
class Transformer(ABC):
    """Abstract base class for data transformers."""

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'Transformer':
        """Fit the transformer to data."""
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        pass

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)
```

#### 2.2.3 Concrete Implementations

```python
# construction/knn.py
from typing import Dict, Any
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree

from .base import GraphConstructor


class KNNConstructor(GraphConstructor):
    """Construct graph by connecting each node to its k nearest neighbors."""

    def __init__(self, k: int = 5):
        """
        Args:
            k: Number of nearest neighbors to connect
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k

    @property
    def name(self) -> str:
        return "knn"

    def construct(
        self,
        data: pd.DataFrame,
        numeric_columns: list
    ) -> nx.Graph:
        G = nx.Graph()

        # Add nodes with all attributes
        for i, row in data.iterrows():
            G.add_node(i, **row.to_dict())

        # Build KD-tree for efficient neighbor queries
        values = data[numeric_columns].values
        tree = cKDTree(values)

        # Connect each node to k nearest neighbors
        for i in G.nodes():
            distances, indices = tree.query(values[i], k=self.k + 1)
            for j in indices[1:]:  # Skip self (index 0)
                G.add_edge(i, int(j))

        return G

    def get_params(self) -> Dict[str, Any]:
        return {"k": self.k}


# construction/distance.py
class DistanceThresholdConstructor(GraphConstructor):
    """Construct graph by connecting nodes within a distance threshold."""

    def __init__(self, threshold: float):
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "distance"

    def construct(
        self,
        data: pd.DataFrame,
        numeric_columns: list
    ) -> nx.Graph:
        G = nx.Graph()

        for i, row in data.iterrows():
            G.add_node(i, **row.to_dict())

        values = data[numeric_columns].values
        tree = cKDTree(values)
        pairs = tree.query_pairs(self.threshold)

        for i, j in pairs:
            G.add_edge(i, j)

        return G

    def get_params(self) -> Dict[str, Any]:
        return {"threshold": self.threshold}


# construction/similarity.py
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class SimilarityThresholdConstructor(GraphConstructor):
    """Construct graph by connecting nodes above a similarity threshold."""

    def __init__(self, threshold: float = 0.8):
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "similarity"

    def construct(
        self,
        data: pd.DataFrame,
        numeric_columns: list
    ) -> nx.Graph:
        G = nx.Graph()

        for i, row in data.iterrows():
            G.add_node(i, **row.to_dict())

        values = data[numeric_columns].values
        sim_matrix = cosine_similarity(values)

        # Zero out diagonal and values below threshold
        np.fill_diagonal(sim_matrix, 0)
        rows, cols = np.where(sim_matrix >= self.threshold)

        for i, j in zip(rows, cols):
            if i < j:  # Avoid duplicate edges
                G.add_edge(int(i), int(j))

        return G

    def get_params(self) -> Dict[str, Any]:
        return {"threshold": self.threshold}
```

#### 2.2.4 Constructor Registry (Extensibility)

```python
# construction/registry.py
from typing import Dict, Type
from .base import GraphConstructor
from .knn import KNNConstructor
from .distance import DistanceThresholdConstructor
from .similarity import SimilarityThresholdConstructor


class ConstructorRegistry:
    """Registry for graph construction methods. Enables extensibility."""

    _constructors: Dict[str, Type[GraphConstructor]] = {}

    @classmethod
    def register(cls, name: str, constructor_class: Type[GraphConstructor]):
        """Register a new constructor."""
        cls._constructors[name] = constructor_class

    @classmethod
    def get(cls, name: str) -> Type[GraphConstructor]:
        """Get a constructor class by name."""
        if name not in cls._constructors:
            raise ValueError(
                f"Unknown constructor: {name}. "
                f"Available: {list(cls._constructors.keys())}"
            )
        return cls._constructors[name]

    @classmethod
    def list_available(cls) -> list:
        """List all available constructors."""
        return list(cls._constructors.keys())


# Register built-in constructors
ConstructorRegistry.register("knn", KNNConstructor)
ConstructorRegistry.register("distance", DistanceThresholdConstructor)
ConstructorRegistry.register("similarity", SimilarityThresholdConstructor)
```

**Usage for extensibility:**

```python
# User can add custom constructors
from tagra.construction import ConstructorRegistry, GraphConstructor

class MyCustomConstructor(GraphConstructor):
    # ... implementation
    pass

ConstructorRegistry.register("my_custom", MyCustomConstructor)
```

#### 2.2.5 Preprocessing Pipeline

```python
# preprocessing/pipeline.py
from typing import List, Optional, Tuple
import pandas as pd

from .scaling import NumericScaler, StandardScaler, MinMaxScaler
from .encoding import CategoricalEncoder, OneHotEncoder, LabelEncoderWrapper
from .missing import MissingValueHandler
from .manifold import ManifoldReducer


class PreprocessingPipeline:
    """
    Configurable preprocessing pipeline for tabular data.

    Handles:
    - Missing value imputation
    - Numeric scaling
    - Categorical encoding
    - Optional manifold learning for visualization
    """

    def __init__(
        self,
        numeric_scaler: Optional[NumericScaler] = None,
        categorical_encoder: Optional[CategoricalEncoder] = None,
        missing_handler: Optional[MissingValueHandler] = None,
        manifold_reducer: Optional[ManifoldReducer] = None
    ):
        self.numeric_scaler = numeric_scaler or StandardScaler()
        self.categorical_encoder = categorical_encoder or OneHotEncoder()
        self.missing_handler = missing_handler or MissingValueHandler()
        self.manifold_reducer = manifold_reducer

        self._numeric_columns: List[str] = []
        self._categorical_columns: List[str] = []
        self._is_fitted = False

    def fit(
        self,
        data: pd.DataFrame,
        numeric_columns: List[str],
        categorical_columns: List[str],
        target_columns: List[str] = None
    ) -> 'PreprocessingPipeline':
        """Fit the pipeline to data."""
        self._numeric_columns = numeric_columns
        self._categorical_columns = categorical_columns
        self._target_columns = target_columns or []

        # Fit each component
        if self._numeric_columns:
            self.numeric_scaler.fit(data[self._numeric_columns])

        if self._categorical_columns:
            self.categorical_encoder.fit(data[self._categorical_columns])

        self._is_fitted = True
        return self

    def transform(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[dict]]:
        """
        Transform data through the pipeline.

        Returns:
            Tuple of (preprocessed_dataframe, manifold_positions)
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")

        df = data.copy()

        # Handle missing values
        df = self.missing_handler.transform(df)

        # Scale numeric columns
        if self._numeric_columns:
            df[self._numeric_columns] = self.numeric_scaler.transform(
                df[self._numeric_columns]
            )

        # Encode categorical columns
        if self._categorical_columns:
            encoded = self.categorical_encoder.transform(df[self._categorical_columns])
            df = df.drop(columns=self._categorical_columns)
            df = pd.concat([df, encoded], axis=1)

        # Apply manifold learning if configured
        manifold_positions = None
        if self.manifold_reducer is not None:
            coords = self.manifold_reducer.fit_transform(df[self._numeric_columns])
            manifold_positions = {i: tuple(coords[i]) for i in range(len(coords))}

        return df, manifold_positions

    def fit_transform(
        self,
        data: pd.DataFrame,
        numeric_columns: List[str],
        categorical_columns: List[str],
        target_columns: List[str] = None
    ) -> Tuple[pd.DataFrame, Optional[dict]]:
        """Fit and transform in one step."""
        self.fit(data, numeric_columns, categorical_columns, target_columns)
        return self.transform(data)
```

#### 2.2.6 Clean Public API

```python
# __init__.py
"""
TaGra: Generate graphs from tabular data through manifold learning.

Basic usage:
    >>> import tagra
    >>> graph = tagra.from_dataframe(df, target='label', method='knn', k=5)
    >>> metrics = tagra.analyze(graph)
    >>> tagra.visualize(graph, output='graph.png')

For ML applications:
    >>> from tagra.ml import label_propagation, augment, detect_anomalies
"""

from tagra.core.graph import TaGraGraph
from tagra.construction import build_graph
from tagra.preprocessing import preprocess
from tagra.analysis import analyze
from tagra.visualization import visualize

# Convenience function combining all steps
def from_dataframe(
    df,
    target=None,
    method='knn',
    k=5,
    distance_threshold=None,
    similarity_threshold=None,
    numeric_columns=None,
    categorical_columns=None,
    scaling='standard',
    encoding='one-hot',
    manifold_method=None,
    verbose=True
) -> TaGraGraph:
    """
    Create a TaGra graph from a pandas DataFrame.

    This is the main entry point for most users. It handles preprocessing
    and graph construction in one call.

    Args:
        df: Input DataFrame
        target: Target column name for analysis
        method: Graph construction method ('knn', 'distance', 'similarity')
        k: Number of neighbors for KNN method
        distance_threshold: Threshold for distance method
        similarity_threshold: Threshold for similarity method
        numeric_columns: Columns to treat as numeric (auto-detected if None)
        categorical_columns: Columns to treat as categorical
        scaling: Numeric scaling method ('standard', 'minmax', None)
        encoding: Categorical encoding method ('one-hot', 'label')
        manifold_method: Dimensionality reduction ('umap', 'tsne', 'isomap', None)
        verbose: Print progress messages

    Returns:
        TaGraGraph object

    Example:
        >>> graph = tagra.from_dataframe(
        ...     df,
        ...     target='outcome',
        ...     method='similarity',
        ...     similarity_threshold=0.8
        ... )
    """
    # Implementation delegates to preprocessing and construction modules
    ...


__version__ = "0.3.0"

__all__ = [
    'TaGraGraph',
    'from_dataframe',
    'preprocess',
    'build_graph',
    'analyze',
    'visualize',
    '__version__'
]
```

---

## Part 3: ML Module Design

### 3.1 Label Propagation

```python
# ml/label_propagation.py
from typing import Optional, Dict, List, Tuple
import numpy as np
from ..core.graph import TaGraGraph


class LabelPropagator:
    """
    Semi-supervised learning via label propagation on TaGra graphs.

    Propagates labels from labeled nodes to unlabeled nodes based on
    graph connectivity.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        alpha: float = 0.8  # Label retention vs propagation balance
    ):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.alpha = alpha

        self._label_distributions: Optional[np.ndarray] = None
        self._classes: Optional[List] = None

    def fit_predict(
        self,
        graph: TaGraGraph,
        target_column: str,
        unlabeled_value=None  # Value indicating unlabeled (e.g., None, -1, np.nan)
    ) -> Dict[int, Tuple[any, float]]:
        """
        Propagate labels and return predictions with confidence.

        Args:
            graph: TaGraGraph with some labeled nodes
            target_column: Column containing labels
            unlabeled_value: Value indicating unlabeled nodes

        Returns:
            Dict mapping node_id -> (predicted_label, confidence)
        """
        g = graph.graph
        nodes = list(g.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        # Extract labels
        labels = [g.nodes[node].get(target_column) for node in nodes]

        # Identify labeled vs unlabeled
        if unlabeled_value is None:
            labeled_mask = [l is not None for l in labels]
        else:
            labeled_mask = [l != unlabeled_value for l in labels]

        # Get unique classes from labeled nodes
        self._classes = list(set(l for l, m in zip(labels, labeled_mask) if m))
        n_classes = len(self._classes)
        class_to_idx = {c: i for i, c in enumerate(self._classes)}

        # Initialize label distributions
        # Labeled nodes: one-hot; Unlabeled: uniform
        Y = np.zeros((n, n_classes))
        for i, (label, is_labeled) in enumerate(zip(labels, labeled_mask)):
            if is_labeled:
                Y[i, class_to_idx[label]] = 1.0
            else:
                Y[i, :] = 1.0 / n_classes

        # Build transition matrix (row-normalized adjacency)
        T = np.zeros((n, n))
        for node in nodes:
            i = node_to_idx[node]
            neighbors = list(g.neighbors(node))
            if neighbors:
                for neighbor in neighbors:
                    j = node_to_idx[neighbor]
                    T[i, j] = 1.0 / len(neighbors)

        # Iterate until convergence
        Y_fixed = Y.copy()
        for iteration in range(self.max_iterations):
            Y_new = self.alpha * (T @ Y) + (1 - self.alpha) * Y_fixed

            # Clamp labeled nodes to their true labels
            for i, is_labeled in enumerate(labeled_mask):
                if is_labeled:
                    Y_new[i] = Y_fixed[i]

            # Check convergence
            if np.max(np.abs(Y_new - Y)) < self.convergence_threshold:
                break

            Y = Y_new

        self._label_distributions = Y

        # Extract predictions
        predictions = {}
        for node in nodes:
            i = node_to_idx[node]
            predicted_class_idx = np.argmax(Y[i])
            confidence = Y[i, predicted_class_idx]
            predictions[node] = (self._classes[predicted_class_idx], confidence)

        return predictions

    def get_uncertain_nodes(
        self,
        threshold: float = 0.6
    ) -> List[int]:
        """Get nodes with prediction confidence below threshold."""
        if self._label_distributions is None:
            raise RuntimeError("Must call fit_predict first")

        max_probs = np.max(self._label_distributions, axis=1)
        uncertain_indices = np.where(max_probs < threshold)[0]
        return uncertain_indices.tolist()
```

### 3.2 Data Augmentation

```python
# ml/augmentation.py
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from ..core.graph import TaGraGraph


class GraphAwareAugmenter:
    """
    Generate synthetic samples by interpolating between connected nodes.

    Unlike SMOTE, only interpolates between nodes that are actually
    connected in the similarity graph, ensuring synthetic samples
    are realistic.
    """

    def __init__(
        self,
        interpolation_range: Tuple[float, float] = (0.3, 0.7),
        same_class_only: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Args:
            interpolation_range: (min, max) for interpolation alpha
            same_class_only: Only interpolate between same-class nodes
            random_state: Random seed for reproducibility
        """
        self.interpolation_range = interpolation_range
        self.same_class_only = same_class_only
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def augment(
        self,
        graph: TaGraGraph,
        target_column: str,
        target_class: any,
        n_samples: int,
        numeric_columns: List[str]
    ) -> pd.DataFrame:
        """
        Generate synthetic samples for a target class.

        Args:
            graph: TaGraGraph to sample from
            target_column: Column containing class labels
            target_class: Class to augment
            n_samples: Number of synthetic samples to generate
            numeric_columns: Columns to interpolate

        Returns:
            DataFrame of synthetic samples
        """
        g = graph.graph

        # Find nodes of target class
        target_nodes = [
            n for n in g.nodes()
            if g.nodes[n].get(target_column) == target_class
        ]

        if len(target_nodes) < 2:
            raise ValueError(f"Need at least 2 nodes of class {target_class}")

        # Find valid edges for interpolation
        valid_edges = []
        for node in target_nodes:
            for neighbor in g.neighbors(node):
                if self.same_class_only:
                    if g.nodes[neighbor].get(target_column) == target_class:
                        valid_edges.append((node, neighbor))
                else:
                    valid_edges.append((node, neighbor))

        if not valid_edges:
            raise ValueError("No valid edges for interpolation")

        # Generate synthetic samples
        synthetic_samples = []
        for _ in range(n_samples):
            # Random edge
            node_i, node_j = valid_edges[self._rng.integers(len(valid_edges))]

            # Random interpolation factor
            alpha = self._rng.uniform(*self.interpolation_range)

            # Interpolate numeric features
            sample = {}
            for col in numeric_columns:
                val_i = g.nodes[node_i].get(col, 0)
                val_j = g.nodes[node_j].get(col, 0)
                sample[col] = alpha * val_i + (1 - alpha) * val_j

            # Copy target class
            sample[target_column] = target_class

            synthetic_samples.append(sample)

        return pd.DataFrame(synthetic_samples)

    def balance_classes(
        self,
        graph: TaGraGraph,
        target_column: str,
        numeric_columns: List[str],
        strategy: str = 'minority'  # 'minority', 'all', or specific ratio
    ) -> pd.DataFrame:
        """
        Balance classes by augmenting minority classes.

        Returns DataFrame with synthetic samples only (combine with original data).
        """
        g = graph.graph

        # Count classes
        class_counts = {}
        for node in g.nodes():
            label = g.nodes[node].get(target_column)
            class_counts[label] = class_counts.get(label, 0) + 1

        max_count = max(class_counts.values())

        all_synthetic = []
        for cls, count in class_counts.items():
            if count < max_count:
                n_needed = max_count - count
                try:
                    synthetic = self.augment(
                        graph, target_column, cls, n_needed, numeric_columns
                    )
                    all_synthetic.append(synthetic)
                except ValueError:
                    # Not enough edges for this class
                    pass

        if all_synthetic:
            return pd.concat(all_synthetic, ignore_index=True)
        return pd.DataFrame()
```

### 3.3 Anomaly Detection

```python
# ml/anomaly.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
from ..core.graph import TaGraGraph


@dataclass
class AnomalyExplanation:
    """Explanation for why a node is considered anomalous."""
    node_id: int
    anomaly_score: float
    reasons: List[str]
    neighbor_class_distribution: Dict[Any, float]
    node_class: Any
    degree: int

    def __str__(self) -> str:
        reasons_str = "\n  - ".join(self.reasons)
        return (
            f"Node {self.node_id} (anomaly score: {self.anomaly_score:.3f})\n"
            f"  Class: {self.node_class}, Degree: {self.degree}\n"
            f"  Neighbor distribution: {self.neighbor_class_distribution}\n"
            f"  Reasons:\n  - {reasons_str}"
        )


class GraphAnomalyDetector:
    """
    Detect anomalies based on graph structure with explainable results.

    Anomaly indicators:
    - Isolated nodes (low degree)
    - Class incongruence (different class than neighbors)
    - Local density anomaly (sparse neighborhood in dense region)
    """

    def __init__(
        self,
        isolation_weight: float = 0.3,
        incongruence_weight: float = 0.5,
        density_weight: float = 0.2
    ):
        self.isolation_weight = isolation_weight
        self.incongruence_weight = incongruence_weight
        self.density_weight = density_weight

    def detect(
        self,
        graph: TaGraGraph,
        target_column: Optional[str] = None
    ) -> List[AnomalyExplanation]:
        """
        Detect anomalies and return explanations.

        Args:
            graph: TaGraGraph to analyze
            target_column: Column for class incongruence analysis

        Returns:
            List of AnomalyExplanation for all nodes, sorted by score
        """
        g = graph.graph
        nodes = list(g.nodes())

        # Compute degree statistics
        degrees = dict(g.degree())
        max_degree = max(degrees.values()) if degrees else 1
        mean_degree = np.mean(list(degrees.values())) if degrees else 0

        explanations = []

        for node in nodes:
            reasons = []
            scores = []

            degree = degrees[node]
            neighbors = list(g.neighbors(node))

            # 1. Isolation score
            isolation_score = 1 - (degree / max_degree) if max_degree > 0 else 1
            scores.append(isolation_score * self.isolation_weight)

            if degree == 0:
                reasons.append("Completely isolated (no connections)")
            elif degree < mean_degree * 0.2:
                reasons.append(f"Very low connectivity (degree={degree}, mean={mean_degree:.1f})")

            # 2. Class incongruence score
            incongruence_score = 0
            neighbor_dist = {}
            node_class = None

            if target_column and neighbors:
                node_class = g.nodes[node].get(target_column)
                neighbor_classes = [g.nodes[n].get(target_column) for n in neighbors]

                for cls in set(neighbor_classes):
                    neighbor_dist[cls] = neighbor_classes.count(cls) / len(neighbor_classes)

                same_class_ratio = neighbor_dist.get(node_class, 0)
                incongruence_score = 1 - same_class_ratio
                scores.append(incongruence_score * self.incongruence_weight)

                if same_class_ratio < 0.3:
                    reasons.append(
                        f"Class incongruence: node is {node_class} but "
                        f"{(1-same_class_ratio)*100:.0f}% of neighbors are different"
                    )

            # 3. Local density anomaly
            density_score = 0
            if neighbors:
                neighbor_degrees = [degrees[n] for n in neighbors]
                mean_neighbor_degree = np.mean(neighbor_degrees)

                if mean_neighbor_degree > 0:
                    density_ratio = degree / mean_neighbor_degree
                    if density_ratio < 0.5:
                        density_score = 1 - density_ratio
                        reasons.append(
                            f"Sparse in dense region (degree={degree}, "
                            f"neighbor mean={mean_neighbor_degree:.1f})"
                        )

                scores.append(density_score * self.density_weight)

            # Combine scores
            total_score = sum(scores)

            explanations.append(AnomalyExplanation(
                node_id=node,
                anomaly_score=total_score,
                reasons=reasons if reasons else ["No anomalous characteristics"],
                neighbor_class_distribution=neighbor_dist,
                node_class=node_class,
                degree=degree
            ))

        # Sort by anomaly score (descending)
        explanations.sort(key=lambda x: x.anomaly_score, reverse=True)

        return explanations

    def get_top_anomalies(
        self,
        graph: TaGraGraph,
        target_column: Optional[str] = None,
        n: int = 10,
        min_score: float = 0.5
    ) -> List[AnomalyExplanation]:
        """Get top N anomalies above minimum score threshold."""
        all_anomalies = self.detect(graph, target_column)
        return [a for a in all_anomalies[:n] if a.anomaly_score >= min_score]
```

---

## Part 4: Migration Strategy

### 4.1 Phased Approach

#### Phase 1: Foundation (Non-Breaking)

Create new module structure alongside existing code:

```
tagra/
├── __init__.py              # Add new imports, keep old ones
├── config.py                # Keep as-is
├── preprocessing.py         # Keep as-is (legacy)
├── graph.py                 # Keep as-is (legacy)
├── analysis.py              # Keep as-is (legacy)
├── utils.py                 # Keep as-is (legacy)
│
├── core/                    # NEW
├── construction/            # NEW
├── preprocessing_v2/        # NEW (temporary name)
└── ...
```

#### Phase 2: Internal Refactoring

Refactor old modules to use new internal structure:

```python
# preprocessing.py (updated)
from tagra.preprocessing_v2 import PreprocessingPipeline

def preprocess_dataframe(...):
    """Legacy function - delegates to new implementation."""
    pipeline = PreprocessingPipeline(...)
    # ... map old parameters to new API
    return pipeline.fit_transform(...)
```

#### Phase 3: Deprecation Warnings

Add deprecation warnings to old API:

```python
import warnings

def preprocess_dataframe(...):
    warnings.warn(
        "preprocess_dataframe() is deprecated. "
        "Use tagra.preprocess() or PreprocessingPipeline instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # ... existing implementation
```

#### Phase 4: Documentation Update

- Update README with new API
- Add migration guide
- Update examples

#### Phase 5: Cleanup (Major Version)

- Remove legacy modules
- Rename `preprocessing_v2/` to `preprocessing/`
- Bump to version 1.0.0

### 4.2 Backward Compatibility Layer

```python
# compat/legacy.py
"""
Backward compatibility layer for TaGra < 1.0 API.

This module provides the old function signatures that delegate to
the new implementation. It will be removed in version 2.0.
"""

import warnings
from typing import Optional, List, Tuple
import pandas as pd

from tagra.preprocessing import PreprocessingPipeline
from tagra.construction import ConstructorRegistry
from tagra.core.graph import TaGraGraph


def preprocess_dataframe(
    input_dataframe=None,
    output_directory="results/",
    preprocessed_filename=None,
    # ... all old parameters
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """
    Legacy preprocessing function.

    .. deprecated:: 0.3.0
        Use :func:`tagra.preprocess` or :class:`PreprocessingPipeline` instead.
    """
    warnings.warn(
        "preprocess_dataframe() is deprecated since version 0.3.0. "
        "Use tagra.preprocess() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Map old parameters to new API
    pipeline = PreprocessingPipeline(
        # ... parameter mapping
    )

    # ... implementation using new internals

    return df_preprocessed, manifold_positions
```

---

## Part 5: Testing Strategy

### 5.1 Test Structure

```
tests/
├── unit/
│   ├── test_preprocessing/
│   │   ├── test_scaling.py
│   │   ├── test_encoding.py
│   │   └── test_pipeline.py
│   ├── test_construction/
│   │   ├── test_knn.py
│   │   ├── test_distance.py
│   │   └── test_similarity.py
│   ├── test_analysis/
│   │   └── ...
│   └── test_ml/
│       ├── test_label_propagation.py
│       ├── test_augmentation.py
│       └── test_anomaly.py
│
├── integration/
│   ├── test_full_pipeline.py
│   └── test_backward_compat.py
│
└── fixtures/
    ├── sample_data.csv
    └── expected_outputs/
```

### 5.2 Key Test Cases

```python
# tests/integration/test_backward_compat.py
"""Ensure old API still works identically."""

import pandas as pd
from tagra.compat.legacy import preprocess_dataframe, create_graph, analyze_graph


def test_legacy_pipeline_produces_same_results():
    """Old API should produce identical results to new API."""
    df = pd.read_csv('tests/fixtures/sample_data.csv')

    # Old way
    df_old, pos_old = preprocess_dataframe(
        input_dataframe=df,
        numeric_scaling='standard',
        # ...
    )

    # New way
    from tagra import preprocess
    df_new, pos_new = preprocess(
        df,
        scaling='standard',
        # ...
    )

    pd.testing.assert_frame_equal(df_old, df_new)
    assert pos_old == pos_new
```

---

## Part 6: Timeline and Priorities

### Recommended Sequence

| Phase | Tasks | Priority | Dependency |
|-------|-------|----------|------------|
| 1 | Create `core/` with `TaGraGraph` class | High | None |
| 2 | Create `construction/` with base class and implementations | High | Phase 1 |
| 3 | Create `ml/` module (label propagation, augmentation, anomaly) | High | Phase 2 |
| 4 | Refactor `preprocessing.py` into `preprocessing/` package | Medium | None |
| 5 | Refactor `analysis.py` and `utils.py` into `analysis/` and `visualization/` | Medium | Phase 1 |
| 6 | Create `longitudinal/` module | Medium | Phases 1-3 |
| 7 | Create `io/` module with Cytoscape export | Low | Phase 1 |
| 8 | Add backward compatibility layer | Medium | Phases 4-5 |
| 9 | Update documentation and examples | High | All above |

### Suggested Development Order

1. **Start with `core/` and `construction/`** - These are foundational and enable testing new patterns
2. **Add `ml/` immediately** - This is the main value-add for the follow-up paper
3. **Refactor existing modules gradually** - Don't block ML development on full refactoring
4. **Add backward compatibility** - Only needed when new API is stable

---

## Part 7: Summary

### Current State
- Functional but monolithic
- Hard to extend, hard to test components in isolation
- No clear path for adding ML capabilities

### Target State
- Modular, extensible architecture
- Clear public API hiding implementation details
- Easy to add new graph construction methods, analysis metrics, or ML algorithms
- Backward compatible with existing users

### Key Design Principles
1. **Single Responsibility**: Each module/class does one thing well
2. **Open/Closed**: Add new methods without modifying existing code
3. **Dependency Inversion**: Depend on abstractions (base classes), not concretions
4. **Explicit over Implicit**: Clear public API, documented behavior

### Risk Mitigation
- Phased migration prevents breaking existing users
- Comprehensive test suite catches regressions
- Backward compatibility layer provides smooth transition path
