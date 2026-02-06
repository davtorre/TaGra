"""
TaGra Machine Learning Module.

This module provides machine learning capabilities that leverage
graph structure for improved predictions and insights.

Features
--------
- Label Propagation: Semi-supervised learning using graph structure
- Graph-Aware Augmentation: Data augmentation using neighbor information
- Anomaly Detection: Identify outliers based on graph properties
- Graph Imputation: Fill missing values using neighbor information

Examples
--------
Label propagation for semi-supervised learning:

>>> from tagra.ml import LabelPropagator
>>> propagator = LabelPropagator()
>>> propagator.fit(G, labels, labeled_mask)
>>> predictions = propagator.predict()

Graph-aware data augmentation:

>>> from tagra.ml import GraphAwareAugmenter
>>> augmenter = GraphAwareAugmenter(n_samples=2)
>>> X_aug, y_aug = augmenter.augment(G, X, y)

Anomaly detection:

>>> from tagra.ml import GraphAnomalyDetector
>>> detector = GraphAnomalyDetector()
>>> detector.fit(G)
>>> anomalies = detector.get_anomalies()
>>> explanation = detector.explain(node_id)

Graph-based imputation:

>>> from tagra.ml import GraphImputer
>>> imputer = GraphImputer(strategy='weighted_mean')
>>> df_imputed = imputer.fit_transform(G, df)
"""

from .label_propagation import LabelPropagator, propagate_labels
from .augmentation import GraphAwareAugmenter
from .anomaly import GraphAnomalyDetector, AnomalyExplanation
from .imputation import GraphImputer, impute_with_graph

__all__ = [
    # Label propagation
    'LabelPropagator',
    'propagate_labels',
    # Augmentation
    'GraphAwareAugmenter',
    # Anomaly detection
    'GraphAnomalyDetector',
    'AnomalyExplanation',
    # Imputation
    'GraphImputer',
    'impute_with_graph'
]
