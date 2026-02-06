"""
Graph-based anomaly detection.

This module provides anomaly detection methods that leverage
graph structure for identifying outliers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, Union, List
import numpy as np
import networkx as nx


@dataclass
class AnomalyExplanation:
    """
    Explanation for an anomaly detection result.

    Attributes
    ----------
    node_id : int
        Node identifier
    anomaly_score : float
        Anomaly score (higher = more anomalous)
    is_anomaly : bool
        Whether classified as anomaly
    reasons : List[str]
        Human-readable explanation reasons
    metrics : Dict[str, float]
        Detailed metrics contributing to score
    """
    node_id: int
    anomaly_score: float
    is_anomaly: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class GraphAnomalyDetector:
    """
    Anomaly detection using graph structure.

    Detects anomalous nodes based on their structural properties
    and relationship to neighbors.

    Parameters
    ----------
    method : str, default='combined'
        Detection method:
        - 'structural': Based on degree, clustering, etc.
        - 'attribute': Based on attribute deviation from neighbors
        - 'combined': Both structural and attribute
    contamination : float, default=0.1
        Expected proportion of anomalies
    threshold_percentile : float, optional
        Percentile threshold for anomaly classification
    verbose : bool, default=True
        Print progress messages

    Attributes
    ----------
    anomaly_scores_ : np.ndarray
        Anomaly scores for each node
    threshold_ : float
        Score threshold for classification
    is_anomaly_ : np.ndarray
        Boolean array indicating anomalies

    Examples
    --------
    >>> detector = GraphAnomalyDetector(method='combined')
    >>> detector.fit(G)
    >>> anomalies = detector.get_anomalies()
    """

    def __init__(
        self,
        method: str = 'combined',
        contamination: float = 0.1,
        threshold_percentile: Optional[float] = None,
        verbose: bool = True
    ):
        self.method = method
        self.contamination = contamination
        self.threshold_percentile = threshold_percentile or (1 - contamination) * 100
        self.verbose = verbose

        self.anomaly_scores_: Optional[np.ndarray] = None
        self.threshold_: float = 0.0
        self.is_anomaly_: Optional[np.ndarray] = None
        self._nodes: List[int] = []
        self._G: Optional[nx.Graph] = None

    def fit(
        self,
        graph: Union[nx.Graph, 'TaGraGraph'],
        attribute_columns: Optional[List[str]] = None
    ) -> 'GraphAnomalyDetector':
        """
        Fit the anomaly detector.

        Parameters
        ----------
        graph : nx.Graph or TaGraGraph
            Graph to analyze
        attribute_columns : List[str], optional
            Node attributes to use for attribute-based detection

        Returns
        -------
        GraphAnomalyDetector
            Self for method chaining
        """
        if hasattr(graph, 'to_networkx'):
            G = graph.to_networkx()
        else:
            G = graph

        self._G = G
        self._nodes = list(G.nodes())
        n_nodes = len(self._nodes)

        if self.verbose:
            print(f"{datetime.now()}: Fitting anomaly detector on {n_nodes} nodes...")

        scores = np.zeros(n_nodes)

        if self.method in ['structural', 'combined']:
            structural_scores = self._compute_structural_scores(G)
            scores += structural_scores

        if self.method in ['attribute', 'combined'] and attribute_columns:
            attribute_scores = self._compute_attribute_scores(G, attribute_columns)
            scores += attribute_scores

        # Normalize scores
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())

        self.anomaly_scores_ = scores
        self.threshold_ = np.percentile(scores, self.threshold_percentile)
        self.is_anomaly_ = scores > self.threshold_

        if self.verbose:
            n_anomalies = np.sum(self.is_anomaly_)
            print(f"{datetime.now()}: Detected {n_anomalies} anomalies "
                  f"({n_anomalies/n_nodes*100:.1f}%)")

        return self

    def _compute_structural_scores(self, G: nx.Graph) -> np.ndarray:
        """Compute anomaly scores based on structural properties."""
        n_nodes = len(self._nodes)
        scores = np.zeros(n_nodes)

        # Degree-based score
        degrees = np.array([G.degree(n) for n in self._nodes])
        mean_degree = np.mean(degrees)
        std_degree = np.std(degrees)
        if std_degree > 0:
            degree_z = np.abs(degrees - mean_degree) / std_degree
            scores += degree_z

        # Clustering coefficient anomaly
        clustering = nx.clustering(G)
        clustering_vals = np.array([clustering.get(n, 0) for n in self._nodes])
        mean_cc = np.mean(clustering_vals)
        std_cc = np.std(clustering_vals)
        if std_cc > 0:
            cc_z = np.abs(clustering_vals - mean_cc) / std_cc
            scores += cc_z

        # Local outlier factor approximation (based on neighbor density)
        for idx, node in enumerate(self._nodes):
            neighbors = list(G.neighbors(node))
            if len(neighbors) < 2:
                scores[idx] += 1  # Low connectivity is suspicious
                continue

            # Average neighbor degree
            neighbor_degrees = [G.degree(n) for n in neighbors]
            avg_neighbor_degree = np.mean(neighbor_degrees)

            # If node's degree is very different from neighbors
            if avg_neighbor_degree > 0:
                ratio = abs(degrees[idx] - avg_neighbor_degree) / avg_neighbor_degree
                scores[idx] += ratio

        return scores

    def _compute_attribute_scores(
        self,
        G: nx.Graph,
        attribute_columns: List[str]
    ) -> np.ndarray:
        """Compute anomaly scores based on attribute deviation."""
        n_nodes = len(self._nodes)
        scores = np.zeros(n_nodes)

        for col in attribute_columns:
            # Get attribute values
            values = []
            for node in self._nodes:
                val = G.nodes[node].get(col)
                if val is not None and isinstance(val, (int, float)):
                    values.append(val)
                else:
                    values.append(np.nan)

            values = np.array(values)
            valid_mask = ~np.isnan(values)

            if not np.any(valid_mask):
                continue

            # Compare each node to its neighbors
            for idx, node in enumerate(self._nodes):
                if not valid_mask[idx]:
                    continue

                neighbors = list(G.neighbors(node))
                neighbor_indices = [self._nodes.index(n) for n in neighbors if n in self._nodes]
                neighbor_values = values[neighbor_indices]
                neighbor_values = neighbor_values[~np.isnan(neighbor_values)]

                if len(neighbor_values) == 0:
                    continue

                # Z-score relative to neighbors
                neighbor_mean = np.mean(neighbor_values)
                neighbor_std = np.std(neighbor_values)

                if neighbor_std > 0:
                    z = abs(values[idx] - neighbor_mean) / neighbor_std
                    scores[idx] += z

        return scores

    def predict(self) -> np.ndarray:
        """
        Get anomaly predictions.

        Returns
        -------
        np.ndarray
            Boolean array indicating anomalies
        """
        if self.is_anomaly_ is None:
            raise RuntimeError("Model must be fitted before predict")
        return self.is_anomaly_.copy()

    def get_scores(self) -> np.ndarray:
        """
        Get anomaly scores.

        Returns
        -------
        np.ndarray
            Anomaly scores for each node
        """
        if self.anomaly_scores_ is None:
            raise RuntimeError("Model must be fitted before get_scores")
        return self.anomaly_scores_.copy()

    def get_anomalies(self) -> List[int]:
        """
        Get list of anomalous node IDs.

        Returns
        -------
        List[int]
            Node IDs classified as anomalies
        """
        if self.is_anomaly_ is None:
            raise RuntimeError("Model must be fitted before get_anomalies")
        return [self._nodes[i] for i in np.where(self.is_anomaly_)[0]]

    def explain(self, node_id: int) -> AnomalyExplanation:
        """
        Get explanation for a node's anomaly score.

        Parameters
        ----------
        node_id : int
            Node to explain

        Returns
        -------
        AnomalyExplanation
            Detailed explanation
        """
        if self.anomaly_scores_ is None or self._G is None:
            raise RuntimeError("Model must be fitted before explain")

        idx = self._nodes.index(node_id)
        score = self.anomaly_scores_[idx]
        is_anomaly = self.is_anomaly_[idx]

        reasons = []
        metrics = {}

        # Degree analysis
        degree = self._G.degree(node_id)
        all_degrees = [self._G.degree(n) for n in self._nodes]
        mean_degree = np.mean(all_degrees)
        metrics['degree'] = degree
        metrics['mean_degree'] = mean_degree

        if degree < mean_degree * 0.5:
            reasons.append(f"Low degree ({degree}) compared to average ({mean_degree:.1f})")
        elif degree > mean_degree * 2:
            reasons.append(f"High degree ({degree}) compared to average ({mean_degree:.1f})")

        # Clustering analysis
        cc = nx.clustering(self._G, node_id)
        metrics['clustering_coefficient'] = cc
        if cc < 0.1:
            reasons.append(f"Low clustering coefficient ({cc:.3f})")

        if not reasons:
            reasons.append("No specific structural anomalies detected")

        return AnomalyExplanation(
            node_id=node_id,
            anomaly_score=score,
            is_anomaly=is_anomaly,
            reasons=reasons,
            metrics=metrics
        )
