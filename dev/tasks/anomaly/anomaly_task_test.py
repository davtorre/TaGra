#!/usr/bin/env python3
"""
Anomaly Detection Comparison Script

Systematically compares traditional anomaly detection methods with TaGra's graph-based
anomaly detection across multiple graph representations and parameters.

Usage:
    python anomaly_task_test.py -d /path/to/dataframe.csv [OPTIONS]

Example:
    python anomaly_task_test.py -d data/diabetes.csv --target outcome --anomaly-label 1
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Traditional anomaly detection
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# TaGra imports
from tagra import from_dataframe
from tagra.preprocessing import preprocess
from tagra.construction import build_graph
from tagra.ml import GraphAnomalyDetector
from tagra.cytoscape_vis import CytoscapeVisualizer

warnings.filterwarnings('ignore')


class TraditionalDetectors:
    """Wrapper for sklearn-based anomaly detection methods"""

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state

    def fit_isolation_forest(self, X: np.ndarray, n_estimators: int = 100) -> Dict[str, Any]:
        """Run Isolation Forest"""
        clf = IsolationForest(
            contamination=self.contamination,
            n_estimators=n_estimators,
            random_state=self.random_state
        )
        preds = clf.fit_predict(X)
        scores = -clf.score_samples(X)  # Higher = more anomalous

        return {
            'predictions': preds == -1,
            'scores': scores,
            'method': 'isolation_forest'
        }

    def fit_lof(self, X: np.ndarray, n_neighbors: int = 20) -> Dict[str, Any]:
        """Run Local Outlier Factor"""
        clf = LocalOutlierFactor(
            contamination=self.contamination,
            n_neighbors=n_neighbors
        )
        preds = clf.fit_predict(X)
        scores = -clf.negative_outlier_factor_

        return {
            'predictions': preds == -1,
            'scores': scores,
            'method': f'lof_n{n_neighbors}'
        }

    def fit_ocsvm(self, X: np.ndarray, kernel: str = 'rbf') -> Dict[str, Any]:
        """Run One-Class SVM"""
        clf = OneClassSVM(
            nu=self.contamination,
            kernel=kernel,
            gamma='auto'
        )
        preds = clf.fit_predict(X)
        scores = -clf.score_samples(X)

        return {
            'predictions': preds == -1,
            'scores': scores,
            'method': 'ocsvm'
        }

    def fit_dbscan(self, X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> Dict[str, Any]:
        """Run DBSCAN for outlier detection"""
        clf = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clf.fit_predict(X)

        return {
            'predictions': labels == -1,  # -1 = outlier
            'scores': None,  # DBSCAN doesn't produce scores
            'method': f'dbscan_eps{eps}'
        }


class GraphAnomalyTester:
    """Test graph-based anomaly detection across parameter grid"""

    GRAPH_PARAM_GRID = {
        'knn': {
            'k': [2, 3, 5, 7, 10]
        },
        'distance': {
            'distance_threshold': [0.5, 1.0, 1.5, 2.0, 3.0]
        },
        'similarity': {
            'similarity_threshold': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
        }
    }

    DETECTION_METHODS = ['structural', 'attribute', 'combined']

    def __init__(self, df: pd.DataFrame, df_preprocessed: np.ndarray,
                 contamination: float = 0.1, verbose: bool = False):
        self.df = df
        self.df_preprocessed = df_preprocessed
        self.contamination = contamination
        self.verbose = verbose
        self.graphs = {}

    def build_graph_variants(self, graph_methods: List[str] = None) -> Dict[str, Any]:
        """Build all graph variants based on parameter grid"""
        if graph_methods is None:
            graph_methods = list(self.GRAPH_PARAM_GRID.keys())

        print("\nBuilding graph variants...")

        for method in graph_methods:
            if method not in self.GRAPH_PARAM_GRID:
                continue

            params = self.GRAPH_PARAM_GRID[method]

            if method == 'knn':
                for k in params['k']:
                    key = f'knn_k{k}'
                    try:
                        graph = from_dataframe(
                            self.df,
                            method='knn',
                            k=k,
                            verbose=False
                        )
                        self.graphs[key] = graph
                        if self.verbose:
                            print(f"  - KNN (k={k}): {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
                    except Exception as e:
                        print(f"  - KNN (k={k}): Failed - {e}")

            elif method == 'distance':
                for threshold in params['distance_threshold']:
                    key = f'distance_{threshold}'
                    try:
                        graph = from_dataframe(
                            self.df,
                            method='distance',
                            distance_threshold=threshold,
                            verbose=False
                        )
                        self.graphs[key] = graph
                        if self.verbose:
                            print(f"  - Distance (threshold={threshold}): {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
                    except Exception as e:
                        print(f"  - Distance (threshold={threshold}): Failed - {e}")

            elif method == 'similarity':
                for threshold in params['similarity_threshold']:
                    key = f'similarity_{threshold}'
                    try:
                        graph = from_dataframe(
                            self.df,
                            method='similarity',
                            similarity_threshold=threshold,
                            verbose=False
                        )
                        self.graphs[key] = graph
                        if self.verbose:
                            print(f"  - Similarity (threshold={threshold}): {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
                    except Exception as e:
                        print(f"  - Similarity (threshold={threshold}): Failed - {e}")

        print(f"Successfully built {len(self.graphs)} graph variants")
        return self.graphs

    def detect_anomalies_on_graph(self, graph: Any, method: str, graph_config: str) -> Dict[str, Any]:
        """Run anomaly detection on a single graph"""
        try:
            # Get numeric columns for attribute-based detection
            numeric_cols = []
            for node in list(graph.nodes())[:1]:  # Check first node
                node_data = graph.graph.nodes[node]
                for key, value in node_data.items():
                    if isinstance(value, (int, float, np.number)) and key not in ['x', 'y']:
                        numeric_cols.append(key)
                break

            detector = GraphAnomalyDetector(
                method=method,
                contamination=self.contamination,
                verbose=False
            )

            # Fit detector
            if method == 'structural':
                detector.fit(graph)
            else:
                detector.fit(graph, attribute_columns=numeric_cols)

            # Get predictions and scores
            predictions = detector.predict()
            scores = detector.get_scores()
            anomaly_ids = detector.get_anomalies()

            return {
                'predictions': predictions,
                'scores': scores,
                'anomaly_ids': anomaly_ids,
                'graph_config': graph_config,
                'method': method,
                'num_anomalies': len(anomaly_ids)
            }

        except Exception as e:
            print(f"  - Warning: {graph_config}_{method} failed - {e}")
            return None

    def run_all_combinations(self, detection_methods: List[str] = None) -> Dict[str, Any]:
        """Run all combinations of graph variants and detection methods"""
        if detection_methods is None:
            detection_methods = self.DETECTION_METHODS

        print("\nRunning graph-based anomaly detection...")
        results = {}

        for graph_config, graph in self.graphs.items():
            for method in detection_methods:
                key = f"graph_{graph_config}_{method}"
                result = self.detect_anomalies_on_graph(graph, method, graph_config)

                if result is not None:
                    results[key] = result
                    if self.verbose:
                        print(f"  - {key}: {result['num_anomalies']} anomalies detected")

        print(f"Successfully ran {len(results)} graph-based detection methods")
        return results


class ResultsAggregator:
    """Combine and analyze results from all methods"""

    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        self.results_df = None
        self.metrics = None

    def aggregate(self, traditional_results: Dict, graph_results: Dict) -> pd.DataFrame:
        """Combine all results into a single DataFrame"""
        print("\nAggregating results...")

        # Initialize DataFrame with sample IDs
        data = {'sample_id': np.arange(self.n_samples)}

        # Add traditional method results
        for method_name, result in traditional_results.items():
            data[f'traditional_{method_name}'] = result['predictions']

        # Add graph method results
        for method_name, result in graph_results.items():
            data[method_name] = result['predictions']

        df = pd.DataFrame(data)

        # Calculate summary statistics
        method_cols = [col for col in df.columns if col != 'sample_id']
        df['total_detections'] = df[method_cols].sum(axis=1)

        traditional_cols = [col for col in df.columns if col.startswith('traditional_')]
        df['traditional_count'] = df[traditional_cols].sum(axis=1)

        graph_cols = [col for col in df.columns if col.startswith('graph_')]
        df['graph_count'] = df[graph_cols].sum(axis=1)

        df['agreement_score'] = df['total_detections'] / len(method_cols)

        self.results_df = df

        # Print statistics
        n_methods = len(method_cols)
        n_flagged = (df['total_detections'] > 0).sum()
        n_consensus = (df['total_detections'] >= n_methods / 2).sum()
        avg_agreement = df['agreement_score'].mean()

        print(f"  - Total methods: {n_methods}")
        print(f"  - Samples flagged by at least one method: {n_flagged} ({100*n_flagged/self.n_samples:.1f}%)")
        print(f"  - Samples flagged by ≥50% methods: {n_consensus} ({100*n_consensus/self.n_samples:.1f}%)")
        print(f"  - Average agreement score: {avg_agreement:.3f}")

        return df

    def compute_metrics(self, ground_truth: pd.Series, anomaly_label: Any,
                       traditional_results: Dict, graph_results: Dict) -> Dict[str, Dict]:
        """Compute evaluation metrics if ground truth is available"""
        print("\nComputing evaluation metrics...")

        y_true = (ground_truth == anomaly_label).astype(int).values
        metrics = {}

        # Traditional methods
        for method_name, result in traditional_results.items():
            y_pred = result['predictions'].astype(int)
            scores = result['scores']

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )

            metrics[f'traditional_{method_name}'] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }

            # Add AUC if scores available
            if scores is not None:
                try:
                    auc = roc_auc_score(y_true, scores)
                    metrics[f'traditional_{method_name}']['auc'] = float(auc)
                except:
                    metrics[f'traditional_{method_name}']['auc'] = None

        # Graph methods
        for method_name, result in graph_results.items():
            y_pred = result['predictions'].astype(int)
            scores = result['scores']

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )

            metrics[method_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }

            # Add AUC if scores available
            if scores is not None:
                try:
                    auc = roc_auc_score(y_true, scores)
                    metrics[method_name]['auc'] = float(auc)
                except:
                    metrics[method_name]['auc'] = None

        self.metrics = metrics

        # Print best performers
        f1_scores = {k: v['f1'] for k, v in metrics.items()}
        best_method = max(f1_scores, key=f1_scores.get)
        print(f"  - Best performing method: {best_method} (F1={f1_scores[best_method]:.3f})")

        return metrics


class Visualizer:
    """Generate comparison visualizations"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def plot_detection_heatmap(self, results_df: pd.DataFrame):
        """Generate detection heatmap"""
        print("  ✓ Generating detection_heatmap.png")

        # Sort by total detections
        df_sorted = results_df.sort_values('total_detections', ascending=False)

        # Get method columns
        method_cols = [col for col in df_sorted.columns
                      if col not in ['sample_id', 'total_detections', 'traditional_count',
                                    'graph_count', 'agreement_score']]

        # Create heatmap data
        heatmap_data = df_sorted[method_cols].astype(int)

        # Plot
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.heatmap(heatmap_data.T, cmap='RdYlGn_r', cbar_kws={'label': 'Detected'},
                   ax=ax, xticklabels=False, yticklabels=True)
        ax.set_xlabel('Samples (sorted by total detections)')
        ax.set_ylabel('Detection Methods')
        ax.set_title('Anomaly Detection Heatmap')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'detection_heatmap.png', dpi=150)
        plt.close()

    def plot_method_agreement(self, results_df: pd.DataFrame):
        """Generate method agreement matrix"""
        print("  ✓ Generating method_agreement.png")

        # Get method columns
        method_cols = [col for col in results_df.columns
                      if col not in ['sample_id', 'total_detections', 'traditional_count',
                                    'graph_count', 'agreement_score']]

        # Compute Jaccard similarity matrix
        n_methods = len(method_cols)
        similarity_matrix = np.zeros((n_methods, n_methods))

        for i, method1 in enumerate(method_cols):
            for j, method2 in enumerate(method_cols):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    set1 = set(results_df[results_df[method1]]['sample_id'])
                    set2 = set(results_df[results_df[method2]]['sample_id'])
                    if len(set1) == 0 and len(set2) == 0:
                        similarity_matrix[i, j] = 1.0
                    elif len(set1.union(set2)) == 0:
                        similarity_matrix[i, j] = 0.0
                    else:
                        similarity_matrix[i, j] = len(set1.intersection(set2)) / len(set1.union(set2))

        # Plot
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(similarity_matrix, annot=False, cmap='YlOrRd',
                   xticklabels=method_cols, yticklabels=method_cols,
                   vmin=0, vmax=1, ax=ax)
        ax.set_title('Method Agreement Matrix (Jaccard Similarity)')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'method_agreement.png', dpi=150)
        plt.close()

    def plot_detection_distribution(self, results_df: pd.DataFrame):
        """Generate detection count distribution"""
        print("  ✓ Generating detection_distribution.png")

        fig, ax = plt.subplots(figsize=(10, 6))

        counts = results_df['total_detections'].value_counts().sort_index()
        ax.bar(counts.index, counts.values, color='steelblue', alpha=0.7)
        ax.set_xlabel('Number of Methods Flagging Sample')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Distribution of Detection Counts')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'detection_distribution.png', dpi=150)
        plt.close()

    def plot_score_distributions(self, traditional_results: Dict, graph_results: Dict):
        """Generate anomaly score distributions"""
        print("  ✓ Generating score_distributions.png")

        # Collect scores from methods that have them
        score_data = []

        for method_name, result in traditional_results.items():
            if result['scores'] is not None:
                for score in result['scores']:
                    score_data.append({
                        'method': f"trad_{method_name}",
                        'score': score
                    })

        for method_name, result in graph_results.items():
            if result['scores'] is not None:
                for score in result['scores']:
                    score_data.append({
                        'method': method_name,
                        'score': score
                    })

        if len(score_data) == 0:
            print("    (No scores available)")
            return

        df_scores = pd.DataFrame(score_data)

        fig, ax = plt.subplots(figsize=(14, 6))

        # Use violin plot for better visualization
        sns.violinplot(data=df_scores, x='method', y='score', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title('Anomaly Score Distributions by Method')
        ax.set_ylabel('Anomaly Score')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'score_distributions.png', dpi=150)
        plt.close()

    def plot_graph_param_impact(self, graph_results: Dict):
        """Generate graph parameter impact plot"""
        print("  ✓ Generating graph_param_impact.png")

        # Extract parameter info and detection counts
        param_data = []

        for method_name, result in graph_results.items():
            parts = method_name.split('_')
            if len(parts) >= 3:
                graph_type = parts[1]
                if graph_type == 'knn' and 'k' in parts[2]:
                    param_val = parts[2].replace('k', '')
                    param_data.append({
                        'type': 'KNN',
                        'param': f"k={param_val}",
                        'param_val': int(param_val),
                        'detections': result['num_anomalies'],
                        'detection_method': parts[-1]
                    })
                elif graph_type in ['distance', 'similarity']:
                    try:
                        param_val = float(parts[2])
                        param_data.append({
                            'type': graph_type.capitalize(),
                            'param': f"t={param_val}",
                            'param_val': param_val,
                            'detections': result['num_anomalies'],
                            'detection_method': parts[-1]
                        })
                    except:
                        pass

        if len(param_data) == 0:
            return

        df_params = pd.DataFrame(param_data)

        # Create subplots for each graph type
        graph_types = df_params['type'].unique()
        n_types = len(graph_types)

        fig, axes = plt.subplots(1, n_types, figsize=(6*n_types, 5))
        if n_types == 1:
            axes = [axes]

        for ax, graph_type in zip(axes, graph_types):
            df_subset = df_params[df_params['type'] == graph_type]

            # Group by parameter value and get mean detections
            grouped = df_subset.groupby('param_val')['detections'].mean()

            ax.plot(grouped.index, grouped.values, marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Average Anomalies Detected')
            ax.set_title(f'{graph_type} Parameter Impact')
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'graph_param_impact.png', dpi=150)
        plt.close()

    def plot_traditional_vs_graph(self, results_df: pd.DataFrame):
        """Generate traditional vs graph comparison"""
        print("  ✓ Generating traditional_vs_graph.png")

        # Get samples detected by traditional methods
        traditional_cols = [col for col in results_df.columns if col.startswith('traditional_')]
        traditional_detected = set(results_df[results_df[traditional_cols].any(axis=1)]['sample_id'])

        # Get samples detected by graph methods
        graph_cols = [col for col in results_df.columns if col.startswith('graph_')]
        graph_detected = set(results_df[results_df[graph_cols].any(axis=1)]['sample_id'])

        # Calculate overlaps
        both = len(traditional_detected.intersection(graph_detected))
        only_traditional = len(traditional_detected - graph_detected)
        only_graph = len(graph_detected - traditional_detected)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create bar chart
        categories = ['Only Traditional', 'Both', 'Only Graph']
        values = [only_traditional, both, only_graph]
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e']

        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_ylabel('Number of Samples')
        ax.set_title('Traditional vs Graph Anomaly Detection')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'traditional_vs_graph.png', dpi=150)
        plt.close()

    def plot_performance_comparison(self, metrics: Dict):
        """Generate performance comparison plot"""
        print("  ✓ Generating performance_comparison.png")

        # Extract metrics
        methods = list(metrics.keys())
        precisions = [metrics[m]['precision'] for m in methods]
        recalls = [metrics[m]['recall'] for m in methods]
        f1_scores = [metrics[m]['f1'] for m in methods]

        # Create grouped bar chart
        x = np.arange(len(methods))
        width = 0.25

        fig, ax = plt.subplots(figsize=(16, 6))

        ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1', alpha=0.8)

        ax.set_ylabel('Score')
        ax.set_title('Performance Comparison Across Methods')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=90)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'performance_comparison.png', dpi=150)
        plt.close()

    def plot_roc_curves(self, ground_truth: pd.Series, anomaly_label: Any,
                       traditional_results: Dict, graph_results: Dict):
        """Generate ROC curves"""
        print("  ✓ Generating roc_curves.png")

        y_true = (ground_truth == anomaly_label).astype(int).values

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot traditional methods
        for method_name, result in traditional_results.items():
            if result['scores'] is not None:
                try:
                    fpr, tpr, _ = roc_curve(y_true, result['scores'])
                    auc = roc_auc_score(y_true, result['scores'])
                    ax.plot(fpr, tpr, label=f"trad_{method_name} (AUC={auc:.3f})", alpha=0.7)
                except:
                    pass

        # Plot graph methods (sample a few to avoid clutter)
        graph_items = list(graph_results.items())
        step = max(1, len(graph_items) // 10)  # Sample at most 10 graph methods

        for method_name, result in graph_items[::step]:
            if result['scores'] is not None:
                try:
                    fpr, tpr, _ = roc_curve(y_true, result['scores'])
                    auc = roc_auc_score(y_true, result['scores'])
                    ax.plot(fpr, tpr, label=f"{method_name} (AUC={auc:.3f})", alpha=0.5, linestyle='--')
                except:
                    pass

        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

    def generate_all_plots(self, results_df: pd.DataFrame,
                          traditional_results: Dict, graph_results: Dict,
                          ground_truth: pd.Series = None, anomaly_label: Any = None,
                          metrics: Dict = None):
        """Generate all visualization plots"""
        print("\nGenerating visualizations...")

        # Always generate these
        self.plot_detection_heatmap(results_df)
        self.plot_method_agreement(results_df)
        self.plot_detection_distribution(results_df)
        self.plot_score_distributions(traditional_results, graph_results)
        self.plot_graph_param_impact(graph_results)
        self.plot_traditional_vs_graph(results_df)

        # Generate these only if ground truth available
        if ground_truth is not None and metrics is not None:
            self.plot_performance_comparison(metrics)
            self.plot_roc_curves(ground_truth, anomaly_label, traditional_results, graph_results)


def run_traditional_methods(X: np.ndarray, contamination: float,
                           methods: List[str]) -> Dict[str, Any]:
    """Run all traditional anomaly detection methods"""
    print("\nRunning traditional anomaly detection methods...")

    detector = TraditionalDetectors(contamination=contamination)
    results = {}

    if 'isolation_forest' in methods:
        result = detector.fit_isolation_forest(X)
        results['isolation_forest'] = result
        print(f"  - Isolation Forest: {result['predictions'].sum()} anomalies detected")

    if 'lof' in methods:
        for n in [5, 10, 20]:
            result = detector.fit_lof(X, n_neighbors=n)
            results[f'lof_n{n}'] = result
            print(f"  - LOF (n={n}): {result['predictions'].sum()} anomalies detected")

    if 'svm' in methods:
        result = detector.fit_ocsvm(X)
        results['ocsvm'] = result
        print(f"  - One-Class SVM: {result['predictions'].sum()} anomalies detected")

    if 'dbscan' in methods:
        for eps in [0.5, 1.0, 1.5]:
            result = detector.fit_dbscan(X, eps=eps)
            results[f'dbscan_eps{eps}'] = result
            print(f"  - DBSCAN (eps={eps}): {result['predictions'].sum()} anomalies detected")

    return results


def create_cytoscape_export(graph: Any, results_df: pd.DataFrame,
                           output_dir: Path):
    """Create Cytoscape visualization files"""
    print("\nCreating Cytoscape export...")

    cyto_dir = output_dir / 'cytoscape'
    cyto_dir.mkdir(parents=True, exist_ok=True)

    # Get method columns
    method_cols = [col for col in results_df.columns
                  if col not in ['sample_id', 'total_detections', 'traditional_count',
                                'graph_count', 'agreement_score']]

    # Add detection results as node attributes
    for node_id in graph.nodes():
        node_data = graph.graph.nodes[node_id]

        # Get results for this node
        sample_results = results_df[results_df['sample_id'] == node_id].iloc[0]

        # Add detection flags
        for method in method_cols:
            node_data[f'anomaly_{method}'] = bool(sample_results[method])

        # Add summary attributes
        node_data['total_detections'] = int(sample_results['total_detections'])
        node_data['traditional_detections'] = int(sample_results['traditional_count'])
        node_data['graph_detections'] = int(sample_results['graph_count'])
        node_data['agreement_score'] = float(sample_results['agreement_score'])
        node_data['consensus_anomaly'] = bool(sample_results['total_detections'] >= len(method_cols) / 2)

    # Create visualizer (pass the underlying NetworkX graph)
    viz = CytoscapeVisualizer(graph.graph, verbose=False)

    # Export to different formats
    try:
        # CYJS format for Cytoscape Desktop
        cyjs_path = cyto_dir / 'anomaly_graph.cyjs'
        viz.export_cytoscape_json(str(cyjs_path))
        print(f"  ✓ {cyjs_path.name}")
    except Exception as e:
        print(f"  - Warning: CYJS export failed - {e}")

    try:
        # HTML format for interactive web viewing
        html_path = cyto_dir / 'anomaly_graph.html'
        viz.export_html(str(html_path))
        print(f"  ✓ {html_path.name}")
    except Exception as e:
        print(f"  - Warning: HTML export failed - {e}")

    try:
        # JSON format for Cytoscape.js
        json_path = cyto_dir / 'anomaly_graph.json'
        viz.export_json(str(json_path))
        print(f"  ✓ {json_path.name}")
    except Exception as e:
        print(f"  - Warning: JSON export failed - {e}")


def save_summary_report(results_df: pd.DataFrame, traditional_results: Dict,
                       graph_results: Dict, metrics: Dict, output_dir: Path,
                       dataframe_path: str, contamination: float):
    """Save human-readable summary report"""

    report_path = output_dir / 'summary_report.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Anomaly Detection Comparison Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Dataset: {Path(dataframe_path).name}\n")
        f.write(f"Samples: {len(results_df)}\n")
        f.write(f"Contamination: {contamination:.2f}\n\n")

        f.write("Traditional Methods:\n")
        f.write("-" * 60 + "\n")
        for method_name, result in traditional_results.items():
            n_detections = result['predictions'].sum()
            f.write(f"  - {method_name}: {n_detections} detections\n")

        f.write("\nGraph-Based Methods:\n")
        f.write("-" * 60 + "\n")
        if len(graph_results) > 0:
            detection_counts = [r['num_anomalies'] for r in graph_results.values()]
            f.write(f"  - Total configurations: {len(graph_results)}\n")
            f.write(f"  - Average detections: {np.mean(detection_counts):.1f}\n")
            f.write(f"  - Min detections: {np.min(detection_counts)}\n")
            f.write(f"  - Max detections: {np.max(detection_counts)}\n")
        else:
            f.write(f"  - No graph-based methods successfully ran\n")

        f.write("\nAgreement Analysis:\n")
        f.write("-" * 60 + "\n")
        method_cols = [col for col in results_df.columns
                      if col not in ['sample_id', 'total_detections', 'traditional_count',
                                    'graph_count', 'agreement_score']]
        n_methods = len(method_cols)
        n_flagged = (results_df['total_detections'] > 0).sum()
        n_consensus = (results_df['total_detections'] >= n_methods / 2).sum()

        f.write(f"  - Samples flagged by ≥1 method: {n_flagged} ({100*n_flagged/len(results_df):.1f}%)\n")
        f.write(f"  - Samples flagged by ≥50% methods: {n_consensus} ({100*n_consensus/len(results_df):.1f}%)\n")
        f.write(f"  - Average agreement score: {results_df['agreement_score'].mean():.3f}\n")

        if metrics is not None:
            f.write("\nPerformance Metrics (with ground truth):\n")
            f.write("-" * 60 + "\n")

            # Find best methods
            f1_scores = {k: v['f1'] for k, v in metrics.items()}
            best_overall = max(f1_scores, key=f1_scores.get)

            traditional_f1 = {k: v for k, v in f1_scores.items() if k.startswith('traditional_')}
            graph_f1 = {k: v for k, v in f1_scores.items() if k.startswith('graph_')}

            best_traditional = max(traditional_f1, key=traditional_f1.get) if traditional_f1 else None
            best_graph = max(graph_f1, key=graph_f1.get) if graph_f1 else None

            if best_traditional:
                f.write(f"  - Best Traditional: {best_traditional} (F1={traditional_f1[best_traditional]:.3f})\n")
            if best_graph:
                f.write(f"  - Best Graph: {best_graph} (F1={graph_f1[best_graph]:.3f})\n")
            f.write(f"  - Best Overall: {best_overall} (F1={f1_scores[best_overall]:.3f})\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Results saved to: {output_dir}\n")
        f.write("=" * 60 + "\n")

    print(f"\nSummary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare traditional and graph-based anomaly detection methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (unsupervised)
  python anomaly_task_test.py -d data/heart_failure.csv

  # With ground truth evaluation
  python anomaly_task_test.py -d data/diabetes.csv --target outcome --anomaly-label 1

  # Custom configuration
  python anomaly_task_test.py -d data.csv --contamination 0.05 --methods isolation_forest,lof
        """
    )

    # Required arguments
    parser.add_argument('-d', '--dataframe', required=True,
                       help='Path to input dataframe (CSV)')

    # Optional arguments
    parser.add_argument('--target', default=None,
                       help='Target column for evaluation (if ground truth exists)')
    parser.add_argument('--anomaly-label', default=None,
                       help='Value in target column indicating anomaly')
    parser.add_argument('--contamination', type=float, default=0.1,
                       help='Expected anomaly proportion (default: 0.1)')
    parser.add_argument('--output-dir', default='./anomaly_results/',
                       help='Output directory (default: ./anomaly_results/)')
    parser.add_argument('--no-cytoscape', action='store_true',
                       help='Skip Cytoscape visualization generation')
    parser.add_argument('--methods', default='isolation_forest,lof,svm,dbscan',
                       help='Comma-separated list of traditional methods (default: all)')
    parser.add_argument('--graph-methods', default='knn,distance,similarity',
                       help='Graph construction methods (default: all)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress')

    args = parser.parse_args()

    # Parse method lists
    traditional_methods = [m.strip() for m in args.methods.split(',')]
    graph_methods = [m.strip() for m in args.graph_methods.split(',')]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Anomaly Detection Comparison Script")
    print("=" * 60)

    # 1. Load data
    print("\nLoading dataframe...")
    df = pd.read_csv(args.dataframe)
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {list(df.columns)}")

    # 2. Preprocess data
    print("\nPreprocessing data...")

    # Separate target if specified
    if args.target:
        if args.target not in df.columns:
            print(f"Error: Target column '{args.target}' not found in dataframe")
            sys.exit(1)

        ground_truth = df[args.target].copy()
        df_for_detection = df.drop(columns=[args.target])

        # Determine anomaly label
        if args.anomaly_label is not None:
            # Try to convert to appropriate type
            unique_vals = ground_truth.unique()
            if len(unique_vals) == 2:
                if isinstance(unique_vals[0], str):
                    anomaly_label = str(args.anomaly_label)
                elif isinstance(unique_vals[0], (int, np.integer)):
                    anomaly_label = int(args.anomaly_label)
                else:
                    anomaly_label = float(args.anomaly_label)
            else:
                anomaly_label = args.anomaly_label
        else:
            # Auto-detect: assume minority class is anomaly
            value_counts = ground_truth.value_counts()
            anomaly_label = value_counts.idxmin()
            print(f"  - Auto-detected anomaly label: {anomaly_label}")
    else:
        ground_truth = None
        anomaly_label = None
        df_for_detection = df.copy()

    # Preprocess for traditional methods
    from sklearn.preprocessing import StandardScaler

    # Select only numeric columns
    numeric_cols = df_for_detection.select_dtypes(include=[np.number]).columns
    df_numeric = df_for_detection[numeric_cols].fillna(df_for_detection[numeric_cols].mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    print(f"  - Preprocessed shape: {X_scaled.shape}")

    # 3. Run traditional methods
    traditional_results = run_traditional_methods(
        X_scaled,
        contamination=args.contamination,
        methods=traditional_methods
    )

    # 4. Build graph variants and run graph-based detection
    graph_tester = GraphAnomalyTester(
        df_for_detection,
        X_scaled,
        contamination=args.contamination,
        verbose=args.verbose
    )

    graph_tester.build_graph_variants(graph_methods=graph_methods)
    graph_results = graph_tester.run_all_combinations()

    # 5. Aggregate results
    aggregator = ResultsAggregator(n_samples=len(df))
    results_df = aggregator.aggregate(traditional_results, graph_results)

    # Save results to CSV
    results_csv_path = output_dir / 'detection_results.csv'
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nDetection results saved to: {results_csv_path}")

    # 6. Compute evaluation metrics (if ground truth)
    metrics = None
    if ground_truth is not None:
        metrics = aggregator.compute_metrics(
            ground_truth,
            anomaly_label,
            traditional_results,
            graph_results
        )

        metrics_json_path = output_dir / 'method_metrics.json'
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Evaluation metrics saved to: {metrics_json_path}")

    # 7. Generate visualizations
    visualizer = Visualizer(output_dir)
    visualizer.generate_all_plots(
        results_df,
        traditional_results,
        graph_results,
        ground_truth=ground_truth,
        anomaly_label=anomaly_label,
        metrics=metrics
    )

    # 8. Create Cytoscape export (if not disabled)
    if not args.no_cytoscape and len(graph_tester.graphs) > 0:
        # Use the first KNN graph as representative
        representative_graph = None
        for key, graph in graph_tester.graphs.items():
            if 'knn' in key:
                representative_graph = graph
                break

        if representative_graph is None:
            representative_graph = list(graph_tester.graphs.values())[0]

        create_cytoscape_export(representative_graph, results_df, output_dir)

    # 9. Save summary report
    save_summary_report(
        results_df,
        traditional_results,
        graph_results,
        metrics,
        output_dir,
        args.dataframe,
        args.contamination
    )

    print("\n" + "=" * 60)
    print(f"All results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
