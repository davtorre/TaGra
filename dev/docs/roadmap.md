# TaGra Development Roadmap

## Completed

### v0.3.0 Code Restructuring (6 Feb 2026)

Restructured from monolithic design (6 flat modules) to modular architecture (9 subpackages).

| Phase | What | Status |
|-------|------|--------|
| 1 | Core module: TaGraGraph, GraphMetadata, enums | Done |
| 2 | Construction module: KNN, distance, similarity constructors + registry | Done |
| 3 | Preprocessing module: pipeline, scaling, encoding, missing, manifold | Done |
| 4 | IO module: readers, writers, Cytoscape/GraphML exporters | Done |
| 5 | Analysis module: metrics, neighborhood, community, report | Done |
| 6 | Visualization module: graph plots, heatmaps, distributions | Done |
| 7 | ML module: label propagation, augmentation, anomaly detection, imputation | Done |
| 8 | Backward compatibility layer with deprecation warnings | Done |
| 9 | Test reorganization (unit + integration) | Done |

New public API: `tagra.from_dataframe()`, `tagra.analyze()`, `tagra.visualize()`.
Legacy API preserved in `tagra.compat`.

### Anomaly Detection Comparison Tool (9 Feb 2026)

Script comparing sklearn anomaly detectors (Isolation Forest, LOF, One-Class SVM, DBSCAN) with TaGra's `GraphAnomalyDetector` across multiple graph configurations.

| Feature | Status |
|---------|--------|
| Traditional detectors (4 types, 8 configs) | Done |
| Graph-based detection (16 graphs x 3 methods = 48 configs) | Done |
| Results aggregation and agreement analysis | Done |
| 6 visualization plots | Done |
| Cytoscape export (.cyjs, .html, .json) | Done |
| Ground truth evaluation (precision, recall, F1, ROC) | Done |
| CLI interface | Done |

Script: `dev/tasks/anomaly/anomaly_task_test.py`

Tested on diabetes dataset (768 samples, 9 features). All outputs generated successfully.

---

## In Progress

### 0. Clustering Task: TaGra as a Generalization of DBSCAN (16 Feb 2026)

Preliminary validation that TaGra's graph-based clustering captures meaningful structure, before testing downstream ML tasks.

| Step | Status |
|------|--------|
| Theoretical analysis (DBSCAN = distance graph + degree filter + components) | Done |
| Comparison plan with DBCV metric | Done |
| Comparison script (`clustering_comparison.py`) | Done |
| Config file (`clustering_config.json`) | Done |
| Run on test datasets | Pending |
| Analyze results | Pending |

Script: `dev/tasks/clustering/clustering_comparison.py`

The script compares:
- DBSCAN baseline (single eps, min_samples from config)
- TaGra distance-threshold graph (same eps, validates theoretical equivalence)
- TaGra KNN graph (various k)
- TaGra similarity graph (various thresholds)

All graph methods use a degree filter = min_samples - 1 for fair comparison.

Metrics: DBCV, Silhouette, V-measure (agreement with DBSCAN).

---

## Planned

### 1. Semi-Supervised Learning / Label Propagation

**Priority**: High (core contribution for follow-up paper)

Test TaGra's `LabelPropagator` against standard semi-supervised baselines.

- Setup: Vary labeled fraction (1%, 5%, 10%, 20%)
- Baselines: Self-training, Label Spreading, GCN
- Metrics: Accuracy, Macro-F1, AUC on held-out test set
- Ablation: Compare KNN vs distance vs similarity graph construction

### 2. Graph-Based Data Augmentation

**Priority**: High

Test `GraphAwareAugmenter` against SMOTE and variants.

- Setup: Imbalanced datasets (1:10, 1:20, 1:50 minority ratio)
- Baselines: Random oversampling, SMOTE, ADASYN, borderline-SMOTE
- Metrics: Minority class precision/recall/F1, balanced accuracy

### 3. Explainable Anomaly Detection

**Priority**: High

Evaluate `GraphAnomalyDetector` with injected anomalies.

- Setup: Inject label flips and feature corruptions at known rates
- Baselines: Isolation Forest, LOF, One-Class SVM
- Metrics: Precision, Recall, F1, AUC-ROC
- User study: Are TaGra explanations actionable?

### 4. Missing Value Imputation

**Priority**: Medium

Test `GraphImputer` against standard imputation methods.

- Setup: Artificially introduce missingness (MCAR, MAR, MNAR)
- Baselines: Mean, KNN, MICE, missForest
- Metrics: RMSE, MAE, downstream classification accuracy

### 5. Feature Importance via Graph Perturbation

**Priority**: Medium

Measure how graph structure changes when features are removed.

- Remove each feature, rebuild graph, measure edge Jaccard / homophily change
- Compare rankings against permutation importance, SHAP

### 6. Longitudinal Analysis

**Priority**: Lower (requires temporal datasets)

Build graphs at multiple timepoints, track neighborhood evolution.

- Neighborhood stability score
- Class drift velocity
- Community transition probabilities
- Boundary proximity

### 7. Documentation Update

**Priority**: High (deferred)

- Update README.md with v0.3.0 API, ML module, migration guide
- Update examples

---

## Key Questions to Resolve

1. Which graph construction method works best for each task?
2. How sensitive are results to hyperparameters (k, thresholds)?
3. Computational ceiling? Profile on datasets of increasing size.
4. How to handle new/unseen data points? Rebuild graph vs incremental addition.
5. Target venue? ML venue (methodology) vs medical informatics (applications) vs software (tool).
