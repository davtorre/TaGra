# Scientific Log: TaGra Follow-Up Paper

This document tracks the scientific reasoning behind each step of the TaGra follow-up work. For each decision, we record *why* we made it.

---

## 0. Clustering: TaGra Generalizes DBSCAN

### Context

In many domains, including medical data analysis, discovering latent groupings in unlabeled data is a fundamental first step. Clustering allows identifying patient subgroups, disease subtypes, or risk strata without requiring labels. Density-based clustering — and DBSCAN in particular — is widely used because it handles non-convex cluster shapes and naturally identifies noise points. However, DBSCAN's reliance on a fixed Euclidean distance threshold limits its applicability to high-dimensional, heterogeneous, or domain-specific data where Euclidean distance is not the right notion of similarity.

We use clustering as the first and most basic validation of TaGra's graph construction: if the graph captures meaningful structure, it should at minimum recover clusters as well as DBSCAN. This test precedes all downstream ML tasks (label propagation, augmentation, anomaly detection) and provides the theoretical foundation for them.

### Connection to DBSCAN: the graph equivalence

DBSCAN operates with two parameters: **ε (epsilon)**, a distance threshold, and **min_samples**, a minimum neighborhood size. It classifies every point into one of three categories:

- **Core point**: has at least `min_samples` other points within distance ε. Core points form the dense skeleton of clusters.
- **Border point**: not core, but within ε of at least one core point. Assigned to a nearby cluster, does not propagate membership further.
- **Noise point**: neither core nor border. Labeled −1 and discarded.

DBSCAN can be exactly decomposed into three graph-theoretic operations:

| Step | DBSCAN operation | Graph equivalent |
|------|-----------------|------------------|
| 1. Neighborhood | ε-ball: connect all points with d(i,j) ≤ ε | Distance-threshold graph with threshold = ε |
| 2. Density filter | Core point criterion: degree ≥ min_samples | Node degree filter: keep nodes with degree ≥ min_samples − 1 |
| 3. Clustering | Density-reachable connected components | Connected components of the filtered graph |

The −1 in the degree filter arises because sklearn's DBSCAN counts a point as one of its own neighbors (so `min_samples` includes the point itself), whereas a graph has no self-loops. A node in the distance-threshold graph has degree equal to the number of *other* points within ε, so the threshold is `min_samples − 1`.

**Critical implementation note**: node degrees must be computed once on the original unfiltered graph, not iteratively. Removing a node reduces its neighbors' degrees, but DBSCAN's core-point criterion is defined on the original neighborhood. Iterative removal would over-prune and would not be equivalent to DBSCAN.

**Border points**: the degree-filtered graph captures only core-point clusters. Border points — non-core points with a core-point neighbor — are not assigned. In practice this gap is minor: border points are few, their assignment is order-dependent in DBSCAN, and for quality metrics like DBCV the core structure is what matters.

**Counterexample for naive connected components**: suppose two dense clusters are connected by a thin chain of points, each with only 2 neighbors within ε. DBSCAN with `min_samples=5` correctly identifies the chain points as noise and keeps the two clusters separate. Connected components without a degree filter would merge the two clusters into one. The degree filter is therefore not optional — it is what makes the graph equivalence exact.

### Why TaGra extends DBSCAN

DBSCAN is locked to step 1: its neighborhood is always defined by a fixed Euclidean distance threshold ε. TaGra generalizes step 1 by offering alternative graph construction methods, while steps 2 (degree filter) and 3 (connected components) remain identical:

- **Distance-threshold graph** (threshold = ε): exactly reproduces DBSCAN's neighborhood. This is the special case that validates the equivalence.
- **KNN graph** (k neighbors): connects each node to its k nearest neighbors regardless of absolute distance. Adapts to local density — a known weakness of DBSCAN's fixed ε, which struggles when clusters have different densities. The degree filter threshold is still `min_samples − 1`, applied after graph construction.
- **Cosine similarity graph** (threshold on cosine similarity): connects nodes whose feature vectors point in similar directions. Captures angular relationships that Euclidean distance misses, useful for high-dimensional or sparse data.

The generalization is not about the distance metric (sklearn's DBSCAN also supports non-Euclidean metrics). The key difference is the *type* of neighborhood structure: threshold-based, adaptive (KNN), or similarity-based. Each encodes a different notion of "nearness" and produces a topologically different graph, leading to different cluster geometries.

The homophily metrics already computed by TaGra can further validate whether the resulting graph structure is appropriate for the task.

### Clustering with TaGra: algorithmic steps

```
1. Preprocess the dataframe (scale numeric features, encode categorical ones)
2. Build a TaGra graph using one of:
     - Distance-threshold (threshold = ε)          → equivalent to DBSCAN neighborhood
     - KNN (k neighbors)                            → adaptive to local density
     - Cosine similarity (threshold on similarity)  → angular neighborhoods
3. Compute node degrees on the original graph (do NOT iterate)
4. Remove all nodes with degree < min_samples − 1  → these become noise (label = −1)
5. Extract connected components of the filtered graph → each component is a cluster
6. Discard components below a minimum size (optional)
```

Parameters:
- `ε` / `distance_threshold`: same value used in both DBSCAN and the distance-threshold graph, enabling direct comparison
- `min_samples`: DBSCAN parameter; graph degree filter = `min_samples − 1`
- `k`: KNN parameter; the degree filter is still `min_samples − 1` for cross-method comparability

### Clustering comparison strategy

The comparison script (`dev/tasks/clustering/clustering_comparison.py`) runs the following methods on the same standardized data and evaluates them with common metrics:

1. **DBSCAN baseline**: sklearn DBSCAN with `(eps, min_samples)` from the config
2. **TaGra distance-threshold graph**: same `eps` value, degree filter = `min_samples − 1`. Should reproduce DBSCAN — this is the **validation check** for the graph equivalence
3. **TaGra KNN graph**: several values of `k`, same degree filter
4. **TaGra cosine similarity graph**: several similarity thresholds, same degree filter

Metrics:
- **DBCV** (Density-Based Clustering Validation, [−1, +1]): the primary metric; designed for density-based clusters, does not require ground truth
- **Silhouette coefficient** ([−1, +1]): secondary metric; requires at least 2 clusters and 2 non-noise points
- **V-measure** against DBSCAN: measures structural agreement between each graph clustering and the DBSCAN output, regardless of cluster quality. A score of 1 means the clusterings are identical; 0 means no shared structure

A key design constraint: the degree filter is always set to `min_samples − 1` across all graph methods, so the density requirement is held constant and only the neighborhood definition varies.

### Results on Taipei dataset (taipei_preprocessed.csv, 926 patients, 26 features)

The first run on the Taipei colorectal cancer dataset revealed a mismatch between the default parameters and the data's geometry. With `eps=1.0` and `min_samples=5` on 26 standardized dimensions, DBSCAN found zero clusters: every point was classified as noise. The distance-threshold graph confirmed this — only 40 edges formed across 926 nodes at threshold 1.0, so no node reached degree 4. This is a well-known consequence of the **curse of dimensionality**: in high-dimensional spaces, pairwise Euclidean distances concentrate around a common value, making any fixed ε either too small (no neighbors) or too large (everything connects).

The KNN graphs showed the opposite extreme: with k ≥ 5 and degree filter 4, every node survived and the entire dataset collapsed into a single connected component. No separation was possible because KNN graphs are always connected for sufficiently large k relative to the graph's structure.

The cosine similarity graphs produced the most informative results. At threshold 0.8, 16 clusters were found among 60 core nodes (866 noise points), with a silhouette of 0.45. This indicates that roughly 60 patients have feature vectors that are highly similar to at least 4 others, forming 16 tight groups. Whether these groups correspond to clinically meaningful subtypes requires further investigation.

The primary conclusion from this run is that **the parameters need calibration before the comparison is interpretable**. A k-distance plot would identify a meaningful ε for DBSCAN in this dimensionality, and dimensionality reduction (PCA or UMAP) in the preprocessing step would make Euclidean distances more meaningful. The similarity-based graph, which is less sensitive to dimensionality than Euclidean threshold methods, is likely the most promising construction method for this dataset.

### Experimental design

Compare DBCV scores across:

1. **DBSCAN baseline**: standard sklearn DBSCAN.
2. **Distance-threshold graph + degree filter**: should approximate DBSCAN (validates theory).
3. **KNN graph + degree filter**: tests adaptive neighborhoods.
4. **Similarity graph + degree filter**: tests angular similarity.

All methods use the same preprocessed data. The degree filter is set to min_samples - 1 throughout. V-measure between clusterings measures agreement.

**Expected outcomes:**
- Method 2 should closely match DBSCAN (validates the equivalence).
- Methods 3/4 may outperform DBSCAN on datasets with varying-density clusters.

---

## 1. Semi-Supervised Learning via Label Propagation

### Motivation

In many domains (especially medical), labeled data is scarce. The similarity graph provides a natural structure for propagating labels: if a patient's 5 nearest neighbors are 4 diseased and 1 healthy, we can infer their likely label.

### Why TaGra adds value over standard label propagation

Standard label propagation (Zhu & Ghahramani 2002) requires constructing a similarity graph as a prerequisite. TaGra provides this graph in a principled, configurable way. The multiple construction methods (KNN, distance, similarity) allow tuning for different data characteristics. The homophily metrics already computed by TaGra validate whether the graph is suitable for propagation.

### Evaluation strategy

- Vary labeled fraction: 1%, 5%, 10%, 20%
- Baselines: self-training, label spreading, GCN
- Metrics: accuracy, macro-F1, AUC
- Ablation: which graph construction method works best

---

## 2. Graph-Aware Data Augmentation

### Motivation

Imbalanced datasets lead to biased models. SMOTE interpolates between k-nearest neighbors but ignores global structure. TaGra's graph ensures interpolation only occurs between genuinely similar data points that are actually connected.

### Why graph-aware augmentation is better than SMOTE

- SMOTE's k-nearest neighbors may span a decision boundary (interpolating between two classes).
- TaGra's graph can enforce same-class connectivity via community structure.
- The graph provides a natural validation: do synthetic points fall within their expected neighborhood?

### Evaluation strategy

- Imbalanced datasets: minority ratio 1:10, 1:20, 1:50
- Baselines: random oversampling, SMOTE, ADASYN, borderline-SMOTE
- Metrics: minority class F1, balanced accuracy
- Visualization: t-SNE of original vs augmented data

---

## 3. Explainable Anomaly Detection

### Motivation

Traditional anomaly detection (Isolation Forest, LOF) provides scores but limited explanations. In medical contexts, clinicians need to understand *why* a patient is anomalous.

### Why graph-based anomaly detection is different

TaGra anomalies are defined by graph position. The explanation is built-in:

| Indicator | What it measures |
|-----------|-----------------|
| Isolation score | 1 - (degree / max_degree). High = few connections. |
| Class incongruence | Fraction of neighbors with different class. High = disagreement with neighborhood. |
| Local density ratio | node_degree / mean(neighbor_degrees). Low = sparse in dense region. |

LOF gives a score. TaGra gives a score + explanation ("this patient is anomalous because 80% of their neighbors have a different outcome").

### Preliminary results (9 Feb 2026)

The anomaly detection comparison tool was built and tested on the diabetes dataset. It systematically compares sklearn detectors vs TaGra across 48+ configurations.

### Evaluation strategy

- Inject known anomalies (label flips, feature corruptions)
- Baselines: Isolation Forest, LOF, One-Class SVM
- Metrics: precision, recall, F1, AUC
- User study: clinicians rate explanation quality

---

## 4. Missing Value Imputation

### Motivation

Real-world datasets contain missing values. Mean/median imputation ignores structure. TaGra imputes using graph neighbors, which is more principled than global statistics because it uses local similarity.

### Why this is different from KNN imputation

KNN imputation uses the same idea (impute from similar points), but TaGra's graph is built once and reused. The graph construction can use only complete features, then impute on the graph. Edge weights can influence imputation quality.

### Evaluation strategy

- Artificially introduce missingness: MCAR, MAR, MNAR
- Baselines: mean, KNN, MICE, missForest
- Metrics: RMSE, MAE, downstream classification accuracy

---

## 5. Feature Importance via Graph Perturbation

### Motivation

Understanding which features drive similarity relationships helps with feature selection and interpretation.

### Approach

Remove each feature, rebuild the graph, measure how the graph changes:
- Edge Jaccard: |E_original ∩ E_perturbed| / |E_original ∪ E_perturbed|
- Homophily change: |H_original - H_perturbed|
- Modularity change

Features that dramatically change the graph are important for the similarity structure.

---

## 6. Longitudinal Analysis

### Motivation

Tracking patient trajectories over time requires understanding how neighborhood relationships evolve. TaGra can build graphs at multiple timepoints and track:

- **Neighborhood stability**: Jaccard between neighbor sets at t1 and t2. Low stability = condition changing.
- **Class drift velocity**: Rate of neighborhood composition shift toward a different class. Positive drift toward "disease" = early warning.
- **Community transitions**: Markov model of disease progression based on community membership over time.
- **Boundary proximity**: How close a patient is to the decision boundary (mixed neighborhood).

---

## Paper Outline

### Working titles

1. "Beyond Visualization: Leveraging Graph-Based Data Representations for Semi-Supervised Learning and Anomaly Detection"
2. "TaGra-ML: A Graph-Based Framework for Semi-Supervised Learning, Data Augmentation, and Explainable Anomaly Detection"

### Structure

1. **Introduction**: TaGra transforms tables to graphs. Previously used for visualization. The graph enables ML tasks that leverage neighborhood structure.
2. **Related work**: Graph-based SSL, data augmentation (SMOTE), anomaly detection, graph construction from tabular data.
3. **Methods**: Graph construction recap. Label propagation algorithm. Augmentation strategy. Anomaly scoring and explanation.
4. **Experiments**: Datasets (heart failure, diabetes, UCI benchmarks). Semi-supervised (vary label fraction). Augmentation (vary imbalance). Anomaly detection (inject anomalies). Computational efficiency.
5. **Results**: Tables and figures for each task. Key findings.
6. **Discussion**: When does TaGra work best? Limitations. Future directions (longitudinal, GNNs).
7. **Conclusion**: Graph representations of tabular data enable neighborhood-aware ML.

### Target venues

- ML: JMLR, NeurIPS workshop
- Medical informatics: Journal of Biomedical Informatics
- Software: JOSS, SoftwareX
