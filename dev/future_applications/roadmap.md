# TaGra Future Applications Roadmap

## Executive Summary

TaGra's current strength lies in visualization and exploratory analysis. However, the true potential of transforming tabular data into a graph representation lies in **leveraging explicit neighborhood relationships for downstream machine learning tasks**. This document outlines potential applications where graph-based data modeling provides genuine added value over traditional approaches.

---

## Part 1: Potential Applications

### 1. Semi-Supervised Learning / Label Propagation

**The Problem**: In many domains (especially medical), labeled data is scarce and expensive to obtain, while unlabeled data is abundant.

**How TaGra Helps**: The similarity graph provides a natural structure for propagating labels from labeled to unlabeled nodes. If a patient is connected to 5 labeled patients (4 with disease, 1 without), we can infer their likely label.

**Why It's Better Than Alternatives**:
- Standard label propagation requires constructing a similarity graph anyway - TaGra provides a principled, configurable way to do this
- The multiple graph construction methods (KNN, distance, similarity) allow tuning for different data characteristics
- The homophily metrics can validate whether the graph structure is suitable for label propagation

**Algorithmic Sketch**:
```
1. Build TaGra graph from mixed labeled/unlabeled data
2. Initialize labels: known labels fixed, unknown labels set to uniform distribution
3. Iterate until convergence:
   For each unlabeled node i:
     P(label_i = c) = sum over neighbors j of [P(label_j = c)] / degree(i)
4. Assign labels based on final probabilities
```

**Evaluation Strategy**: Compare against standard semi-supervised methods (self-training, co-training, graph-based SSL baselines) on benchmark datasets with varying label fractions (1%, 5%, 10% labeled).

---

### 2. Graph-Based Data Augmentation / Synthetic Data Generation

**The Problem**: Imbalanced datasets (common in medical applications) lead to biased models. Traditional augmentation (SMOTE, random oversampling) can generate unrealistic samples.

**How TaGra Helps**: Generate synthetic samples by interpolating between connected nodes. The graph structure ensures interpolation only occurs between genuinely similar data points.

**Why It's Better Than Alternatives**:
- SMOTE interpolates between k-nearest neighbors but doesn't consider global structure
- TaGra's graph can incorporate similarity thresholds, ensuring interpolation only between sufficiently similar points
- Community structure can guide augmentation (augment within communities, not across)

**Algorithmic Sketch**:
```
1. Build TaGra graph from training data
2. Identify minority class nodes
3. For each minority node i:
   a. Find same-class neighbors N_same(i)
   b. For each neighbor j in N_same(i):
      - Generate synthetic point: x_new = alpha * x_i + (1-alpha) * x_j
      - where alpha ~ Uniform(0.3, 0.7) to stay near the middle
4. Optionally: validate synthetic points by checking their predicted neighborhood
```

**Evaluation Strategy**: Compare classifier performance (AUC, F1, balanced accuracy) when trained on:
- Original imbalanced data
- SMOTE-augmented data
- TaGra-augmented data

---

### 3. Explainable Anomaly Detection

**The Problem**: Traditional anomaly detection methods (Isolation Forest, LOF) provide scores but limited explanations. In medical contexts, clinicians need to understand *why* a data point is anomalous.

**How TaGra Helps**: Anomalies are nodes with unusual neighborhood characteristics. The explanation is built-in: "This patient is anomalous because their 5 nearest neighbors all have outcome=0, but they have outcome=1."

**Why It's Better Than Alternatives**:
- LOF gives a score; TaGra gives a score + explanation
- Isolation Forest isolates points; TaGra shows *which* normal points they differ from
- The graph visualization makes anomalies immediately visible (isolated nodes, nodes with cross-class edges)

**Anomaly Indicators**:
| Indicator | Formula | Interpretation |
|-----------|---------|----------------|
| Isolation score | 1 - (degree / max_degree) | High = potential outlier |
| Class incongruence | fraction of neighbors with different class | High = potential labeling error or interesting case |
| Local density ratio | node_degree / mean(neighbor_degrees) | Low = sparse region, potential outlier |

**Evaluation Strategy**:
- Inject known anomalies into datasets
- Compare detection rates and false positive rates against LOF, Isolation Forest, One-Class SVM
- Conduct user study: do clinicians find TaGra explanations more actionable?

---

### 4. Missing Value Imputation

**The Problem**: Real-world datasets, especially EHRs, contain missing values. Mean/median imputation ignores data structure; model-based imputation (MICE, missForest) can be computationally expensive.

**How TaGra Helps**: Impute missing values using the values of graph neighbors. This is more principled than global statistics because it uses local similarity structure.

**Why It's Better Than Alternatives**:
- Mean imputation: ignores all structure
- KNN imputation: similar idea, but TaGra's graph can be built on available features, then used to impute
- The graph can be built once and reused for multiple imputation tasks

**Algorithmic Sketch**:
```
1. Build TaGra graph using only complete features (or features with low missingness)
2. For each node i with missing value in feature f:
   a. Find neighbors N(i) with observed values for f
   b. Impute: x_i[f] = weighted_mean(x_j[f] for j in N(i))
      - Weights can be inverse distance or uniform
3. Optionally: iterate (rebuild graph with imputed values, re-impute)
```

**Evaluation Strategy**:
- Take complete datasets, artificially introduce missingness (MCAR, MAR, MNAR patterns)
- Compare imputation accuracy (RMSE, MAE) against mean, KNN, MICE, missForest
- Measure downstream task performance (classification accuracy with imputed data)

---

### 5. Feature Importance via Graph Perturbation

**The Problem**: Understanding which features drive similarity relationships helps with feature selection and model interpretation.

**How TaGra Helps**: Measure how graph structure changes when features are removed or perturbed. Features that dramatically change the graph are important for the similarity structure.

**Algorithmic Sketch**:
```
1. Build baseline TaGra graph G_0 with all features
2. Compute baseline metrics: homophily H_0, modularity M_0, edge set E_0
3. For each feature f:
   a. Build graph G_f without feature f
   b. Compute:
      - Edge Jaccard: |E_0 ∩ E_f| / |E_0 ∪ E_f|
      - Homophily change: |H_0 - H_f|
      - Modularity change: |M_0 - M_f|
   c. Importance(f) = weighted combination of changes
4. Rank features by importance
```

**Evaluation Strategy**:
- Compare feature rankings against established methods (permutation importance, SHAP)
- Validate: does removing "important" features degrade classification more than removing "unimportant" ones?

---

### 6. Data Validation and Quality Control

**The Problem**: Data entry errors, labeling mistakes, and measurement artifacts corrupt datasets. Manual review is infeasible for large datasets.

**How TaGra Helps**: Flag data points whose graph position is inconsistent with their attributes.

**Validation Checks**:
| Check | Condition | Interpretation |
|-------|-----------|----------------|
| Label inconsistency | Node has label L but >80% of neighbors have label L' | Possible labeling error |
| Feature outlier | Node's feature values differ significantly from all neighbors | Possible measurement error |
| Impossible island | Small connected component with unusual class composition | Possible systematic error in data subset |

**Evaluation Strategy**:
- Inject known errors (label flips, value corruptions) into clean datasets
- Measure precision/recall of error detection
- Compare against statistical outlier detection methods

---

### 7. Longitudinal Data Analysis (Detailed)

**The Problem**: Tracking patient trajectories over time is crucial for understanding disease progression, treatment response, and prognosis.

**How TaGra Helps**: Build graphs at multiple timepoints and track how each patient's neighborhood evolves.

#### 7.1 Neighborhood Stability Score

Measures how consistent a patient's local neighborhood is over time.

```
Stability(i, t1, t2) = |N(i, t1) ∩ N(i, t2)| / |N(i, t1) ∪ N(i, t2)|
```

**Interpretation**:
- High stability: patient's condition is consistent
- Low stability: patient's similarity relationships are changing (possible disease progression or treatment effect)

#### 7.2 Class Drift Velocity

Measures the rate at which a patient's neighborhood composition shifts toward a different class.

```
ClassComposition(i, t, c) = |{j ∈ N(i,t) : label(j) = c}| / |N(i,t)|

DriftVelocity(i, t1→t2, c) = [ClassComposition(i, t2, c) - ClassComposition(i, t1, c)] / (t2 - t1)
```

**Interpretation**:
- Positive drift toward "disease" class: early warning signal
- Negative drift toward "healthy" class: treatment may be working

#### 7.3 Community Transition Probability

Estimate the probability of transitioning between graph communities over time.

```
P(community_t2 = B | community_t1 = A) = count(A→B transitions) / count(patients in A at t1)
```

**Application**: Build Markov models of disease progression based on community transitions.

#### 7.4 Boundary Proximity

Identify patients near the "decision boundary" between classes.

```
BoundaryProximity(i) = 1 - |ClassComposition(i, majority_class) - 0.5| * 2
```

**Interpretation**:
- High boundary proximity: patient is in a mixed neighborhood, outcome uncertain
- Low boundary proximity: patient is firmly in one class region

#### 7.5 Concrete Clinical Application

**Scenario**: Heart failure patients with EHR data at 6-month intervals

**Workflow**:
1. Build TaGra graphs at T0, T6, T12, T18, T24
2. For each patient, compute:
   - Neighborhood stability trajectory
   - Drift velocity toward mortality class
   - Community membership over time
3. Flag patients with:
   - Decreasing stability (condition changing)
   - Positive drift velocity toward mortality
   - Community transition from "stable" to "at-risk"
4. Output: Risk scores with explanations for clinical review

---

## Part 2: Follow-Up Paper Sketch

### Title Options

1. "Beyond Visualization: Leveraging Graph-Based Data Representations for Semi-Supervised Learning and Anomaly Detection"
2. "TaGra-ML: A Graph-Based Framework for Semi-Supervised Learning, Data Augmentation, and Explainable Anomaly Detection"
3. "From Tables to Graphs to Predictions: Neighborhood-Aware Machine Learning with TaGra"

### Abstract (Draft)

> Transforming tabular data into graph representations enables a class of machine learning methods that leverage explicit neighborhood relationships. While our previous work introduced TaGra for data visualization, here we demonstrate that the resulting graph structure provides genuine value for downstream tasks. We present TaGra-ML, an extension that enables (1) semi-supervised learning via label propagation, (2) graph-aware data augmentation for imbalanced datasets, and (3) explainable anomaly detection. On [N] benchmark datasets from the medical domain, we show that TaGra-ML's semi-supervised learning outperforms standard baselines by [X]% when only [Y]% of labels are available. Our augmentation strategy improves minority class F1 by [Z]% compared to SMOTE. For anomaly detection, TaGra-ML achieves comparable detection rates to Isolation Forest while providing interpretable explanations preferred by clinicians in a user study. Code and reproducible experiments are available at [URL].

### 1. Introduction

- Recap TaGra's contribution: table → graph transformation with visualization
- Identify the gap: visualization is useful but doesn't leverage the graph for ML tasks
- Thesis: the graph structure enables neighborhood-aware ML that outperforms methods ignoring this structure
- Contributions:
  1. Semi-supervised learning module with label propagation
  2. Graph-aware data augmentation
  3. Explainable anomaly detection
  4. Comprehensive evaluation on medical datasets

### 2. Related Work

#### 2.1 Graph-Based Semi-Supervised Learning
- Label propagation (Zhu & Ghahramani, 2002)
- Graph neural networks for semi-supervised learning
- How TaGra-ML differs: principled graph construction from tabular data with configurable methods

#### 2.2 Data Augmentation for Imbalanced Data
- SMOTE and variants
- Generative approaches (GANs, VAEs)
- How TaGra-ML differs: augmentation respects graph structure, not just local neighborhoods

#### 2.3 Anomaly Detection
- Isolation Forest, LOF, One-Class SVM
- Explainable anomaly detection
- How TaGra-ML differs: anomalies defined by graph position, explanations via neighborhood analysis

### 3. Methods

#### 3.1 Graph Construction (recap from TaGra paper)
- KNN, distance threshold, similarity threshold methods
- Preprocessing pipeline

#### 3.2 Semi-Supervised Learning via Label Propagation
- Algorithm description
- Confidence estimation
- Handling of disconnected components

#### 3.3 Graph-Aware Data Augmentation
- Interpolation strategy
- Community-aware augmentation
- Validation of synthetic samples

#### 3.4 Explainable Anomaly Detection
- Anomaly scoring based on graph metrics
- Explanation generation
- Visualization of anomalies

### 4. Experiments

#### 4.1 Datasets
- Heart Failure EHRs (from original TaGra paper)
- Diabetes EHRs (from original TaGra paper)
- 2-3 additional medical datasets for generalization
- UCI benchmark datasets for comparison with existing literature

#### 4.2 Semi-Supervised Learning Experiments
- **Setup**: Vary labeled fraction (1%, 5%, 10%, 20%)
- **Baselines**: Self-training, Label Spreading, Graph Convolutional Networks
- **Metrics**: Accuracy, Macro-F1, AUC on held-out test set
- **Ablation**: Compare KNN vs distance vs similarity graph construction

#### 4.3 Data Augmentation Experiments
- **Setup**: Imbalanced datasets with minority class ratio 1:10, 1:20, 1:50
- **Baselines**: No augmentation, random oversampling, SMOTE, ADASYN, borderline-SMOTE
- **Metrics**: Minority class precision/recall/F1, overall balanced accuracy
- **Analysis**: Quality of synthetic samples (do they fall in expected regions?)

#### 4.4 Anomaly Detection Experiments
- **Setup**: Inject anomalies (label flips, feature corruptions) at known rates
- **Baselines**: Isolation Forest, LOF, One-Class SVM
- **Metrics**: Precision, Recall, F1 at various thresholds; AUC-ROC
- **User Study**: Present anomaly explanations to clinicians, measure perceived usefulness

#### 4.5 Computational Efficiency
- Runtime comparison across dataset sizes
- Memory usage

### 5. Results

#### 5.1 Semi-Supervised Learning Results
- Table: Accuracy/F1/AUC across datasets and label fractions
- Figure: Performance vs. labeled fraction curves
- Key finding: TaGra-ML particularly effective at very low label fractions (1-5%)

#### 5.2 Data Augmentation Results
- Table: Minority class F1 across imbalance ratios
- Figure: t-SNE visualization of original vs. augmented data
- Key finding: TaGra augmentation generates more realistic samples than SMOTE

#### 5.3 Anomaly Detection Results
- Table: Detection metrics across anomaly types
- Figure: Example explanations
- User study results: clinicians prefer TaGra explanations

### 6. Discussion

#### 6.1 When Does TaGra-ML Work Best?
- High-dimensional data where local structure matters
- Datasets with meaningful similarity relationships
- Scenarios requiring explainability

#### 6.2 Limitations
- Computational cost for very large datasets
- Sensitivity to graph construction parameters
- Assumes similarity graph captures relevant structure

#### 6.3 Future Directions
- Longitudinal extensions
- Integration with deep learning (GNNs on TaGra graphs)
- Automated parameter selection

### 7. Conclusion

- Summary of contributions
- Key takeaway: graph representations of tabular data enable neighborhood-aware ML
- Call to action: TaGra-ML as a standard preprocessing step for medical data analysis

### References

[Standard references to related work]

---

## Part 3: Implementation Priorities

### High Priority (Core Contributions)
1. **Semi-supervised label propagation module** - Most straightforward to implement, clear evaluation strategy
2. **Explainable anomaly detection** - High value for medical applications, differentiator from existing tools
3. **Graph-aware data augmentation** - Addresses real pain point (class imbalance)

### Medium Priority (Supporting Features)
4. **Missing value imputation** - Useful but incremental over existing methods
5. **Feature importance** - Interesting but harder to evaluate rigorously

### Lower Priority (Future Work)
6. **Longitudinal analysis** - Requires temporal datasets, more complex evaluation
7. **Cytoscape integration** - Nice-to-have for visualization, not core contribution

---

## Part 4: Suggested Timeline

### Phase 1: Foundation (Month 1-2)
- Implement label propagation module
- Implement anomaly detection scoring and explanation generation
- Set up evaluation infrastructure (datasets, baselines, metrics)

### Phase 2: Core Development (Month 3-4)
- Implement data augmentation module
- Run experiments for all three core contributions
- Iterate based on results

### Phase 3: Paper Writing (Month 5-6)
- Analyze results, generate figures and tables
- Write paper draft
- Internal review and revision

### Phase 4: Submission (Month 7)
- Final revisions
- Submit to target venue (suggestions: Journal of Machine Learning Research, Nature Machine Intelligence, or domain-specific venues like Journal of Biomedical Informatics)

---

## Part 5: Key Questions to Resolve

1. **Which graph construction method works best for each task?**
   - Hypothesis: KNN for semi-supervised (guarantees connectivity), similarity for anomaly detection (allows isolation)

2. **How sensitive are results to hyperparameters (k, thresholds)?**
   - Need systematic ablation studies

3. **What's the computational ceiling?**
   - Profile performance on datasets of increasing size
   - Identify bottlenecks (graph construction? label propagation? community detection?)

4. **How do we handle new/unseen data points?**
   - Option A: Rebuild entire graph (expensive)
   - Option B: Add new node to existing graph based on distances to existing nodes (faster)

5. **What's the target venue?**
   - ML venue: emphasize methodological contributions
   - Medical informatics venue: emphasize clinical applications and explainability
   - Software venue: emphasize tool and reproducibility

---

## Conclusion

TaGra has genuine potential beyond visualization. The key is to demonstrate that the graph representation enables ML tasks that are **better** than alternatives—not just different. The three highest-value directions are semi-supervised learning, data augmentation, and explainable anomaly detection. A focused paper demonstrating improvements on these tasks, with rigorous evaluation against baselines, would establish TaGra as a valuable tool for the ML and medical informatics communities.
