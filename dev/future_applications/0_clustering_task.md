# Clustering Task: TaGra as a Generalization of DBSCAN

## 1. Motivation

Before testing downstream ML tasks like label propagation, we want to validate a more fundamental claim: that TaGra's graph-based approach captures meaningful cluster structure in data. Clustering is a natural first test because it is unsupervised (no ground truth required) and directly relates to the graph topology TaGra produces.

The central idea is that TaGra *generalizes* DBSCAN: DBSCAN can be seen as a special case of graph-based clustering, and TaGra extends this by offering graph construction methods beyond Euclidean distance thresholding.

---

## 2. How DBSCAN Works

DBSCAN has two parameters:
- **eps** (we call it **r**): a distance threshold
- **min_samples** (we call it **n**): the minimum number of points in a neighborhood (not the minimum number of clusters)

### 2.1 Core, Border, and Noise Points

DBSCAN classifies every data point into one of three categories:

**Core points**: A point p is *core* if it has at least `min_samples` points within distance eps of it (including itself, in sklearn's convention). Core points are in dense regions and can "seed" clusters.

**Border points**: A point that is *not* core, but falls within distance eps of at least one core point. Border points are on the edges of clusters. They are assigned to the cluster of the nearest core point, but they do not propagate cluster membership further.

**Noise points**: Points that are neither core nor border. They are isolated from any dense region and are labeled as outliers (-1).

### 2.2 Cluster Formation

1. Pick an unvisited point p.
2. If p is a core point, start a new cluster. Add p and all points within eps of p.
3. For each core point added to the cluster, recursively add its eps-neighbors.
4. Border points get absorbed into the cluster but do not propagate further.
5. Repeat until all points are visited. Remaining unassigned points are noise.

The key insight is that **only core points propagate cluster membership**. A chain of core points within eps of each other forms the skeleton of a cluster. Border points hang off this skeleton. Noise points are discarded.

---

## 3. Initial (Incorrect) Equivalence Claim

### 3.1 The Claim

The initial idea was:

> Build a distance-threshold graph (connecting all points within distance r), find connected components, discard components with fewer than n points. This is equivalent to DBSCAN.

### 3.2 Why It's Wrong

Connected components do **not** account for the density requirement. DBSCAN's core point criterion acts as a density filter that connected components lack.

**Counterexample**: Consider two dense clusters connected by a thin chain of points, where each chain point has only 2 neighbors within r.

- **DBSCAN** (min_samples=5): The chain points have fewer than 5 neighbors, so they are *not* core points. They become noise. The two dense regions are identified as **separate clusters**.
- **Connected components**: The chain links the two regions into **one single component**, merging what should be two distinct clusters.

The equivalence only holds trivially when min_samples is very small (1 or 2), where almost every non-isolated point becomes a core point. For typical values of min_samples (5 or more), the two approaches diverge significantly.

---

## 4. The Fix: Node Degree Filter

### 4.1 The Observation

In a distance-threshold graph, the degree of a node equals the number of other data points within distance r. This is directly related to DBSCAN's core point criterion:

- A point is **core** in DBSCAN if it has >= min_samples points within eps (including itself, per sklearn)
- A node is **core** in the graph if its degree >= min_samples - 1 (the graph does not include self-loops, hence the -1)

Therefore, filtering out nodes with `degree < min_samples - 1` from the distance-threshold graph is equivalent to keeping only DBSCAN's core points.

### 4.2 The Corrected Equivalence

```
1. Build distance-threshold graph with radius r
2. Compute node degrees (once, on the original graph — not iteratively)
3. Remove all nodes with degree < (min_samples - 1) and their edges
4. Find connected components on the filtered subgraph
5. Discard components smaller than a size threshold
```

The connected components of step 4 correspond exactly to DBSCAN's **core-point clusters**: the dense skeletons of each cluster.

### 4.3 Why Degree Evaluation Must Not Be Iterative

When a node is removed, its neighbors lose edges and their degrees decrease. However, DBSCAN determines core status from the **original** neighborhood, not after filtering. So degrees must be computed once on the full graph, the filter applied in one pass, and then components extracted from the result.

### 4.4 Remaining Gap: Border Points

The degree-filtered connected components capture the core structure of DBSCAN clusters but discard border points entirely. In DBSCAN, border points are assigned to the nearest core point's cluster.

This gap is minor in practice:
- Border points are typically a small fraction of the data
- Their assignment is order-dependent in DBSCAN (if a border point neighbors core points from two different clusters, it is assigned to whichever is processed first)
- For the purpose of comparing clustering quality via DBCV, the core structure is what matters most

If full equivalence were needed, a second pass could assign each removed node to the component of its nearest surviving neighbor. However, for the purpose of our comparison test, this is not necessary.

---

## 5. The Generalization Argument

### 5.1 DBSCAN as a Special Case

With the degree filter, DBSCAN can be decomposed into three graph operations:

| Step | DBSCAN operation | Graph equivalent |
|------|-----------------|------------------|
| 1. Neighborhood | eps-ball in Euclidean space | Distance-threshold graph |
| 2. Density filter | Core point criterion (min_samples) | Node degree filter |
| 3. Clustering | Density-reachable components | Connected components |

### 5.2 How TaGra Generalizes Step 1

DBSCAN is locked into distance-threshold neighborhoods (though sklearn does support non-Euclidean metrics via the `metric` parameter). TaGra generalizes the graph construction step by offering fundamentally different neighborhood structures:

- **KNN graph**: Each node connects to its k nearest neighbors regardless of absolute distance. This adapts to local density — a known weakness of DBSCAN's fixed-radius approach.
- **Cosine similarity graph**: Connects nodes whose feature vectors point in similar directions, capturing angular relationships that Euclidean distance misses.
- **Distance-threshold graph**: Equivalent to DBSCAN's neighborhood definition (the special case).

Steps 2 (degree filter) and 3 (connected components) remain the same across all graph types. The generalization is purely in how the graph is constructed.

### 5.3 Why This Matters

DBSCAN's fixed-radius eps struggles with clusters of varying density. A single eps value cannot simultaneously capture tight clusters (small eps needed) and diffuse clusters (large eps needed). KNN graphs handle this naturally because each node's neighborhood adapts to local density.

---

## 6. Comparison Plan

### 6.1 Objective

Evaluate whether TaGra's graph-based clustering (with degree filtering) produces better clusters than standard DBSCAN, measured by DBCV (Density-Based Clustering Validation, range [-1, +1], higher is better).

### 6.2 Evaluation Metric: DBCV

DBCV is chosen because:
- It does not require ground truth labels
- It is designed for density-based clusters (non-convex shapes)
- It has a bounded, interpretable range [-1, +1]
- It evaluates both cluster cohesion and separation

### 6.3 Methods to Compare

#### Baseline: Standard DBSCAN
```
Preprocessed data → DBSCAN(eps, min_samples) → clusters → DBCV
```

Parameter grid:
- eps: determined via k-distance plot or a range of values
- min_samples: [3, 5, 7, 10]

#### Method A: TaGra Distance-Threshold Graph (Validation)
```
Preprocessed data → distance-threshold graph(r) → degree filter(min_samples) → connected components → DBCV
```

This should approximate DBSCAN results, validating the theoretical equivalence.

Parameter grid:
- r (distance_threshold): same values as DBSCAN eps
- degree filter: same values as DBSCAN min_samples

#### Method B: TaGra KNN Graph
```
Preprocessed data → KNN graph(k) → degree filter(n) → connected components → DBCV
```

Parameter grid:
- k: [3, 5, 7, 10, 15, 20]
- degree filter n: [3, 5, 7, 10]

#### Method C: TaGra Similarity Graph
```
Preprocessed data → cosine similarity graph(threshold) → degree filter(n) → connected components → DBCV
```

Parameter grid:
- similarity_threshold: [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
- degree filter n: [3, 5, 7, 10]

### 6.4 Experimental Controls

To isolate TaGra's contribution, all methods must use the **same preprocessed data** (same scaling, encoding, and feature selection). The only variable is the graph construction method.

### 6.5 Expected Outcomes

1. **Method A vs Baseline**: DBCV scores should be similar, confirming the theoretical equivalence between DBSCAN and degree-filtered distance-threshold graphs.
2. **Method B vs Baseline**: If KNN graphs produce higher DBCV scores, it demonstrates that adaptive neighborhoods improve clustering.
3. **Method C vs Baseline**: Tests whether angular similarity captures cluster structure differently from distance-based methods.

### 6.6 Output

For each dataset and parameter combination:
- DBCV score
- Number of clusters found
- Number of noise points
- Cluster size distribution

Summary visualizations:
- DBCV comparison across methods (bar chart or box plot)
- Cluster count vs DBCV (to detect trivial solutions)
- Parameter sensitivity plots (DBCV vs k, DBCV vs threshold)
- Validation plot: DBSCAN DBCV vs Method A DBCV (should correlate strongly)

### 6.7 Datasets

To be determined. Candidates should include:
- Datasets with clusters of varying density (where DBSCAN struggles)
- Datasets with non-convex cluster shapes
- High-dimensional datasets where Euclidean distance is less meaningful

### 6.8 Success Criteria

1. Method A reproduces DBSCAN's DBCV scores (validates theoretical equivalence)
2. At least one TaGra method (B or C) achieves higher DBCV than DBSCAN on at least some datasets
3. Results are reproducible and statistically meaningful
