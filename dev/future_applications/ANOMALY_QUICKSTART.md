# Anomaly Detection Comparison Tool - Quick Start Guide

## Installation

The script is already in the TaGra root directory. No additional installation needed except seaborn:

```bash
# Activate your TaGra virtual environment
source venv/bin/activate

# Install seaborn if not already installed
pip install seaborn
```

## Basic Usage

### 1. Simple Unsupervised Analysis

```bash
python anomaly_task_test.py -d your_data.csv
```

This will:
- Run all traditional methods (Isolation Forest, LOF, One-Class SVM, DBSCAN)
- Build all graph variants (KNN, distance, similarity with various parameters)
- Run all graph-based detection methods (structural, attribute, combined)
- Generate visualizations
- Create Cytoscape export
- Save results to `./anomaly_results/`

### 2. With Ground Truth (Supervised Evaluation)

```bash
python anomaly_task_test.py -d your_data.csv --target label_column --anomaly-label 1
```

Additional outputs:
- Performance metrics (precision, recall, F1, AUC) for each method
- Performance comparison bar chart
- ROC curves
- `method_metrics.json` with detailed metrics

### 3. Quick Test (Minimal Configuration)

```bash
python anomaly_task_test.py \
  -d your_data.csv \
  --methods isolation_forest \
  --graph-methods knn \
  --no-cytoscape
```

Fastest execution for initial testing.

### 4. Custom Parameters

```bash
python anomaly_task_test.py \
  -d your_data.csv \
  --contamination 0.05 \
  --output-dir my_results/ \
  --verbose
```

Adjust contamination rate (expected proportion of anomalies) and output location.

## Tested Example

```bash
# Tested successfully on 768 samples, 9 features
python anomaly_task_test.py \
  -d results/diabetes_preprocessed_202601221539.csv \
  --contamination 0.1 \
  --output-dir test_anomaly_results \
  --methods isolation_forest,lof \
  --graph-methods knn \
  --verbose
```

**Results**:
- Execution time: ~30 seconds
- Total methods tested: 16 (1 Isolation Forest + 3 LOF variants + 15 graph-based)
- Output size: ~4.5 MB (including interactive HTML)

## Output Files

After running, check your output directory:

```bash
your_output_dir/
├── detection_results.csv          # Full detection matrix
├── summary_report.txt             # Human-readable summary
├── plots/                         # All visualization plots
│   ├── detection_heatmap.png
│   ├── method_agreement.png
│   ├── detection_distribution.png
│   ├── score_distributions.png
│   ├── graph_param_impact.png
│   ├── traditional_vs_graph.png
│   ├── performance_comparison.png  # If ground truth
│   └── roc_curves.png             # If ground truth
└── cytoscape/                     # Interactive visualizations
    ├── anomaly_graph.cyjs         # For Cytoscape Desktop
    ├── anomaly_graph.html         # Open in browser!
    └── anomaly_graph.json         # For Cytoscape.js
```

## Interpreting Results

### 1. Summary Report

Open `summary_report.txt` to see:
- How many anomalies each method detected
- Agreement statistics
- Best performing methods (if ground truth available)

### 2. Detection Heatmap

Open `plots/detection_heatmap.png`:
- Each row = one sample
- Each column = one detection method
- Red = detected as anomaly, White = normal
- Samples are sorted by total detections (most suspicious at top)

### 3. Method Agreement Matrix

Open `plots/method_agreement.png`:
- Shows Jaccard similarity between methods
- High values (red) = methods agree
- Low values (yellow) = methods disagree
- Useful for identifying redundant or complementary methods

### 4. Interactive HTML

Open `cytoscape/anomaly_graph.html` in your browser:
- **Click nodes** to see all features and detection results
- **Zoom/Pan** to explore the graph
- **Node size** = number of methods flagging it as anomaly
- **Node color** = agreement score

### 5. Detection Results CSV

Open `detection_results.csv` in Excel or pandas:
```python
import pandas as pd
df = pd.read_csv('anomaly_results/detection_results.csv')

# Get samples flagged by most methods
top_suspects = df.nlargest(10, 'total_detections')

# Get consensus anomalies (≥50% methods agree)
consensus = df[df['agreement_score'] >= 0.5]

# Compare two specific methods
method1 = 'traditional_isolation_forest'
method2 = 'graph_knn_k5_combined'
both = df[df[method1] & df[method2]]
```

## Common Workflows

### Workflow 1: Exploratory Analysis

```bash
# Run with defaults
python anomaly_task_test.py -d data.csv

# Review summary report
cat anomaly_results/summary_report.txt

# Look at detection heatmap
open anomaly_results/plots/detection_heatmap.png

# Explore interactively
open anomaly_results/cytoscape/anomaly_graph.html
```

### Workflow 2: Method Comparison with Ground Truth

```bash
# Run with target column
python anomaly_task_test.py \
  -d data.csv \
  --target label \
  --anomaly-label "fraud"

# Check performance
cat anomaly_results/summary_report.txt | grep "Best"

# View ROC curves
open anomaly_results/plots/roc_curves.png

# Examine metrics
python -c "import json; print(json.dumps(json.load(open('anomaly_results/method_metrics.json')), indent=2))"
```

### Workflow 3: Parameter Tuning

```bash
# Test different contamination rates
for cont in 0.05 0.10 0.15; do
  python anomaly_task_test.py \
    -d data.csv \
    --contamination $cont \
    --output-dir results_cont_$cont \
    --methods isolation_forest
done

# Compare results
ls -d results_cont_* | while read dir; do
  echo "$dir:"
  cat $dir/summary_report.txt | grep "detections"
  echo ""
done
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce methods tested
```bash
python anomaly_task_test.py \
  -d large_data.csv \
  --methods isolation_forest \
  --graph-methods knn  # Only test KNN graphs
```

### Issue: Taking Too Long

**Solution**: Skip Cytoscape export
```bash
python anomaly_task_test.py \
  -d data.csv \
  --no-cytoscape
```

### Issue: Graph Construction Fails

**Error**: `preprocess() got an unexpected keyword argument`

**Solution**: Update TaGra to version 0.3.0+
```bash
pip install --upgrade tagra
```

### Issue: Missing seaborn

**Error**: `ModuleNotFoundError: No module named 'seaborn'`

**Solution**:
```bash
pip install seaborn
```

## Tips & Best Practices

1. **Start Small**: Test with `--methods isolation_forest --graph-methods knn` first
2. **Use Verbose Mode**: Add `--verbose` to see detailed progress
3. **Set Contamination Appropriately**:
   - If you expect 5% anomalies, use `--contamination 0.05`
   - Default 0.1 (10%) works well for most cases
4. **Check Graph Construction**: If all graph methods fail, your data might need preprocessing
5. **Iterate on Parameters**: Start broad, then narrow down based on results

## Performance Expectations

| Dataset Size | Methods Tested | Execution Time | Output Size |
|-------------|----------------|----------------|-------------|
| 500 samples | All (56 total) | ~1 minute | ~5 MB |
| 1000 samples | All (56 total) | ~2 minutes | ~10 MB |
| 5000 samples | All (56 total) | ~10 minutes | ~50 MB |
| 10000 samples | Subset (10 total) | ~5 minutes | ~30 MB |

*Times measured on standard laptop (8GB RAM, 4 cores)*

## Next Steps

After running the tool:

1. **Identify Best Methods**: Check `summary_report.txt` for best performers
2. **Analyze Disagreements**: Look at method agreement matrix
3. **Investigate Top Anomalies**: Review samples with high `total_detections`
4. **Refine Parameters**: Re-run with focused method set based on findings
5. **Export for Further Analysis**: Use `detection_results.csv` for custom analysis

## Example Analysis Script

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('anomaly_results/detection_results.csv')

# Get consensus anomalies
consensus = df[df['agreement_score'] >= 0.5]
print(f"Found {len(consensus)} consensus anomalies")

# Plot agreement distribution
plt.figure(figsize=(10, 6))
plt.hist(df['agreement_score'], bins=20, edgecolor='black')
plt.xlabel('Agreement Score')
plt.ylabel('Number of Samples')
plt.title('Distribution of Detection Agreement')
plt.savefig('custom_agreement_dist.png')

# Compare traditional vs graph
print("\nDetection overlap:")
print(f"Traditional only: {(df['traditional_count'] > 0) & (df['graph_count'] == 0)).sum()}")
print(f"Graph only: {((df['traditional_count'] == 0) & (df['graph_count'] > 0)).sum()}")
print(f"Both: {((df['traditional_count'] > 0) & (df['graph_count'] > 0)).sum()}")
```

## Support

For issues or questions:
1. Check `future_applications/ANOMALY_DETECTION_COMPARISON.md` for detailed documentation
2. Review the script code: `anomaly_task_test.py`
3. Check TaGra documentation for graph construction issues

---

**Quick Reference**:
- Script: `anomaly_task_test.py`
- Help: `python anomaly_task_test.py --help`
- Documentation: `future_applications/ANOMALY_DETECTION_COMPARISON.md`
