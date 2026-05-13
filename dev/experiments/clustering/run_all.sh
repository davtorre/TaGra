#!/usr/bin/env bash
# Run the exhaustive clustering comparison on all five EHR datasets
# using a single universal config.
#
# Usage (from repo root):
#   bash dev/experiments/clustering/run_all.sh
#
# Or from dev/experiments/clustering/:
#   bash run_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

source "$REPO_ROOT/venv/bin/activate"
cd "$SCRIPT_DIR"

CONFIG="config/config_universal.json"
DATASETS=(hcv ckd cleveland)

echo "Config : $CONFIG"
echo "Datasets: ${DATASETS[*]}"
echo "Started : $(date)"
echo "=============================="

for ds in "${DATASETS[@]}"; do
    echo
    echo ">>> $ds  ($(date +%H:%M:%S))"
    python3 run.py \
        --bundle "preprocessing/bundles/${ds}_bundle.npz" \
        --config "$CONFIG" \
        --top
    echo "<<< $ds done"
done

echo
echo "=============================="
echo "All done: $(date)"
