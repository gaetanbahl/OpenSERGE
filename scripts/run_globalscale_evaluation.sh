#!/bin/bash
# Full evaluation script for OpenSERGE on Global-Scale Road Dataset
# Runs inference on selected regions and computes averaged TOPO and APLS metrics
# Supports parallel processing for faster execution
#
# Usage:
#   ./scripts/run_globalscale_evaluation.sh [WEIGHTS] [N_JOBS] [EVAL_SPLIT]
#
# Arguments:
#   WEIGHTS     - Path to model checkpoint (default: checkpoints/best_globalscale.pt)
#   N_JOBS      - Number of parallel jobs (default: 8)
#   EVAL_SPLIT  - Dataset split to evaluate: 'train', 'valid', 'test', 'ood', or 'all' (default: test)
#
# Examples:
#   ./scripts/run_globalscale_evaluation.sh checkpoints/best_globalscale.pt 8 test
#   ./scripts/run_globalscale_evaluation.sh checkpoints/best_globalscale.pt 4 ood
#   ./scripts/run_globalscale_evaluation.sh checkpoints/best_globalscale.pt 8 all

set -e  # Exit on error

# Configuration
WEIGHTS="${1:-checkpoints/best_globalscale.pt}"
N_JOBS="${2:-8}"  # Number of parallel jobs
EVAL_SPLIT="${3:-test}"  # Which split to evaluate: 'train', 'valid', 'test', 'ood', or 'all'
DATA_ROOT="data/Global-Scale"
OUTPUT_ROOT="evaluation_results_globalscale"
STRIDE=384
IMG_SIZE=512
JUNCTION_THRESH=0.5
EDGE_THRESH=0.5
MERGE_THRESH=16.0
K=4

# Create output directories
mkdir -p "$OUTPUT_ROOT"/{graphs,images,metrics/topo,metrics/apls,logs}

echo "========================================"
echo "OpenSERGE GlobalScale Evaluation"
echo "========================================"
echo "Checkpoint: $WEIGHTS"
echo "Data root: $DATA_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Evaluation split: $EVAL_SPLIT"
echo "Parallel jobs: $N_JOBS"
echo "Stride: $STRIDE"
echo "========================================"

# Function to run inference on a single region
run_inference() {
    local region_file=$1
    # Create unique region name by replacing / with __ in the path
    # Example: data/Global-Scale/out_of_domain/1/region_90_sat.png
    #   -> out_of_domain/1/region_90_sat.png -> out_of_domain__1__region_90
    local region_path=$(echo "$region_file" | sed "s|^$DATA_ROOT/||" | sed 's/_sat\.png$//')
    local region_name=$(echo "$region_path" | sed 's|/|__|g')

    echo "[Inference] Processing $region_name..."

    # Run inference (automatically saves both .json and .p files)
    python -m openserge.infer \
        --weights "$WEIGHTS" \
        --image "$region_file" \
        --output "$OUTPUT_ROOT/graphs/graph_${region_name}.json" \
        --output_image "$OUTPUT_ROOT/images/visualization_${region_name}.png" \
        --img_size $IMG_SIZE \
        --stride $STRIDE \
        --junction_thresh $JUNCTION_THRESH \
        --edge_thresh $EDGE_THRESH \
        --merge_threshold $MERGE_THRESH \
        --device cuda 2>&1 | tee "$OUTPUT_ROOT/logs/infer_${region_name}.log"

    if [ $? -eq 0 ]; then
        echo "[Inference] $region_name complete."
    else
        echo "[Inference] ERROR processing $region_name" >&2
        exit 1
    fi
}

# Function to compute TOPO metrics for a single region
compute_topo() {
    local region_name=$1
    # Reconstruct the original path: just replace __ with /
    # Example: out_of_domain__1__region_90 -> out_of_domain/1/region_90
    local region_path=$(echo "$region_name" | sed 's|__|/|g')

    local gt_graph="$DATA_ROOT/${region_path}_refine_gt_graph.p"
    local pred_graph="$OUTPUT_ROOT/graphs/graph_${region_name}.p"
    local output_file="$OUTPUT_ROOT/metrics/topo/topo_${region_name}.txt"

    if [ ! -f "$gt_graph" ]; then
        echo "[TOPO] Warning: GT file not found for $region_name, skipping..."
        echo "[TOPO]   Expected: $gt_graph"
        return 0
    fi

    if [ ! -f "$pred_graph" ]; then
        echo "[TOPO] Warning: Prediction file not found for $region_name, skipping..."
        return 0
    fi

    echo "[TOPO] Computing TOPO for $region_name..."

    cd metrics/topo
    python main.py \
        -graph_gt "../../$gt_graph" \
        -graph_prop "../../$pred_graph" \
        -output "../../$output_file" 2>&1 | tee "../../$OUTPUT_ROOT/logs/topo_${region_name}.log"
    cd ../../

    echo "[TOPO] $region_name complete."
}

# Function to compute APLS metrics for a single region
compute_apls() {
    local region_name=$1
    # Reconstruct the original path: just replace __ with /
    local region_path=$(echo "$region_name" | sed 's|__|/|g')

    local gt_graph="$DATA_ROOT/${region_path}_refine_gt_graph.p"
    local pred_graph="$OUTPUT_ROOT/graphs/graph_${region_name}.p"

    if [ ! -f "$gt_graph" ]; then
        echo "[APLS] Warning: GT file not found for $region_name, skipping..."
        echo "[APLS]   Expected: $gt_graph"
        return 0
    fi

    if [ ! -f "$pred_graph" ]; then
        echo "[APLS] Warning: Prediction file not found for $region_name, skipping..."
        return 0
    fi

    echo "[APLS] Computing APLS for $region_name..."

    cd metrics/apls

    # Convert to JSON format
    python convert.py "../../$gt_graph" "../../$OUTPUT_ROOT/metrics/apls/${region_name}_gt.json" 2>&1
    python convert.py "../../$pred_graph" "../../$OUTPUT_ROOT/metrics/apls/${region_name}_pred.json" 2>&1

    # Run APLS
    go run main.go \
        "../../$OUTPUT_ROOT/metrics/apls/${region_name}_gt.json" \
        "../../$OUTPUT_ROOT/metrics/apls/${region_name}_pred.json" \
        "../../$OUTPUT_ROOT/metrics/apls/apls_${region_name}.txt" 2>&1 | tee "../../$OUTPUT_ROOT/logs/apls_${region_name}.log"

    cd ../../

    echo "[APLS] $region_name complete."
}

# Export functions for GNU parallel
export -f run_inference
export -f compute_topo
export -f compute_apls
export WEIGHTS DATA_ROOT OUTPUT_ROOT STRIDE IMG_SIZE JUNCTION_THRESH EDGE_THRESH MERGE_THRESH K

# Discover regions based on split
python3 -c "
import os
import sys

DATA_ROOT = '$DATA_ROOT'
EVAL_SPLIT = '$EVAL_SPLIT'

# Map split names to directory names
split_dirs = {
    'train': 'train',
    'valid': 'validation',
    'validation': 'validation',
    'test': 'in-domain-test',
    'in-domain-test': 'in-domain-test',
    'ood': 'out_of_domain',
    'out-of-domain': 'out_of_domain',
    'out_of_domain': 'out_of_domain'
}

# Determine which splits to process
if EVAL_SPLIT == 'all':
    splits_to_eval = ['train', 'validation', 'in-domain-test', 'out_of_domain']
elif EVAL_SPLIT in split_dirs:
    splits_to_eval = [split_dirs[EVAL_SPLIT]]
else:
    print(f'Error: Invalid split \"{EVAL_SPLIT}\". Must be one of: train, valid, test, ood, all', file=sys.stderr)
    sys.exit(1)

# Discover all satellite images
region_files = []
for split in splits_to_eval:
    split_dir = os.path.join(DATA_ROOT, split)
    if not os.path.exists(split_dir):
        print(f'Warning: Split directory not found: {split_dir}', file=sys.stderr)
        continue

    # Walk through subdirectories to find all satellite images
    for root, dirs, files in os.walk(split_dir):
        for f in files:
            if f.endswith('_sat.png'):
                region_files.append(os.path.join(root, f))

if not region_files:
    print(f'Error: No regions found for split(s): {splits_to_eval}', file=sys.stderr)
    sys.exit(1)

# Sort for consistent ordering
for region_file in sorted(region_files):
    print(region_file)
" > /tmp/globalscale_regions_to_eval.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to discover regions"
    exit 1
fi

NUM_REGIONS=$(wc -l < /tmp/globalscale_regions_to_eval.txt)
echo "Evaluating $NUM_REGIONS regions from split: $EVAL_SPLIT"

# Step 1: Run inference on selected regions in parallel
echo ""
echo "[1/4] Running inference on $NUM_REGIONS regions (parallel jobs: $N_JOBS)..."
echo ""

# Check if GNU parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for inference..."
    cat /tmp/globalscale_regions_to_eval.txt | parallel -j $N_JOBS --progress run_inference
else
    echo "GNU parallel not found, using xargs (less efficient)..."
    cat /tmp/globalscale_regions_to_eval.txt | xargs  -I {} bash -c 'run_inference "$@"' _ {}
fi

echo ""
echo "Inference complete for all regions!"
echo ""

# Step 2: Setup APLS (only once)
echo "[Setup] Initializing APLS Go dependencies..."
cd metrics/apls
if [ ! -f "go.mod" ]; then
    go mod init apls
    go get github.com/dhconnelly/rtreego
fi
cd ../../

# Step 3: Collect region names from predictions
echo ""
echo "[2/4] Computing TOPO metrics (parallel jobs: $N_JOBS)..."
echo ""

# Get list of region names that have predictions
region_names=()
for pred_file in "$OUTPUT_ROOT"/graphs/graph_*.p; do
    if [ -f "$pred_file" ]; then
        region_name=$(basename "$pred_file" | sed 's/graph_\(.*\)\.p/\1/')
        region_names+=("$region_name")
    fi
done

if [ ${#region_names[@]} -eq 0 ]; then
    echo "Warning: No predictions found to evaluate!"
    exit 1
fi

echo "Found ${#region_names[@]} predictions to evaluate"

# Compute TOPO metrics in parallel
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for TOPO metrics..."
    printf '%s\n' "${region_names[@]}" | parallel -j $N_JOBS --progress compute_topo
else
    echo "GNU parallel not found, using xargs..."
    printf '%s\n' "${region_names[@]}" | xargs -P $N_JOBS -I {} bash -c 'compute_topo "$@"' _ {}
fi

echo ""
echo "TOPO metrics complete!"
echo ""

# Step 4: Compute APLS metrics in parallel
echo ""
echo "[3/4] Computing APLS metrics (parallel jobs: $N_JOBS)..."
echo ""

if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for APLS metrics..."
    printf '%s\n' "${region_names[@]}" | parallel -j $N_JOBS --progress compute_apls
else
    echo "GNU parallel not found, using xargs..."
    printf '%s\n' "${region_names[@]}" | xargs -P $N_JOBS -I {} bash -c 'compute_apls "$@"' _ {}
fi

echo ""
echo "APLS metrics complete!"
echo ""

# Step 5: Aggregate results
echo "[4/4] Saving configuration and aggregating results..."
echo ""

# Save config to summary.json
cat > "$OUTPUT_ROOT/summary.json" <<EOF
{
  "config": {
    "weights": "$WEIGHTS",
    "dataset": "globalscale",
    "eval_split": "$EVAL_SPLIT",
    "stride": $STRIDE,
    "img_size": $IMG_SIZE,
    "junction_thresh": $JUNCTION_THRESH,
    "edge_thresh": $EDGE_THRESH,
    "merge_threshold": $MERGE_THRESH,
    "k": $K,
    "n_jobs": $N_JOBS
  }
}
EOF

# Aggregate results (extract metrics from individual files)
python3 -c "
import json
import os
import re

OUTPUT_ROOT = '$OUTPUT_ROOT'

# Aggregate TOPO metrics
topo_metrics = {'precision': [], 'recall': [], 'f1': []}
topo_dir = os.path.join(OUTPUT_ROOT, 'metrics/topo')
if os.path.exists(topo_dir):
    for topo_file in os.listdir(topo_dir):
        if topo_file.endswith('.txt'):
            with open(os.path.join(topo_dir, topo_file), 'r') as f:
                content = f.read()
                # Extract metrics using regex
                prec_match = re.search(r'Precision:\s*([\d.]+)', content)
                rec_match = re.search(r'Recall:\s*([\d.]+)', content)
                f1_match = re.search(r'F1:\s*([\d.]+)', content)
                if prec_match and rec_match and f1_match:
                    topo_metrics['precision'].append(float(prec_match.group(1)))
                    topo_metrics['recall'].append(float(rec_match.group(1)))
                    topo_metrics['f1'].append(float(f1_match.group(1)))

# Aggregate APLS metrics
apls_scores = []
apls_dir = os.path.join(OUTPUT_ROOT, 'metrics/apls')
if os.path.exists(apls_dir):
    for apls_file in os.listdir(apls_dir):
        if apls_file.startswith('apls_') and apls_file.endswith('.txt'):
            with open(os.path.join(apls_dir, apls_file), 'r') as f:
                content = f.read()
                # APLS output is usually just a single number
                score_match = re.search(r'([\d.]+)', content)
                if score_match:
                    apls_scores.append(float(score_match.group(1)))

# Load existing config
with open(os.path.join(OUTPUT_ROOT, 'summary.json'), 'r') as f:
    summary = json.load(f)

# Add aggregated metrics
summary['results'] = {}

if topo_metrics['f1']:
    summary['results']['topo'] = {
        'precision': {
            'mean': sum(topo_metrics['precision']) / len(topo_metrics['precision']),
            'count': len(topo_metrics['precision'])
        },
        'recall': {
            'mean': sum(topo_metrics['recall']) / len(topo_metrics['recall']),
            'count': len(topo_metrics['recall'])
        },
        'f1': {
            'mean': sum(topo_metrics['f1']) / len(topo_metrics['f1']),
            'count': len(topo_metrics['f1'])
        }
    }

if apls_scores:
    summary['results']['apls'] = {
        'mean': sum(apls_scores) / len(apls_scores),
        'count': len(apls_scores)
    }

# Save updated summary
with open(os.path.join(OUTPUT_ROOT, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

# Print results
print('\\n' + '='*50)
print('EVALUATION RESULTS SUMMARY')
print('='*50)
if 'topo' in summary['results']:
    print(f\"TOPO Precision: {summary['results']['topo']['precision']['mean']:.4f} (n={summary['results']['topo']['precision']['count']})\")
    print(f\"TOPO Recall:    {summary['results']['topo']['recall']['mean']:.4f} (n={summary['results']['topo']['recall']['count']})\")
    print(f\"TOPO F1:        {summary['results']['topo']['f1']['mean']:.4f} (n={summary['results']['topo']['f1']['count']})\")
if 'apls' in summary['results']:
    print(f\"APLS Score:     {summary['results']['apls']['mean']:.4f} (n={summary['results']['apls']['count']})\")
print('='*50)
"

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_ROOT/"
echo "Summary: $OUTPUT_ROOT/summary.json"
echo "========================================"
