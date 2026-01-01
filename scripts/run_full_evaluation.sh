#!/bin/bash
# Full evaluation script for OpenSERGE on Sat2Graph dataset
# Runs inference on all regions and computes averaged TOPO and APLS metrics
# Supports parallel processing for faster execution

set -e  # Exit on error

# Configuration
WEIGHTS="${1:-checkpoints/best_model.pt}"
N_JOBS="${2:-8}"  # Number of parallel jobs
DATA_ROOT="data/Sat2Graph/data/20cities"
SPLIT_FILE="data/Sat2Graph/data/data_split.json"
OUTPUT_ROOT="evaluation_results"
STRIDE=384
IMG_SIZE=512
JUNCTION_THRESH=0.4
EDGE_THRESH=0.5
MERGE_THRESH=16.0
K=4

# Create output directories
mkdir -p "$OUTPUT_ROOT"/{graphs,images,metrics/topo,metrics/apls,logs}

echo "========================================"
echo "OpenSERGE Full Dataset Evaluation"
echo "========================================"
echo "Checkpoint: $WEIGHTS"
echo "Data root: $DATA_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Parallel jobs: $N_JOBS"
echo "Stride: $STRIDE"
echo "========================================"

# Function to run inference on a single region
run_inference() {
    local region_file=$1
    local region=$(basename "$region_file" | sed 's/region_\([0-9]*\)_sat.png/\1/')

    echo "[Inference] Processing region $region..."

    # Run inference (automatically saves both .json and .p files)
    python -m openserge.infer \
        --weights "$WEIGHTS" \
        --image "$region_file" \
        --output "$OUTPUT_ROOT/graphs/graph_${region}.json" \
        --output_image "$OUTPUT_ROOT/images/visualization_${region}.png" \
        --img_size $IMG_SIZE \
        --stride $STRIDE \
        --junction_thresh $JUNCTION_THRESH \
        --edge_thresh $EDGE_THRESH \
        --merge_threshold $MERGE_THRESH \
        --device cuda 2>&1 | tee "$OUTPUT_ROOT/logs/infer_${region}.log"

    if [ $? -eq 0 ]; then
        echo "[Inference] Region $region complete."
    else
        echo "[Inference] ERROR processing region $region" >&2
        exit 1
    fi
}

# Function to compute TOPO metrics for a single region
compute_topo() {
    local region=$1
    local gt_graph="$DATA_ROOT/region_${region}_refine_gt_graph.p"
    local pred_graph="$OUTPUT_ROOT/graphs/graph_${region}.p"
    local output_file="$OUTPUT_ROOT/metrics/topo/topo_${region}.txt"

    if [ ! -f "$gt_graph" ]; then
        echo "[TOPO] Warning: GT file not found for region $region, skipping..."
        return 0
    fi

    if [ ! -f "$pred_graph" ]; then
        echo "[TOPO] Warning: Prediction file not found for region $region, skipping..."
        return 0
    fi

    echo "[TOPO] Computing TOPO for region $region..."

    cd metrics/Sat2Graph/metrics/topo
    python main.py \
        -graph_gt "../../../../$gt_graph" \
        -graph_prop "../../../../$pred_graph" \
        -output "../../../../$output_file" 2>&1 | tee "../../../../$OUTPUT_ROOT/logs/topo_${region}.log"
    cd ../../../../

    echo "[TOPO] Region $region complete."
}

# Function to compute APLS metrics for a single region
compute_apls() {
    local region=$1
    local gt_graph="$DATA_ROOT/region_${region}_refine_gt_graph.p"
    local pred_graph="$OUTPUT_ROOT/graphs/graph_${region}.p"

    if [ ! -f "$gt_graph" ]; then
        echo "[APLS] Warning: GT file not found for region $region, skipping..."
        return 0
    fi

    if [ ! -f "$pred_graph" ]; then
        echo "[APLS] Warning: Prediction file not found for region $region, skipping..."
        return 0
    fi

    echo "[APLS] Computing APLS for region $region..."

    cd metrics/Sat2Graph/metrics/apls

    # Convert to JSON format
    python convert.py "../../../../$gt_graph" "../../../../$OUTPUT_ROOT/metrics/apls/region_${region}_gt.json" 2>&1
    python convert.py "../../../../$pred_graph" "../../../../$OUTPUT_ROOT/metrics/apls/region_${region}_pred.json" 2>&1

    # Run APLS
    go run main.go \
        "../../../../$OUTPUT_ROOT/metrics/apls/region_${region}_gt.json" \
        "../../../../$OUTPUT_ROOT/metrics/apls/region_${region}_pred.json" \
        "../../../../$OUTPUT_ROOT/metrics/apls/apls_${region}.txt" 2>&1 | tee "../../../../$OUTPUT_ROOT/logs/apls_${region}.log"

    cd ../../../../

    echo "[APLS] Region $region complete."
}

# Export functions for GNU parallel
export -f run_inference
export -f compute_topo
export -f compute_apls
export WEIGHTS DATA_ROOT OUTPUT_ROOT STRIDE IMG_SIZE JUNCTION_THRESH EDGE_THRESH MERGE_THRESH K

# Step 1: Run inference on all regions in parallel
echo ""
echo "[1/4] Running inference on all regions (parallel jobs: $N_JOBS)..."
echo ""

# Check if GNU parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for inference..."
    find "$DATA_ROOT" -name "region_*_sat.png" | parallel -j $N_JOBS --progress run_inference
else
    echo "GNU parallel not found, using xargs (less efficient)..."
    find "$DATA_ROOT" -name "region_*_sat.png" | xargs -P $N_JOBS -I {} bash -c 'run_inference "$@"' _ {}
fi

echo ""
echo "Inference complete for all regions!"
echo ""

# Step 2: Setup APLS (only once)
echo "[Setup] Initializing APLS Go dependencies..."
cd metrics/Sat2Graph/metrics/apls
if [ ! -f "go.mod" ]; then
    go mod init apls
    go get github.com/dhconnelly/rtreego
fi
cd ../../../../

# Step 3: Compute TOPO metrics in parallel
echo ""
echo "[2/4] Computing TOPO metrics (parallel jobs: $N_JOBS)..."
echo ""

# Get list of region IDs that have predictions
region_ids=()
for pred_file in "$OUTPUT_ROOT"/graphs/graph_*.p; do
    if [ -f "$pred_file" ]; then
        region=$(basename "$pred_file" | sed 's/graph_\([0-9]*\)\.p/\1/')
        region_ids+=("$region")
    fi
done

if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for TOPO metrics..."
    printf '%s\n' "${region_ids[@]}" | parallel -j $N_JOBS --progress compute_topo
else
    echo "GNU parallel not found, using xargs..."
    printf '%s\n' "${region_ids[@]}" | xargs -P $N_JOBS -I {} bash -c 'compute_topo "$@"' _ {}
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
    printf '%s\n' "${region_ids[@]}" | parallel -j $N_JOBS --progress compute_apls
else
    echo "GNU parallel not found, using xargs..."
    printf '%s\n' "${region_ids[@]}" | xargs -P $N_JOBS -I {} bash -c 'compute_apls "$@"' _ {}
fi

echo ""
echo "APLS metrics complete!"
echo ""

# Step 5: Save config metadata
echo "[4/4] Saving configuration and aggregating results..."
echo ""

# Save config to summary.json (will be updated by aggregate_results.py)
cat > "$OUTPUT_ROOT/summary.json" <<EOF
{
  "config": {
    "weights": "$WEIGHTS",
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

# Aggregate results by split
python scripts/aggregate_results.py "$OUTPUT_ROOT" "$SPLIT_FILE"

echo "========================================"
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_ROOT/"
echo "========================================"
