#!/bin/bash
# Evaluation script for OpenSERGE on RoadTracer test dataset
# Runs inference on test cities and computes averaged TOPO and APLS metrics
# Supports parallel processing for faster execution
#
# Usage:
#   ./scripts/run_roadtracer_evaluation.sh [WEIGHTS] [N_JOBS]
#
# Arguments:
#   WEIGHTS     - Path to model checkpoint (default: checkpoints/best_model.pt)
#   N_JOBS      - Number of parallel jobs (default: 8)
#
# Examples:
#   ./scripts/run_roadtracer_evaluation.sh checkpoints/best_model.pt 8
#   ./scripts/run_roadtracer_evaluation.sh checkpoints/best_model.pt 4

set -e  # Exit on error

# Configuration
WEIGHTS="${1:-checkpoints/best_model.pt}"
N_JOBS="${2:-8}"  # Number of parallel jobs
DATA_ROOT="data/roadtracer/dataset/data"
OUTPUT_ROOT="${3:-roadtracer_evaluation_results}"
STRIDE=384
IMG_SIZE=512
JUNCTION_THRESH=0.5
EDGE_THRESH=0.5
MERGE_THRESH=16.0

# Create output directories
mkdir -p "$OUTPUT_ROOT"/{graphs,images,metrics/topo,metrics/apls,logs,converted_gt}

echo "========================================"
echo "OpenSERGE RoadTracer Evaluation"
echo "========================================"
echo "Checkpoint: $WEIGHTS"
echo "Data root: $DATA_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Parallel jobs: $N_JOBS"
echo "Stride: $STRIDE"
echo "========================================"

# List of test cities
TEST_CITIES=(
    "amsterdam"
    "boston"
    "chicago"
    "denver"
    "kansas city"
    "la"
    "montreal"
    "new york"
    "paris"
    "pittsburgh"
    "saltlakecity"
    "san diego"
    "tokyo"
    "toronto"
    "vancouver"
)

echo "Test cities: ${#TEST_CITIES[@]}"
for city in "${TEST_CITIES[@]}"; do
    echo "  - $city"
done
echo ""

# Function to convert .graph file to pickle format
convert_graph_to_pickle() {
    local city=$1
    local graph_file="$DATA_ROOT/graphs/${city}.graph"
    local pickle_file="$OUTPUT_ROOT/converted_gt/${city}_gt_graph.p"

    if [ ! -f "$graph_file" ]; then
        echo "[Convert] Warning: Graph file not found: $graph_file"
        return 1
    fi

    echo "[Convert] Converting $city.graph to pickle format..."

    python3 -c "
import pickle
import sys

def load_roadtracer_graph(graph_file):
    '''Load RoadTracer .graph file and convert to adjacency dict.

    RoadTracer .graph format: text file with one coordinate pair per line.
    Each consecutive pair of coordinates forms an edge.
    We need to convert this to the Sat2Graph format: {(y,x): [(y1,x1), ...]}
    '''
    coords = []
    with open(graph_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                x, y = float(parts[0]), float(parts[1])
                coords.append((y, x))  # Store as (y, x) for Sat2Graph format

    # Build adjacency dict from consecutive coordinate pairs
    adj_dict = {}

    for i in range(0, len(coords) - 1, 2):
        if i + 1 < len(coords):
            node1 = coords[i]
            node2 = coords[i + 1]

            # Add bidirectional edges
            if node1 not in adj_dict:
                adj_dict[node1] = []
            if node2 not in adj_dict:
                adj_dict[node2] = []

            if node2 not in adj_dict[node1]:
                adj_dict[node1].append(node2)
            if node1 not in adj_dict[node2]:
                adj_dict[node2].append(node1)

    print(f'Loaded graph: {len(adj_dict)} nodes, {sum(len(v) for v in adj_dict.values()) // 2} edges')
    return adj_dict

try:
    graph = load_roadtracer_graph('$graph_file')
    with open('$pickle_file', 'wb') as f:
        pickle.dump(graph, f)
    print(f'Saved pickle to $pickle_file')
except Exception as e:
    print(f'Error converting graph: {e}', file=sys.stderr)
    sys.exit(1)
"

    if [ $? -eq 0 ]; then
        echo "[Convert] Conversion complete for $city"
    else
        echo "[Convert] ERROR converting $city" >&2
        return 1
    fi
}

# Function to run inference on a single city
run_inference() {
    local city=$1
    local img_file="$DATA_ROOT/testsat/${city}.png"

    if [ ! -f "$img_file" ]; then
        echo "[Inference] Warning: Image not found: $img_file"
        return 1
    fi

    echo "[Inference] Processing $city..."

    # Run inference (automatically saves both .json and .p files)
    python -m openserge.infer \
        --weights "$WEIGHTS" \
        --image "$img_file" \
        --output "$OUTPUT_ROOT/graphs/graph_${city}.json" \
        --output_image "$OUTPUT_ROOT/images/visualization_${city}.png" \
        --img_size $IMG_SIZE \
        --stride $STRIDE \
        --junction_thresh $JUNCTION_THRESH \
        --edge_thresh $EDGE_THRESH \
        --merge_threshold $MERGE_THRESH \
        --device cuda 2>&1 | tee "$OUTPUT_ROOT/logs/infer_${city}.log"

    if [ $? -eq 0 ]; then
        echo "[Inference] $city complete."
    else
        echo "[Inference] ERROR processing $city" >&2
        return 1
    fi
}

# Function to compute TOPO metrics for a single city
compute_topo() {
    local city=$1
    local gt_graph="$OUTPUT_ROOT/converted_gt/${city}_gt_graph.p"
    local pred_graph="$OUTPUT_ROOT/graphs/graph_${city}.p"
    local output_file="$OUTPUT_ROOT/metrics/topo/topo_${city}.txt"

    if [ ! -f "$gt_graph" ]; then
        echo "[TOPO] Warning: GT file not found for $city, skipping..."
        return 0
    fi

    if [ ! -f "$pred_graph" ]; then
        echo "[TOPO] Warning: Prediction file not found for $city, skipping..."
        return 0
    fi

    echo "[TOPO] Computing TOPO for $city..."

    # Convert to absolute paths
    local gt_graph_abs=$(cd $(dirname "$gt_graph") && pwd)/$(basename "$gt_graph")
    local pred_graph_abs=$(cd $(dirname "$pred_graph") && pwd)/$(basename "$pred_graph")
    local output_file_abs=$(cd $(dirname "$output_file") && pwd)/$(basename "$output_file")

    cd metrics/topo
    python main.py \
        -graph_gt "$gt_graph_abs" \
        -graph_prop "$pred_graph_abs" \
        -output "$output_file_abs" 2>&1 | tee "$OUTPUT_ROOT/logs/topo_${city}.log"
    cd ../../

    echo "[TOPO] $city complete."
}

# Function to compute APLS metrics for a single city
compute_apls() {
    local city=$1
    local gt_graph="$OUTPUT_ROOT/converted_gt/${city}_gt_graph.p"
    local pred_graph="$OUTPUT_ROOT/graphs/graph_${city}.p"

    if [ ! -f "$gt_graph" ]; then
        echo "[APLS] Warning: GT file not found for $city, skipping..."
        return 0
    fi

    if [ ! -f "$pred_graph" ]; then
        echo "[APLS] Warning: Prediction file not found for $city, skipping..."
        return 0
    fi

    echo "[APLS] Computing APLS for $city..."

    # Convert to absolute paths
    local gt_graph_abs=$(cd $(dirname "$gt_graph") && pwd)/$(basename "$gt_graph")
    local pred_graph_abs=$(cd $(dirname "$pred_graph") && pwd)/$(basename "$pred_graph")
    local gt_json_abs=$(cd "$OUTPUT_ROOT/metrics/apls" && pwd)/${city}_gt.json
    local pred_json_abs=$(cd "$OUTPUT_ROOT/metrics/apls" && pwd)/${city}_pred.json
    local apls_output_abs=$(cd "$OUTPUT_ROOT/metrics/apls" && pwd)/apls_${city}.txt

    cd metrics/apls

    # Convert to JSON format
    python convert.py "$gt_graph_abs" "$gt_json_abs" 2>&1
    python convert.py "$pred_graph_abs" "$pred_json_abs" 2>&1

    # Run APLS
    go run main.go \
        "$gt_json_abs" \
        "$pred_json_abs" \
        "$apls_output_abs" 2>&1 | tee "$OUTPUT_ROOT/logs/apls_${city}.log"

    cd ../../

    echo "[APLS] $city complete."
}

# Export functions for GNU parallel
export -f run_inference
export -f compute_topo
export -f compute_apls
export -f convert_graph_to_pickle
export WEIGHTS DATA_ROOT OUTPUT_ROOT STRIDE IMG_SIZE JUNCTION_THRESH EDGE_THRESH MERGE_THRESH

# Step 1: Convert all ground truth .graph files to pickle format
echo ""
echo "[1/5] Converting ground truth .graph files to pickle format..."
echo ""

for city in "${TEST_CITIES[@]}"; do
    convert_graph_to_pickle "$city"
done

echo ""
echo "Ground truth conversion complete!"
echo ""

# Step 2: Run inference on test cities
echo ""
echo "[2/5] Running inference on ${#TEST_CITIES[@]} test cities (parallel jobs: $N_JOBS)..."
echo ""

# Check if GNU parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for inference..."
    printf '%s\n' "${TEST_CITIES[@]}" | parallel -j $N_JOBS --progress run_inference
else
    echo "GNU parallel not found, using sequential processing..."
    for city in "${TEST_CITIES[@]}"; do
        run_inference "$city"
    done
fi

echo ""
echo "Inference complete for all cities!"
echo ""

# Step 3: Setup APLS (only once)
echo "[3/5] Initializing APLS Go dependencies..."
cd metrics/apls
if [ ! -f "go.mod" ]; then
    go mod init apls
    go get github.com/dhconnelly/rtreego
fi
cd ../../

# Step 4: Compute TOPO metrics
echo ""
echo "[4/5] Computing TOPO metrics (parallel jobs: $N_JOBS)..."
echo ""

if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for TOPO metrics..."
    printf '%s\n' "${TEST_CITIES[@]}" | parallel -j $N_JOBS --progress compute_topo
else
    echo "GNU parallel not found, using sequential processing..."
    for city in "${TEST_CITIES[@]}"; do
        compute_topo "$city"
    done
fi

echo ""
echo "TOPO metrics complete!"
echo ""

# Step 5: Compute APLS metrics
echo ""
echo "[5/5] Computing APLS metrics (parallel jobs: $N_JOBS)..."
echo ""

if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for APLS metrics..."
    printf '%s\n' "${TEST_CITIES[@]}" | parallel -j $N_JOBS --progress compute_apls
else
    echo "GNU parallel not found, using sequential processing..."
    for city in "${TEST_CITIES[@]}"; do
        compute_apls "$city"
    done
fi

echo ""
echo "APLS metrics complete!"
echo ""

# Aggregate results
echo "Aggregating results..."

# Save config to summary.json
cat > "$OUTPUT_ROOT/summary.json" <<EOF
{
  "config": {
    "weights": "$WEIGHTS",
    "stride": $STRIDE,
    "img_size": $IMG_SIZE,
    "junction_thresh": $JUNCTION_THRESH,
    "edge_thresh": $EDGE_THRESH,
    "merge_threshold": $MERGE_THRESH,
    "n_jobs": $N_JOBS,
    "dataset": "roadtracer",
    "num_test_cities": ${#TEST_CITIES[@]}
  }
}
EOF

# Compute average metrics
python3 -c "
import json
import os
from pathlib import Path

output_root = Path('$OUTPUT_ROOT')

# Aggregate TOPO metrics
topo_dir = output_root / 'metrics/topo'
topo_results = []
for topo_file in topo_dir.glob('topo_*.txt'):
    city = topo_file.stem.replace('topo_', '')
    try:
        with open(topo_file) as f:
            content = f.read().strip()
            # Parse TOPO output (format: precision recall f1)
            parts = content.split()
            if len(parts) >= 3:
                precision = float(parts[0])
                recall = float(parts[1])
                f1 = float(parts[2])
                topo_results.append({
                    'city': city,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
    except Exception as e:
        print(f'Error parsing TOPO for {city}: {e}')

# Aggregate APLS metrics
apls_dir = output_root / 'metrics/apls'
apls_results = []
for apls_file in apls_dir.glob('apls_*.txt'):
    city = apls_file.stem.replace('apls_', '')
    try:
        with open(apls_file) as f:
            content = f.read().strip()
            apls = float(content)
            apls_results.append({
                'city': city,
                'apls': apls
            })
    except Exception as e:
        print(f'Error parsing APLS for {city}: {e}')

# Compute averages
summary = {
    'topo': {
        'num_cities': len(topo_results),
        'avg_precision': sum(r['precision'] for r in topo_results) / len(topo_results) if topo_results else 0.0,
        'avg_recall': sum(r['recall'] for r in topo_results) / len(topo_results) if topo_results else 0.0,
        'avg_f1': sum(r['f1'] for r in topo_results) / len(topo_results) if topo_results else 0.0,
        'per_city': topo_results
    },
    'apls': {
        'num_cities': len(apls_results),
        'avg_apls': sum(r['apls'] for r in apls_results) / len(apls_results) if apls_results else 0.0,
        'per_city': apls_results
    }
}

# Load existing config
with open(output_root / 'summary.json') as f:
    config = json.load(f)

config['metrics'] = summary

# Save updated summary
with open(output_root / 'summary.json', 'w') as f:
    json.dump(config, f, indent=2)

print('=== EVALUATION SUMMARY ===')
print(f'TOPO Metrics ({summary[\"topo\"][\"num_cities\"]} cities):')
print(f'  Precision: {summary[\"topo\"][\"avg_precision\"]:.4f}')
print(f'  Recall:    {summary[\"topo\"][\"avg_recall\"]:.4f}')
print(f'  F1:        {summary[\"topo\"][\"avg_f1\"]:.4f}')
print()
print(f'APLS Metric ({summary[\"apls\"][\"num_cities\"]} cities):')
print(f'  APLS:      {summary[\"apls\"][\"avg_apls\"]:.4f}')
print()
print(f'Results saved to: $OUTPUT_ROOT/summary.json')
"

echo "========================================"
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_ROOT/"
echo "========================================"
