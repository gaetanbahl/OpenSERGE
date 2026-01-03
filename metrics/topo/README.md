# TOPO (Topology Metric) for Road Graphs

> **Note**: This code is adapted from the [Sat2Graph](https://github.com/songtaohe/Sat2Graph) repository.

## Overview

The TOPO metric evaluates road graph topology by measuring the correctness of junctions and edges. It computes precision, recall, and F1 scores for both junctions and edges using geometric matching.

## Requirements

- Python 3.7+
- See `requirements.txt` for dependencies

## Installation

```bash
cd metrics/topo
pip install -r requirements.txt
```

## Usage

```bash
python main.py -graph_gt example/gt.p -graph_prop example/prop.p -output topo_result.txt
```

### Command-line Arguments

- `-graph_gt`: Path to ground truth graph (pickle file)
- `-graph_prop`: Path to predicted/proposed graph (pickle file)
- `-output`: Path to output results file
- `-interval`: Propagation interval in meters (default: 5)
- `-matching_threshold`: Distance threshold for matching in meters (default: 10)

## Graph Format

Input graph files (`.p` files) should be Python pickle files containing a dictionary where:
- **Key**: Tuple `(x, y)` representing a vertex coordinate
- **Value**: List of tuples `[(x1, y1), (x2, y2), ...]` representing neighboring vertices

Example:
```python
{
    (100.5, 200.3): [(150.2, 180.1), (90.0, 220.5)],
    (150.2, 180.1): [(100.5, 200.3)],
    ...
}
```

## Parameters

| Parameter | Default | Note |
|-----------|---------|------|
| Propagation Distance | 300m (large) / 150m (small) | Configured in `main.py` lines 127-130 based on tile size |
| Propagation Interval | 5 meters | Use `-interval` flag to override |
| Matching Distance Threshold | 10 meters | Use `-matching_threshold` flag to override |
| Matching Angle Threshold | 30 degrees | Fixed in code |
| One-to-One Matching | True | Ensures unique junction matching |

## Output Format

The output file contains metrics in the following format:
```
Junction Precision: X.XXX
Junction Recall: X.XXX
Junction F1: X.XXX
Edge Precision: X.XXX
Edge Recall: X.XXX
Edge F1: X.XXX
```

## Visualization

To visualize TOPO results:

```bash
python showTOPO.py
```

This will generate SVG visualizations showing:
- Ground truth graph
- Predicted graph
- Matched and unmatched junctions/edges

## Dependencies

- `hopcroftkarp` - Bipartite graph matching for one-to-one junction assignment
- `rtree` - Spatial indexing for efficient geometric queries
- `pickle` - Graph serialization (Python standard library)

## How It Works

1. **Junction Matching**: Junctions are matched using Hopcroft-Karp algorithm with distance and angle thresholds
2. **Edge Propagation**: Edges are sampled at regular intervals (default 5m) along their length
3. **Edge Matching**: Propagated edge points are matched to ground truth using spatial queries
4. **Metrics Calculation**: Precision, recall, and F1 scores are computed for both junctions and edges

## Troubleshooting

**Import errors**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

**Pickle compatibility**: If you encounter pickle version errors, ensure graphs are saved with compatible Python versions (3.x).
