# Road Graph Extraction Metrics

> **Note**: This code is adapted from the [Sat2Graph](https://github.com/songtaohe/Sat2Graph) repository with fixes and improvements for 2026 compatibility.

This directory contains implementations of two key metrics for evaluating road graph extraction quality:

## Metrics

### 1. APLS (Average Path Length Similarity)
- **Location**: `apls/`
- **Language**: Go + Python
- **Purpose**: Measures the similarity of path lengths between ground truth and predicted graphs
- **See**: [apls/README.md](apls/README.md) for detailed usage

### 2. TOPO (Topology Metric)
- **Location**: `topo/`
- **Language**: Python
- **Purpose**: Evaluates junction and edge topology correctness using geometric matching
- **See**: [topo/README.md](topo/README.md) for detailed usage

## Quick Start

### Installation

1. **Install APLS dependencies:**
```bash
cd apls
go mod download
pip install -r requirements.txt
cd ..
```

2. **Install TOPO dependencies:**
```bash
cd topo
pip install -r requirements.txt
cd ..
```

### Usage

Both metrics are automatically called by our evaluation pipeline:

```bash
./scripts/run_cityscale_evaluation.sh checkpoints/best_model.pt 8 test
```

For manual usage of individual metrics, see their respective README files.

## Graph Format

Both metrics use Python pickle files (`.p`) containing dictionaries where:
- **Key**: Tuple `(x, y)` representing vertex coordinates in pixels
- **Value**: List of tuples `[(x1, y1), (x2, y2), ...]` representing neighboring vertices

Example:
```python
{
    (100.5, 200.3): [(150.2, 180.1), (90.0, 220.5)],
    (150.2, 180.1): [(100.5, 200.3)],
    ...
}
```

Our inference script (`openserge/infer.py`) automatically generates graphs in this format.

## Attribution

This code is adapted from [Sat2Graph](https://github.com/songtaohe/Sat2Graph) by Songtao He et al., with the following modifications:
- Updated Go code for compatibility with Go 1.16+ (module system)
- Fixed import paths for modern dependencies
- Added comprehensive documentation
- Created requirements.txt files for dependency management
- Integrated into OpenSERGE evaluation pipeline

## Citations

If you use these metrics, please cite the original Sat2Graph paper:

```bibtex
@inproceedings{he2020sat2graph,
  title={Sat2Graph: Road Graph Extraction through Graph-Tensor Encoding},
  author={He, Songtao and Bastani, Favyen and Jagwani, Satvat and Alizadeh, Mohammad and Balakrishnan, Hari and Chawla, Sanjay and Madden, Sam and DeWitt, David},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```
