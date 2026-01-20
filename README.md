# OpenSERGE

**Open-Source Simple and Efficient Road Graph Extraction**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/pytorch-2.4+-ee4c2c.svg)](https://pytorch.org/)

---

## 📖 Overview

**OpenSERGE** is an open-source re-implementation and incremental improvement of the **SERGE** (Single-shot End-to-end Road Graph Extraction) method presented at **CVPR EARTHVISION 2022**.

Unlike traditional approaches that rely on pixel-level segmentation followed by vectorization or slow iterative graph construction, OpenSERGE directly extracts vector road graphs from satellite/aerial imagery in a **single forward pass**. The method combines:

- A **Fully Convolutional Network** (CNN) that detects road junctions and predicts precise 2D offsets
- A lightweight **Graph Neural Network** (GNN) that predicts connectivity between detected junctions

This paradigm achieves **up to 500× faster inference** (from **hours** to **SECONDS**) than iterative methods like RNGDet++ and up to 8× faster than state-of-the-art segmentation-based methods like SAM-Road while maintaining competitive accuracy, making it ideal for real-time applications such as disaster response and embedded systems.

![OpenSERGE Architecture](images/SERGE_arch.png)

### Key Features

- ✅ **Single-shot extraction**: Direct graph output without iterative refinement
- ✅ **Flexible backbones**: Support for any CNN or Vision Transformer from [timm](https://github.com/huggingface/pytorch-image-models)
- ✅ **Multi-scale features**: Optional Feature Pyramid Network (FPN) for improved accuracy
- ✅ **Multi-dataset support**: CityScale, GlobalScale, RoadTracer, SpaceNet
- ✅ **Three-stage training**: Progressive curriculum for better convergence
- ✅ **GSD normalization**: Automatic resampling for consistent spatial resolution
- ✅ **Production-ready**: Docker support, comprehensive evaluation scripts, pre-trained weights

### News

2026-01-19 - First release of OpenSERGE code with CityScale checkpoints.

### Next steps

See ![TODO.md](TODO.md)

---

## 🚀 Quick Start

### Installation

#### Option 1: Conda Environment (Recommended)

```bash
# Create environment
conda create -n openserge python=3.10 -y
conda activate openserge

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install OpenSERGE in development mode
pip install -e .
```

#### Option 2: Docker (Easiest)

```bash
# Build the image
docker-compose build

# Start container with Jupyter Lab
docker-compose up -d

# Access at http://localhost:8888
```

See [DOCKER.md](DOCKER.md) for comprehensive Docker documentation.

### Quick Training Example

```bash
# Train on CityScale dataset with default config
python -m openserge.train --config configs/cityscale.json

# Train with custom settings
python -m openserge.train \
  --config configs/roadtracer.json \
  --batch_size 8 \
  --lr 0.0005 \
  --device cuda
```

### Quick Inference Example

```bash
# Extract road graph from large image (e.g., 8192×8192)
python -m openserge.infer \
  --weights checkpoints/cityscale_best.pt \
  --image /path/to/large_image.png \
  --output output_graph.json \
  --output_image visualization.png \
  --img_size 512 \
  --stride 448 \
  --merge_threshold 16.0
```

---

## 📊 Results and Pre-trained Models

### CityScale dataset

| Backbone                         | TOPO Precision | TOPO Recall | TOPO F1 | APLS   | Checkpoint |
|----------------------------------|----------------|-------------|---------|--------|------------|
| resnet50d.a1_in1k                | 55.30          | 71.32       | 62.02   | 59.17  | [Download](https://huggingface.co/gaetanbahl/openserge_cityscale_resnet50d.a1_in1k) |
| resnet101.a1_in1k.               | 62.39          | 68.25       | 64.80   | 58.25  | [Download](https://huggingface.co/gaetanbahl/openserge_cityscale_resnet101.a1_in1k) |
| vit_large_patch16_dinov3.sat493m | 63.78          | 71.68       | 67.24   | 62.30  | [Download](https://huggingface.co/gaetanbahl/openserge_cityscale_vit_large_patch16_dinov3.sat493m) |

Note: metrics were evaluated at 0.3 edge treshold and 0.3 junction threshold.


| Dataset | Backbone | TOPO Precision | TOPO Recall | TOPO F1 | APLS | Checkpoint |
|---------|----------|----------------|-------------|---------|------|------------|
| GlobalScale | ResNet50d | TBD | TBD | TBD | TBD | TBD |
| RoadTracer | ResNet50d | TBD | TBD | TBD | TBD | TBD |
| SpaceNet | ResNet50d | TBD | TBD | TBD | TBD | TBD |

---

## 🏗️ Architecture

OpenSERGE consists of two main components:

### 1. CNN Detection Head

The backbone (ResNet, EfficientNet, ViT, etc.) extracts features from the input image. These features are optionally aggregated through a Feature Pyramid Network (FPN) and fed to three parallel branches:

- **Junction branch**: Predicts junction probability at each grid cell (stride-32)
- **Offset branch**: Regresses 2D offsets (±0.5 cells) for sub-pixel precision
- **Node feature branch**: Extracts 256-D features per detected junction for the GNN

### 2. Graph Neural Network

Detected junctions are connected using a prior graph (complete or k-NN in feature space). The GNN then:

1. Aggregates node features using EdgeConv message passing (3 layers)
2. Scores each candidate edge with an MLP classifier
3. Outputs edge probabilities, thresholded to produce the final graph

### Key Innovations

- **Three-stage training**: Progressive curriculum (junction → GNN → full model)
- **Ground truth junction supervision**: Stage 2 uses GT junctions to stabilize GNN training
- **Optimized edge loss**: Pre-aligned labels eliminate O(E²) complexity
- **GSD normalization**: Resample images to consistent 1.0 m/pixel resolution
- **Positional encoding**: Concatenate normalized (x,y) coordinates to node features
- **Focal loss**: Handle class imbalance in junction detection (α=0.25, γ=2.0)

See [METHODOLOGY.md](METHODOLOGY.md) for detailed technical documentation.

---

## 📁 Repository Structure

```
openserge/
├── models/
│   ├── net.py              # CNN backbone + detection heads
│   ├── gnn.py              # EdgeConv-based GNN
│   ├── wrapper.py          # Full OpenSERGE pipeline
│   └── losses.py           # Multi-task loss functions
├── data/
│   └── dataset.py          # Dataset loaders (CityScale, RoadTracer, SpaceNet, GlobalScale)
├── utils/
│   ├── graph.py            # Graph utilities (k-NN, edge alignment)
│   ├── training.py         # Checkpointing, early stopping
│   ├── args.py             # Argument parser
│   └── utils.py            # Coordinate transforms, visualization
├── train.py                # Training entry point
└── infer.py                # Inference with sliding window

configs/                    # JSON configuration files
scripts/                    # Evaluation scripts (TOPO, APLS metrics)
notebooks/                  # Jupyter notebooks for exploration
metrics/                    # TOPO (Python) and APLS (Go) implementations
```

---

## 🎯 Training

### Configuration System

OpenSERGE uses JSON config files for reproducible experiments. All parameters can be set via config file or command-line arguments (CLI overrides config).

**Example config structure** ([configs/cityscale.json](configs/cityscale.json)):

```json
{
  "experiment_name": "cityscale",
  "dataset": "cityscale",
  "data_root": "data/Sat2Graph/data/",
  "img_size": 512,
  "source_gsd": 1.0,
  "target_gsd": 1.0,
  "preload": false,

  "backbone": "resnet50d.a1_in1k",
  "k": null,
  "use_fpn": true,
  "use_pos_encoding": true,

  "stage1_epochs": 1000,
  "stage1_patience": 100,
  "stage2_epochs": 1000,
  "stage2_patience": 100,
  "stage3_epochs": 2000,
  "stage3_patience": 200,
  "stage3_lr_factor": 0.3,

  "batch_size": 4,
  "lr": 0.001,
  "junction_thresh": 0.5,

  "loss_weight_junction": 1.0,
  "loss_weight_offset": 10.0,
  "loss_weight_edge": 1.0,

  "device": "cuda",
  "num_workers": 12
}
```

### Configuration Parameters

#### Data Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | str | `cityscale` | Dataset name: `cityscale`, `globalscale`, `roadtracer`, `spacenet` |
| `data_root` | str | **required** | Path to dataset root directory |
| `img_size` | int | 512 | Input image crop size (pixels) |
| `source_gsd` | float | None | Source ground sampling distance (m/pixel). CityScale/GlobalScale: 1.0, RoadTracer: 0.6, SpaceNet: 0.3 |
| `target_gsd` | float | None | Target GSD for resampling (m/pixel). Set to 1.0 for normalization across datasets |
| `preload` | bool | false | Preload entire dataset into RAM for faster training (requires significant memory) |

#### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backbone` | str | `resnet50` | Backbone architecture from timm library. Examples: `resnet50d.a1_in1k`, `efficientnet_b0`, `vit_base_patch16_224` |
| `use_fpn` | bool | true | Enable Feature Pyramid Network for multi-scale aggregation (CNN only) |
| `use_pos_encoding` | bool | true | Concatenate normalized (x,y) coordinates to node features before GNN |
| `k` | int/null | null | k for k-NN graph prior. `null` = complete graph, `4` = sparse 4-NN connectivity |
| `pretrained_cnn` | str | null | Path to pre-trained CNN checkpoint for GNN-only training |
| `freeze_pretrained_cnn` | bool | false | Freeze CNN weights (only train GNN) |

#### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 4 | Training batch size |
| `lr` | float | 0.001 | Learning rate (Adam optimizer) |
| `weight_decay` | float | 0.0001 | Weight decay for regularization |
| `junction_thresh` | float | 0.5 | Junction confidence threshold for inference/evaluation |

#### Three-Stage Training

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stage1_epochs` | int | 1000 | Max epochs for Stage 1 (junction detection pre-training) |
| `stage1_patience` | int | 100 | Early stopping patience for Stage 1 |
| `stage2_epochs` | int | 1000 | Max epochs for Stage 2 (GNN training with frozen CNN) |
| `stage2_patience` | int | 100 | Early stopping patience for Stage 2 |
| `stage3_epochs` | int | 2000 | Max epochs for Stage 3 (full model fine-tuning) |
| `stage3_patience` | int | 200 | Early stopping patience for Stage 3 |
| `stage3_lr_factor` | float | 0.3 | Learning rate reduction factor for Stage 3 (0.3 = 70% reduction) |
| `freeze_backbone_stage1` | bool | true | Freeze backbone in Stage 1 (only train detection heads) |
| `freeze_backbone_stage2` | bool | true | Freeze backbone in Stage 2 (only train GNN) |
| `freeze_backbone_stage3` | bool | false | Freeze backbone in Stage 3 (train everything) |
| `use_gt_junctions_stage1` | bool | false | Use ground truth junctions in Stage 1 (typically false) |
| `use_gt_junctions_stage2` | bool | true | Use ground truth junctions in Stage 2 (recommended for stability) |
| `use_gt_junctions_stage3` | bool | false | Use ground truth junctions in Stage 3 (must be false for realistic fine-tuning) |

**Important**: The three-stage training is **always enabled** when using config files. The stages automatically detect early stopping and restore the best checkpoint from each stage before proceeding.

#### Loss Weights

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_weight_junction` | float | 1.0 | Weight for junction detection loss (focal loss) |
| `loss_weight_offset` | float | 10.0 | Weight for offset regression loss (masked MSE) |
| `loss_weight_edge` | float | 1.0 | Weight for edge prediction loss (binary cross-entropy) |

Set any weight to 0 to disable that loss component (e.g., `loss_weight_edge: 0` for junction-only training).

#### Logging and Checkpointing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment_name` | str | auto | Experiment name for logs and checkpoints |
| `log_dir` | str | `logs` | Directory for TensorBoard logs |
| `checkpoint_dir` | str | `checkpoints` | Directory for model checkpoints |
| `save_freq` | int | 200 | Save checkpoint every N epochs |
| `min_delta` | float | 0.0001 | Minimum improvement for early stopping |
| `resume` | str | null | Path to checkpoint to resume training from |

#### Weights & Biases Integration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disable_wandb` | bool | false | Disable W&B logging |
| `wandb_project` | str | `openserge` | W&B project name |
| `wandb_entity` | str | null | W&B team/entity name |
| `wandb_run_name` | str | auto | W&B run name (auto-generated if null) |
| `wandb_tags` | list | [] | Tags for W&B run organization |

See [WANDB.md](WANDB.md) for detailed W&B integration guide.

#### System Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | str | `cuda` | Device: `cuda`, `mps` (Apple Silicon), or `cpu` |
| `num_workers` | int | 4 | Number of DataLoader workers |
| `seed` | int | 42 | Random seed for reproducibility |

### Training Examples

#### Train on CityScale with Default Config

```bash
python -m openserge.train --config configs/cityscale.json
```

This will run three-stage training:
1. **Stage 1**: Junction detection pre-training (~1000 epochs, early stopping at 100 patience)
2. **Stage 2**: GNN training with GT junctions (~1000 epochs, early stopping at 100 patience)
3. **Stage 3**: Full model fine-tuning with reduced LR (~2000 epochs, early stopping at 200 patience)

Checkpoints are saved to `logs/<experiment_name>/checkpoints/` with stage-specific best models:
- `stage1_best.pt` - Best junction detection checkpoint
- `stage2_best.pt` - Best GNN checkpoint
- `stage3_best.pt` - Best full model checkpoint (use this for inference!)

#### Train on RoadTracer with Custom Batch Size

```bash
python -m openserge.train \
  --config configs/roadtracer.json \
  --batch_size 8 \
  --num_workers 16
```

#### Train with EfficientNet Backbone

```bash
python -m openserge.train \
  --config configs/cityscale.json \
  --backbone efficientnet_b0
```

#### Train with Vision Transformer

```bash
python -m openserge.train \
  --config configs/cityscale.json \
  --backbone vit_base_patch16_224.augreg_in21k \
  --use_fpn false  # ViTs don't use FPN
```

#### Resume Training from Checkpoint

```bash
python -m openserge.train \
  --config configs/cityscale.json \
  --resume logs/cityscale/checkpoints/checkpoint_epoch_500.pt
```

#### Train Junction Detection Only (Single-Stage)

Edit config to disable edge loss:

```json
{
  "loss_weight_edge": 0,
  "stage1_epochs": 1000,
  "stage2_epochs": 0,
  "stage3_epochs": 0
}
```

Or use command line:

```bash
python -m openserge.train \
  --config configs/cityscale.json \
  --loss_weight_edge 0
```

#### Train GNN Only (Frozen CNN)

First train junction detection, then:

```bash
python -m openserge.train \
  --config configs/cityscale.json \
  --pretrained_cnn logs/cityscale/checkpoints/stage1_best.pt \
  --freeze_pretrained_cnn \
  --loss_weight_junction 0 \
  --loss_weight_offset 0
```

---

## 🔍 Inference

### Basic Inference

Extract road graph from a single image:

```bash
python -m openserge.infer \
  --weights checkpoints/cityscale_stage3_best.pt \
  --image /path/to/satellite_image.png \
  --output output_graph.json
```

**Output format** (`output_graph.json`):
```json
{
  "nodes": [[x1, y1], [x2, y2], ...],
  "edges": [[src_idx, dst_idx], ...]
}
```

A pickle file (`output_graph.p`) is also saved with adjacency dictionary format:
```python
{
  (x1, y1): [(x2, y2), (x3, y3), ...],
  (x2, y2): [(x1, y1), ...],
  ...
}
```

### Sliding Window Inference for Large Images

For images larger than the training tile size (default 512×512), use sliding window with node deduplication:

```bash
python -m openserge.infer \
  --weights checkpoints/cityscale_stage3_best.pt \
  --image /path/to/large_image.png \
  --output output_graph.json \
  --output_image visualization.png \
  --img_size 512 \
  --stride 448 \
  --merge_threshold 16.0
```

**Parameters**:
- `--img_size`: Tile size (should match training size, default 512)
- `--stride`: Overlap between tiles (smaller = more overlap = better boundary handling, default 448)
- `--merge_threshold`: Distance threshold (pixels) for merging duplicate nodes at tile boundaries (default 16.0)
- `--output_image`: Generate visualization with extracted graph overlay

### Advanced Inference Options

```bash
python -m openserge.infer \
  --weights checkpoints/model.pt \
  --image input.png \
  --output graph.json \
  --junction_thresh 0.6 \        # Higher = fewer, more confident junctions
  --edge_thresh 0.5 \              # Higher = fewer, more confident edges
  --k 4 \                          # Use k-NN graph prior (null = complete)
  --max_nodes 2000 \               # Maximum nodes per tile
  --merge_threshold 20.0 \         # Larger = more aggressive merging
  --device cuda \                  # Use GPU acceleration
  --verbose                        # Print detailed progress
```

### GSD Resampling During Inference

If you trained with GSD normalization (e.g., RoadTracer 0.6m → 1.0m):

```bash
python -m openserge.infer \
  --weights checkpoints/roadtracer_best.pt \
  --image test.png \
  --output graph.json \
  --source_gsd 0.6 \
  --target_gsd 1.0
```

The inference script will:
1. Resample the input image to target GSD
2. Extract the graph
3. Scale coordinates back to the original resolution

**Note**: If `source_gsd` and `target_gsd` are not specified, they are automatically read from the model's config stored in the checkpoint.

### Visualization Options

Customize visualization appearance:

```bash
python -m openserge.infer \
  --weights checkpoints/model.pt \
  --image input.png \
  --output graph.json \
  --output_image viz.png \
  --node_color red \
  --edge_color yellow \
  --node_size 6 \
  --edge_width 3
```

---

## 📈 Evaluation

OpenSERGE includes comprehensive evaluation scripts for all supported datasets with TOPO and APLS metrics.

### CityScale / Sat2Graph Evaluation

Evaluate on the CityScale test set:

```bash
bash scripts/run_cityscale_evaluation.sh \
  checkpoints/cityscale_stage3_best.pt \
  8 \      # Number of parallel processes
  test     # Split: train, valid, test, or all
```

Output:
- Per-region predictions: `evaluation_results/region_*/pred_graph.p`
- Per-region metrics: `evaluation_results/region_*/metrics.json`
- Aggregated results: `evaluation_results/aggregated_results.json`

**Metrics computed**:
- TOPO: Topology-preserving metric (Precision, Recall, F1-score)
- APLS: Average Path Length Similarity

### RoadTracer Evaluation

Evaluate on RoadTracer's 15 test cities:

```bash
bash scripts/run_roadtracer_evaluation.sh \
  checkpoints/roadtracer_stage3_best.pt \
  8        # Number of parallel processes
```

Test cities: Amsterdam, Boston, Chicago, Denver, Kansas City, LA, Montreal, New York, Paris, Pittsburgh, Salt Lake City, San Diego, Tokyo, Toronto, Vancouver.

Output: `roadtracer_evaluation_results/aggregated_results.json`

### GlobalScale Evaluation

Evaluate on GlobalScale test split:

```bash
bash scripts/run_globalscale_evaluation.sh \
  checkpoints/globalscale_stage3_best.pt \
  8 \              # Number of parallel processes
  in-domain-test   # Split: in-domain-test, out-of-domain
```

Output: `globalscale_evaluation_results/aggregated_results.json`

### View Aggregated Results

All evaluation scripts save aggregated metrics in JSON format. Example:

```bash
cat evaluation_results/aggregated_results.json
```

```json
{
  "summary": {
    "num_regions": 180,
    "avg_topo_precision": 0.8234,
    "avg_topo_recall": 0.7891,
    "avg_topo_f1": 0.8058,
    "avg_apls": 0.7612
  },
  "per_region": {
    "region_0": {
      "topo_precision": 0.8456,
      "topo_recall": 0.8123,
      ...
    }
  }
}
```

---

## 🛠️ Extending OpenSERGE

### Adding a New Backbone

OpenSERGE supports any model from the [timm library](https://github.com/huggingface/pytorch-image-models). Simply specify the model name:

```bash
python -m openserge.train \
  --config configs/cityscale.json \
  --backbone convnext_tiny.in12k_ft_in1k
```

For custom backbones not in timm, edit [openserge/models/net.py](openserge/models/net.py):

```python
class Backbone(nn.Module):
    def __init__(self, name='resnet50', pretrained=True):
        super().__init__()
        if name == 'my_custom_backbone':
            # Load your custom backbone
            self.model = MyCustomBackbone()
            self.out_channels = [256, 512, 1024, 2048]  # Output channels at each level
            self.strides = [4, 8, 16, 32]
        else:
            # Existing timm code
            ...
```

### Adding a New Dataset

Create a new dataset class in [openserge/data/dataset.py](openserge/data/dataset.py):

```python
class MyDataset(RoadGraphDataset):
    def __init__(self, data_root, split='train', img_size=512, **kwargs):
        super().__init__(data_root, img_size, **kwargs)
        # Load your dataset
        self.image_paths = ...
        self.graphs = ...

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = self._load_image(idx)

        # Load graph as adjacency dict: {(y,x): [(y1,x1), ...]}
        graph = self.graphs[idx]

        # Rasterize graph to ground truth tensors
        junction_map, offset_map, offset_mask, edges = self._rasterize_graph(
            graph, img.shape[:2], crop_y=0, crop_x=0
        )

        return {
            'image': torch.from_numpy(img).permute(2,0,1).float() / 255.0,
            'junction_map': junction_map,
            'offset_map': offset_map,
            'offset_mask': offset_mask,
            'edges': edges,
            'meta': {'region_id': idx}
        }
```

Register it in `get_dataset()`:

```python
def get_dataset(config, split='train'):
    dataset_name = config.get('dataset', 'cityscale').lower()
    if dataset_name == 'mydataset':
        return MyDataset(config['data_root'], split=split, ...)
    ...
```

### Customizing the Loss Function

Edit [openserge/models/losses.py](openserge/models/losses.py) to modify loss components:

```python
def compute_losses(output, batch, loss_weights):
    """
    Add your custom loss terms here.
    """
    losses = {}

    # Existing losses
    if loss_weights['junction'] > 0:
        losses['junction'] = focal_loss(...)

    # Add custom loss
    if 'my_custom_loss_weight' in loss_weights:
        losses['my_custom'] = my_custom_loss_fn(output, batch)

    # Total loss
    total_loss = sum(w * losses[k] for k, w in loss_weights.items() if k in losses)
    losses['total'] = total_loss

    return losses
```

---

## 🐳 Docker Deployment

OpenSERGE includes production-ready Docker support for training, inference, and evaluation.

### Quick Start with Docker

```bash
# Build image
docker-compose build

# Start container with Jupyter Lab
docker-compose up -d

# Access Jupyter at http://localhost:8888
```

### Run Training in Docker

```bash
docker-compose exec openserge python -m openserge.train \
  --config configs/cityscale.json
```

### Run Inference in Docker

```bash
docker-compose exec openserge python -m openserge.infer \
  --weights checkpoints/best_model.pt \
  --image data/test.png \
  --output results/graph.json
```

### Run Evaluation in Docker

```bash
# CityScale evaluation
docker-compose exec openserge bash scripts/run_cityscale_evaluation.sh \
  checkpoints/best_model.pt 8 test

# RoadTracer evaluation
docker-compose exec openserge bash scripts/run_roadtracer_evaluation.sh \
  checkpoints/best_model.pt 8
```

See [DOCKER.md](DOCKER.md) for comprehensive Docker documentation including GPU setup, volume mounts, troubleshooting, and production deployment.

---

## 📚 Citation

If you use OpenSERGE in your research, please cite the original SERGE paper:

```bibtex
@InProceedings{Bahl_2022_CVPR,
    author    = {Bahl, Gaetan and Bahri, Mehdi and Lafarge, Florent},
    title     = {Single-Shot End-to-End Road Graph Extraction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {1403-1412}
}
```

**Original paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Bahl_Single-Shot_End-to-End_Road_Graph_Extraction_CVPRW_2022_paper.html)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Clone repository
git clone https://github.com/gaetanbahl/OpenSERGE.git
cd OpenSERGE

# Install in development mode
pip install -e .

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Original SERGE method: [Bahl et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Bahl_Single-Shot_End-to-End_Road_Graph_Extraction_CVPRW_2022_paper.html)
- [timm](https://github.com/huggingface/pytorch-image-models): PyTorch Image Models library
- [Sat2Graph](https://github.com/songtaohe/Sat2Graph): CityScale dataset
- [RoadTracer](https://github.com/mitroadmaps/roadtracer): RoadTracer dataset
- [SpaceNet](https://spacenet.ai/): SpaceNet Challenge datasets
- [GlobalScale](https://github.com/htyin/samroadpp): Global-Scale Road Dataset

---

## 📞 Contact

For questions, issues, or collaboration inquiries:

- **GitHub Issues**: [https://github.com/gaetanbahl/OpenSERGE/issues](https://github.com/gaetanbahl/OpenSERGE/issues)
- **Email**: gaetan.bahl@gmail.com
