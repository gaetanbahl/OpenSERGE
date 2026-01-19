# OpenSERGE Methodology: Implementation Enhancements

This document provides detailed technical documentation of the key modifications and enhancements implemented in OpenSERGE compared to the original SERGE method. These modifications improve training stability, computational efficiency, multi-dataset compatibility, and overall performance.

---

## Table of Contents

1. [Three-Stage Training Strategy](#1-three-stage-training-strategy)
2. [Flexible Backbone Architecture](#2-flexible-backbone-architecture-with-automatic-adaptation)
3. [Multi-Scale Feature Aggregation (FPN)](#3-multi-scale-feature-aggregation-fpn)
4. [Optimized Edge Loss Computation](#4-optimized-edge-loss-computation)
5. [Ground Sampling Distance (GSD) Normalization](#5-ground-sampling-distance-gsd-normalization)
6. [Positional Encoding for GNN](#6-positional-encoding-for-gnn)
7. [Flexible Graph Connectivity Priors](#7-flexible-graph-connectivity-priors)
8. [Focal Loss for Junction Detection](#8-focal-loss-for-junction-detection)
9. [Conditional Junction Extraction](#9-conditional-junction-extraction-for-gnn-training)
10. [Sliding Window Inference](#10-sliding-window-inference-with-node-deduplication)
11. [Multi-Dataset Support](#11-multi-dataset-support-and-unified-data-pipeline)
12. [Evaluation Infrastructure](#12-evaluation-infrastructure)

---

## 1. Three-Stage Training Strategy

While the original paper describes a single-stage end-to-end training approach, we implement a **progressive three-stage training strategy** that significantly improves convergence and final performance.

### Stage 1: Junction Detection Pre-training

We first train only the CNN backbone and junction detection head (junction-ness and offset branches) with the edge prediction loss disabled (λ_edge = 0).

**Key characteristics**:
- Focuses on learning robust junction localization without the complexity of edge prediction
- Backbone is trainable by default, allowing it to learn features optimized for junction detection
- Ground truth edge information is not loaded (`skip_edges = True`), reducing memory usage
- Training continues until early stopping (default patience: 100 epochs for CityScale)

### Stage 2: GNN Training with Frozen CNN

After junction detection converges, we freeze all CNN parameters (backbone and detection heads) and train only the GNN and edge classification components.

**Key characteristics**:
- Junction and offset losses are disabled (λ_junction = λ_offset = 0)
- Focus optimization solely on edge prediction
- **Critical innovation**: We use **ground truth junction supervision** - instead of using predicted junctions from the CNN, we extract junctions directly from the ground truth graph and feed them to the GNN
- This eliminates error propagation from junction detection during GNN training
- The frozen CNN continues to provide node features via the node feature branch
- Inspired by curriculum learning principles

### Stage 3: Full Model Fine-tuning

Finally, we unfreeze all parameters and perform end-to-end fine-tuning with all loss components active.

**Key characteristics**:
- Learning rate is reduced by a factor (default: 0.3×) to prevent catastrophic forgetting
- Ground truth junctions are no longer used - the model operates in full inference mode
- Allows the CNN and GNN to adapt to each other
- Model learns to compensate for realistic junction detection errors

### Checkpoint Restoration

After each stage's early stopping, we restore the best checkpoint from that stage before proceeding to the next. This ensures each subsequent stage starts from the optimal parameters learned in the previous stage, rather than the last epoch before early stopping.

---

## 2. Flexible Backbone Architecture with Automatic Adaptation

The original paper uses a ResNet-50 backbone with fixed architecture. We implement a highly flexible backbone system using the [timm library](https://github.com/huggingface/pytorch-image-models) that supports both CNNs and Vision Transformers with automatic architecture adaptation.

### CNN Backbones

For CNN-based architectures, we use `features_only=True` mode to extract intermediate feature maps at multiple scales.

**Automatic adaptation**:
- System automatically detects available feature levels
- Selects outputs at strides [4, 8, 16, 32] for FPN compatibility
- Channel dimensions automatically inferred from model's `feature_info` metadata
- Eliminates manual configuration

**Supported CNN examples**:
- `resnet50`, `resnet50d.a1_in1k`
- `efficientnet_b0`, `efficientnet_b3`
- `convnext_tiny`, `convnext_base`

### Vision Transformer Backbones

For Vision Transformers, which output patch token sequences `[B, N_patches, D_embed]` rather than spatial feature maps, we implement automatic reshaping and stride conversion:

1. Enable `dynamic_img_size=True` to support arbitrary input resolutions
2. Reshape token sequence to spatial grid: `[B, D_embed, H/p, W/p]` where p is patch size
3. For patch-16 models (p=16, stride-16):
   - Apply 2× downsampling convolution to reach stride-32
   - `Conv2d(embed_dim, 256, kernel_size=3, stride=2, padding=1)`
4. For patch-32 models, use directly without downsampling

**Result**: Seamless use of modern transformer backbones like `vit_base_patch16_224.augreg_in21k` without FPN (transformers use single-scale features), while CNN backbones benefit from multi-scale aggregation.

**Supported ViT examples**:
- `vit_base_patch16_224`, `vit_large_patch16_224`
- `swin_base_patch4_window7_224`, `swin_large_patch4_window12_384`

### Pretrained Weight Handling

The system automatically:
- Checks for pretrained weight availability
- Loads ImageNet-1k/21k weights when available
- Stores and uses normalization statistics (mean, std) from pretrained models during inference
- Ensures correct preprocessing

---

## 3. Multi-Scale Feature Aggregation (FPN)

The original paper explicitly states that no Feature Pyramid Network (FPN) is used, arguing that satellite images have fixed ground sampling distance (GSD). However, we found that adding an FPN neck can slightly improve performance in some cases, but we could not prove that it is always the case. We decided to include the option to use an FPN so users can experiment for themselves.

### FPN Implementation

Our FPN aggregates features from four backbone levels at strides [4, 8, 16, 32] using a top-down pathway with lateral connections:

1. Apply 1×1 convolutions to reduce all levels to 256 channels
2. Build top-down pathway: upsample higher-level features and add to current level
3. Apply 3×3 convolutions to each level to reduce aliasing
4. Aggregate all levels to stride-32 by downsampling lower levels and summing

**Output**: A single feature map at stride-32 with 256 channels, enriched with multi-scale contextual information.

**Benefits**:
- Richer representations for both junction detection and node feature extraction
- Improved recall on small junctions
- Better precision in dense urban areas

**Important**: FPN is only used with CNN backbones. Vision Transformers inherently capture multi-scale information through self-attention and do not require (or support) FPN in our implementation.

---

## 4. Optimized Edge Loss Computation

The naive implementation of edge loss requires expensive alignment between predicted edges and ground truth edges during every forward pass. For each predicted edge (i,j), we must search through all ground truth edges to determine if it exists, leading to **O(E_pred × E_gt) complexity**.

### Pre-alignment Strategy

We introduce a pre-alignment strategy that eliminates this bottleneck:

1. During the forward pass, the model outputs `edge_src` and `edge_dst` tensors containing source and destination indices for all candidate edges, along with `edge_logits`

2. We create aligned edge labels in a separate function `create_edge_labels_from_model()`:
   - Builds a set of ground truth edges in (i,j) index space
   - Performs a single lookup to create binary labels matching the model's edge ordering

3. Pre-aligned labels are stored in the batch dictionary as `edge_labels`:
   - Loss function directly computes binary cross-entropy
   - No search operations needed

**Result**: Edge loss computation reduced from **O(E²) to O(E)**, enabling efficient training with complete graph connectivity priors and larger batch sizes.


---

## 5. Ground Sampling Distance (GSD) Normalization

Different road graph datasets are collected at varying ground sampling distances:
- **CityScale and GlobalScale**: 1.0 m/pixel
- **RoadTracer**: 0.6 m/pixel
- **SpaceNet**: 0.3 m/pixel

Training models directly on these varying resolutions leads to inconsistent junction detection behavior and poor cross-dataset generalization.

### GSD Resampling Algorithm

We implement GSD resampling to normalize all inputs to a target GSD (default: 1.0 m/pixel):

```
I_resampled = Resize(I_original, s_GSD)
```

Where the scale factor is:

```
s_GSD = GSD_source / GSD_target
```

**Example**: RoadTracer (0.6m → 1.0m):
- s_GSD = 0.6 / 1.0 = 0.6
- Image is downsampled to 60% of original size
- A 2048×2048 image becomes 1229×1229

### Coordinate Transformation

Graph coordinates are scaled accordingly:

```
(x', y') = s_GSD × (x, y)
```

### Implementation Details

- **Timing**: Resampling is applied immediately after image loading, before any cropping or data augmentation
- **Consistency**: Ensures a 32-pixel stride always corresponds to approximately 32 meters in the real world
- **Inference**: Apply same GSD resampling to input images, then scale output coordinates back to original resolution by inverse factor 1/s_GSD

### Configuration Example

```json
{
  "source_gsd": 0.6,
  "target_gsd": 1.0
}
```

---

## 6. Positional Encoding for GNN

While the original paper mentions using node coordinates as features, we implement an optional **normalized positional encoding** scheme that provides geometric context to the GNN without dominating the learned image features.

### Implementation

When `use_pos_encoding=True`, we augment node features with normalized 2D coordinates before GNN processing:

```
x_i^input = [x_i^CNN, p_i^norm]
```

Where:
- `x_i^CNN ∈ ℝ^256` are features from the CNN node feature branch
- `p_i^norm` is the normalized position:

```
p_i^norm = (1 / W_I) × [x_i, y_i] ∈ [0,1]²
```

### Benefits

1. **Spatial awareness**: The GNN learns that distant junctions are unlikely to connect, even if their image features are similar

2. **Geometric priors**: Enables learning of directional patterns (e.g., roads tend to continue in straight lines)

3. **Scale invariance**: Normalization to [0,1] ensures consistency across different image sizes

4. **Minimal overhead**: Only adds 2 dimensions to the 256-dimensional feature vectors (<1% increase)

### Results

The positional encoding is concatenated before the first EdgeConv layer, allowing the GNN to jointly reason about visual appearance and spatial relationships. This is particularly effective when using complete graph connectivity, where the GNN must distinguish true connections from many false candidate edges.

---

## 7. Flexible Graph Connectivity Priors

The original paper discusses both complete graph (E^0 = all pairs) and k-NN priors, with k=4 motivated by typical junction degrees. We implement both options and make the prior configurable via the k hyperparameter.

### Configuration Options

- **k = None**: Complete graph connectivity
  - All N(N-1)/2 possible edges
  - Better for training (more supervision signal)
  - Computationally expensive for inference

- **k = 4**: Sparse k-NN graph built in node feature space
  - Only k nearest neighbors in feature space
  - Faster inference
  - May miss long-range connections

### Efficient Implementation

Our implementation uses efficient PyTorch operations, avoiding for-loops.

**Complete graph**:
```python
src, dst = torch.combinations(torch.arange(N), r=2).T
```

**k-NN graph**:
```python
# Compute pairwise distances in feature space
dist = torch.cdist(node_features, node_features)

# Get k nearest neighbors
k_nearest = torch.topk(dist, k=k+1, largest=False, dim=1)
```

---

## 8. Focal Loss for Junction Detection

While the original paper mentions binary cross-entropy for junction classification, we implement **focal loss** as the default to address severe class imbalance as the number of junctions is typically low compared to the number of pixels in stride 32 feature maps.

### Focal Loss Formula

```
L_focal = -α_t (1 - p_t)^γ log(p_t)
```

Where:
- `p_t = p` if `y=1` else `1-p`
- `α = 0.25` (weight for positive class)
- `γ = 2.0` (focusing parameter)


---

## 9. Conditional Junction Extraction for GNN Training

To implement ground truth junction supervision in Stage 2, we extend the model's forward pass with a conditional extraction mechanism.

### Implementation

```python
def forward(self, images, use_gt_junctions=False,
            gt_junction_map=None, gt_offset_map=None):
    # ... CNN forward pass ...

    nodes = []
    for b in range(batch_size):
        if use_gt_junctions:
            # Extract from ground truth (Stage 2)
            idx = (gt_junction_map[b] > 0.5).nonzero(as_tuple=False)
            y_off = gt_offset_map[b, 0][idx[:, 0], idx[:, 1]]
            x_off = gt_offset_map[b, 1][idx[:, 0], idx[:, 1]]
        else:
            # Extract from predictions (Stage 1 and 3)
            idx = (junction_probs[b] > threshold).nonzero(as_tuple=False)
            y_off = predicted_offsets[b, 0][idx[:, 0], idx[:, 1]]
            x_off = predicted_offsets[b, 1][idx[:, 0], idx[:, 1]]

        # Convert to pixel coordinates
        x = (idx[:, 1].float() + 0.5 + x_off) * stride
        y = (idx[:, 0].float() + 0.5 + y_off) * stride
        nodes.append(torch.stack([x, y], dim=1))

    # ... GNN forward pass with extracted nodes ...
```

### Purpose

When `use_gt_junctions=True`, the model uses ground truth junction heatmaps and offset maps instead of predictions, ensuring the GNN receives perfect node inputs during Stage 2.

**Benefits**:
- Decouples CNN and GNN training
- Prevents GNN from overfitting to early-stage CNN errors
- More stable convergence
- Better final performance

---

## 10. Sliding Window Inference with Node Deduplication

Following the original paper's approach for large images, we implement **sliding window inference** with overlapping tiles. The key challenge is merging duplicate junctions detected in overlapping regions.

### Algorithm

1. **Tile extraction**: Extract junctions from all tiles, converting to global image coordinates by adding tile offsets

2. **Spatial clustering**: For each unprocessed junction j_i, find all junctions within distance d_merge (default: 16 pixels)

3. **Merge clusters**: Merge each cluster by averaging positions:
   ```
   p_merged = (1 / |cluster|) × Σ p_j for j in cluster
   ```

4. **Edge remapping**: Remap all edges referencing clustered nodes to the merged node

5. **Deduplication**: Remove duplicate edges (same source-destination pair)

### Configuration

```bash
python -m openserge.infer \
  --weights checkpoint.pt \
  --image large_image.png \
  --img_size 512 \        # Tile size
  --stride 448 \          # Overlap (smaller = more overlap)
  --merge_threshold 16.0  # Distance threshold for merging
```

### Results

This produces a unified graph covering the entire large image while avoiding fragmentation at tile boundaries.

**Performance**:
- 8192×8192 image: ~256 tiles with stride=448
- Inference time: ~30s on GPU (vs. 0.1s per tile)
- Memory efficient: processes one tile at a time

---

## 11. Multi-Dataset Support and Unified Data Pipeline

We implement a unified dataset interface supporting four major road graph datasets with different formats.

### Supported Datasets

1. **CityScale (Sat2Graph)**
   - Format: Pickle adjacency dictionaries
   - Structure: `{(y,x): [(y1,x1), ...]}`
   - GSD: 1.0 m/pixel
   - Split: train/valid/test via JSON

2. **GlobalScale**
   - Format: Hierarchical directory structure with refined graph annotations
   - Structure: Organized by country/region/tile
   - GSD: 1.0 m/pixel
   - Split: in-domain-test, out-of-domain

3. **RoadTracer**
   - Format: JSON format with embedded node/edge lists
   - Structure: Per-city graph files
   - GSD: 0.6 m/pixel
   - Split: automatic train/val, fixed test cities

4. **SpaceNet**
   - Format: Dense graphs with bottom-left origin (Y-flip required)
   - Structure: Per-region pickle files
   - GSD: 0.3 m/pixel
   - Split: train/test via JSON

### Unified Internal Representation

All datasets are converted to a common format:
- **Images**: RGB arrays in [0, 255]
- **Graphs**: Rasterized to junction heatmaps, offset fields, and edge lists
- **Coordinates**: (y, x) format with origin at top-left

This allows seamless switching between datasets without model architecture changes.

### Dataset Factory

```python
def get_dataset(config, split='train'):
    dataset_name = config.get('dataset', 'cityscale').lower()

    if dataset_name == 'cityscale':
        return CityScale(config['data_root'], ...)
    elif dataset_name == 'globalscale':
        return GlobalScale(config['data_root'], ...)
    elif dataset_name == 'roadtracer':
        return RoadTracer(config['data_root'], ...)
    elif dataset_name == 'spacenet':
        return SpaceNet(config['data_root'], ...)
```

---

## 12. Evaluation Infrastructure

We provide comprehensive evaluation scripts for computing TOPO and APLS metrics on all supported datasets.

### Evaluation Scripts

1. **`scripts/run_cityscale_evaluation.sh`**
   - Dataset: CityScale/Sat2Graph (20cities)
   - Splits: train/valid/test/all
   - Metrics: TOPO, APLS
   - Parallel processing supported

2. **`scripts/run_roadtracer_evaluation.sh`**
   - Dataset: RoadTracer (15 test cities)
   - Cities: Amsterdam, Boston, Chicago, Denver, Kansas City, LA, Montreal, New York, Paris, Pittsburgh, Salt Lake City, San Diego, Tokyo, Toronto, Vancouver
   - Graph format conversion: Text → Pickle
   - Metrics: TOPO, APLS

3. **`scripts/run_globalscale_evaluation.sh`**
   - Dataset: GlobalScale
   - Splits: in-domain-test, out-of-domain
   - Hierarchical path handling
   - Metrics: TOPO, APLS

### Metrics

**TOPO (Topology-Preserving Metric)**:
- Precision: Fraction of predicted edges that match ground truth
- Recall: Fraction of ground truth edges that are predicted
- F1-score: Harmonic mean of precision and recall

**APLS (Average Path Length Similarity)**:
- Measures similarity of shortest paths between all node pairs
- Accounts for both topology and geometry
- Range: [0, 1] where 1 = perfect match

### Parallel Processing

All scripts support parallel processing via GNU parallel:

```bash
bash scripts/run_cityscale_evaluation.sh \
  checkpoints/best_model.pt \
  8 \    # Number of parallel workers
  test
```

### Output Format

Results are saved in JSON format with per-region breakdowns:

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
      "topo_f1": 0.8286,
      "apls": 0.7834
    }
  }
}
```

---

## References

**Focal Loss**:
- Lin, T.Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *ICCV 2017*.

**Curriculum Learning**:
- Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. *ICML 2009*.

**timm Library**:
- Wightman, R. (2019). PyTorch Image Models. GitHub repository: https://github.com/rwightman/pytorch-image-models

**Original SERGE Paper**:
- Bahl, G., Bahri, M., & Lafarge, F. (2022). Single-Shot End-to-End Road Graph Extraction. *CVPR Workshops 2022*.
