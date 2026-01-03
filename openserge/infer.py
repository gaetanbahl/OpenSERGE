"""
Inference script for OpenSERGE road graph extraction.

Performs sliding window inference on large images with proper node deduplication
at tile boundaries (as described in the paper). Optionally outputs visualization.
"""
import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .models.wrapper import OpenSERGE


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(
        description='OpenSERGE road graph extraction inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output
    ap.add_argument('--weights', type=str, required=True,
                    help='Path to model checkpoint (.pt file)')
    ap.add_argument('--image', type=str, required=True,
                    help='Path to input image (PNG, JPG, etc.)')
    ap.add_argument('--output', type=str, default='output_graph.json',
                    help='Output path for extracted graph (JSON format)')
    ap.add_argument('--output_image', type=str, default=None,
                    help='Optional: output path for visualization image')

    # Tiling parameters
    ap.add_argument('--img_size', type=int, default=512,
                    help='Tile size for sliding window inference')
    ap.add_argument('--stride', type=int, default=448,
                    help='Stride for sliding window (smaller = more overlap)')

    # Model parameters
    ap.add_argument('--junction_thresh', type=float, default=0.5,
                    help='Threshold for junction detection (0-1)')
    ap.add_argument('--edge_thresh', type=float, default=0.5,
                    help='Threshold for edge prediction (0-1)')
    ap.add_argument('--k', type=int, default=None,
                    help='k for k-NN graph prior (None = complete graph)')
    ap.add_argument('--backbone', type=str, default='resnet50',
                    help='CNN backbone architecture')
    ap.add_argument('--max_nodes', type=int, default=2000,
                    help='Maximum nodes per tile')

    # Node deduplication parameters
    ap.add_argument('--merge_threshold', type=float, default=16.0,
                    help='Distance threshold (pixels) for merging duplicate nodes')

    # Visualization parameters
    ap.add_argument('--node_color', type=str, default='red',
                    help='Node color for visualization')
    ap.add_argument('--edge_color', type=str, default='yellow',
                    help='Edge color for visualization')
    ap.add_argument('--node_size', type=int, default=4,
                    help='Node radius for visualization')
    ap.add_argument('--edge_width', type=int, default=2,
                    help='Edge line width for visualization')

    # Device
    ap.add_argument('--device', type=str, default='cuda',
                    choices=['cuda', 'mps', 'cpu'],
                    help='Device to run inference on')
    ap.add_argument('--verbose', action='store_true',
                    help='Enable verbose logging')

    return ap.parse_args()


def load_model(checkpoint_path: str, k: Optional[int], backbone: str, device: torch.device, logger) -> Tuple[OpenSERGE, Tuple, Tuple]:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        k: k for k-NN prior (None = complete graph)
        backbone: Backbone architecture
        device: Device to load model on
        logger: Logger instance

    Returns:
        (model, normalize_mean, normalize_std) where:
            model: Loaded OpenSERGE model in eval mode
            normalize_mean: Normalization mean (or None)
            normalize_std: Normalization std (or None)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config from checkpoint if available
    config = checkpoint.get('config', {})

    # Use checkpoint config with command-line overrides
    model_k = k if k is not None else config.get('k')
    model_backbone = config.get('backbone', backbone)
    model_use_fpn = config.get('use_fpn', False)
    model_use_pos_encoding = config.get('use_pos_encoding', False)
    model_img_size = config.get('img_size', 512)

    # Extract normalization parameters from config
    normalize_mean = config.get('normalize_mean')
    normalize_std = config.get('normalize_std')

    logger.info(f"Model configuration:")
    logger.info(f"  Backbone: {model_backbone}")
    logger.info(f"  k: {model_k if model_k is not None else 'complete graph'}")
    logger.info(f"  FPN: {model_use_fpn}")
    logger.info(f"  Position encoding: {model_use_pos_encoding}")
    logger.info(f"  Image size: {model_img_size}")
    if normalize_mean is not None and normalize_std is not None:
        logger.info(f"  Normalization: mean={normalize_mean}, std={normalize_std}")
    else:
        logger.info(f"  Normalization: None (using [0,1] scaling)")

    if 'epoch' in checkpoint:
        logger.info(f"  Checkpoint epoch: {checkpoint['epoch']}")
    if 'val_losses' in checkpoint and checkpoint['val_losses']:
        logger.info(f"  Validation loss: {checkpoint['val_losses'].get('total', 'N/A'):.4f}")

    # Create model
    model = OpenSERGE(backbone=model_backbone, k=model_k, use_fpn=model_use_fpn,
                      use_pos_encoding=model_use_pos_encoding, img_size=model_img_size)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {num_params:,}")

    return model, normalize_mean, normalize_std


def load_image(image_path: str, logger) -> np.ndarray:
    """
    Load image from file.

    Args:
        image_path: Path to image file
        logger: Logger instance

    Returns:
        Image as RGB numpy array [H, W, 3] in uint8
    """
    logger.info(f"Loading image from {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info(f"Image shape: {img.shape}")

    return img


def extract_tiles(img: np.ndarray, tile_size: int, stride: int, logger) -> List[Tuple[np.ndarray, int, int]]:
    """
    Extract overlapping tiles from image using sliding window.

    Args:
        img: Input image [H, W, 3]
        tile_size: Size of each tile
        stride: Stride between tiles
        logger: Logger instance

    Returns:
        List of (tile, y_offset, x_offset) tuples
    """
    H, W = img.shape[:2]
    tiles = []

    # Calculate number of tiles
    n_rows = max(1, (H - tile_size) // stride + 1) if H > tile_size else 1
    n_cols = max(1, (W - tile_size) // stride + 1) if W > tile_size else 1

    logger.info(f"Extracting {n_rows * n_cols} tiles ({n_rows} rows x {n_cols} cols)")
    logger.info(f"Tile size: {tile_size}x{tile_size}, Stride: {stride}")

    for y0 in range(0, max(1, H - tile_size + 1), stride):
        for x0 in range(0, max(1, W - tile_size + 1), stride):
            # Handle edge cases where tile extends beyond image
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)

            tile = img[y0:y1, x0:x1]

            # Pad if necessary
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=tile.dtype)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded

            tiles.append((tile, y0, x0))

    return tiles


def process_tile(tile: np.ndarray, model: OpenSERGE, device: torch.device,
                 junction_thresh: float, edge_thresh: float, max_nodes: int,
                 y_offset: int, x_offset: int, normalize_mean: Tuple[float, float, float] = None,
                 normalize_std: Tuple[float, float, float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a single tile through the model.

    Args:
        tile: Tile image [H, W, 3] in uint8
        model: OpenSERGE model
        device: Device to run on
        junction_thresh: Junction detection threshold
        edge_thresh: Edge prediction threshold
        max_nodes: Maximum nodes per tile
        y_offset: Y offset of tile in original image
        x_offset: X offset of tile in original image
        normalize_mean: Mean for normalization (default None)
        normalize_std: Std for normalization (default None)

    Returns:
        (nodes, edges, edge_probs) where:
            nodes: [N, 2] node positions in global image coordinates
            edges: [M, 2] edge indices
            edge_probs: [M] edge probabilities
    """
    # Prepare input tensor
    tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0

    # Apply normalization if specified
    if normalize_mean is not None and normalize_std is not None:
        mean = torch.tensor(normalize_mean).view(3, 1, 1)
        std = torch.tensor(normalize_std).view(3, 1, 1)
        tile_tensor = (tile_tensor - mean) / std

    tile_tensor = tile_tensor.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(tile_tensor, j_thr=junction_thresh, e_thr=edge_thresh, max_nodes=max_nodes)

    # Extract graph
    graph = output['graphs'][0]
    nodes = graph['nodes'].cpu().numpy()  # [N, 2] in (x, y) format
    edges = graph['edges'].cpu().numpy()  # [M, 2]

    # Handle edge_probs - it might not be present or might be a tensor
    if 'edge_probs' in graph:
        edge_probs = graph['edge_probs'].cpu().numpy()
    else:
        edge_probs = np.ones(len(edges))

    # Transform nodes to global coordinates
    if len(nodes) > 0:
        nodes[:, 0] += x_offset  # x coordinate
        nodes[:, 1] += y_offset  # y coordinate

    return nodes, edges, edge_probs


def merge_graphs(tile_results: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                 merge_threshold: float, logger) -> Dict:
    """
    Merge graphs from multiple tiles with node deduplication.

    This implements the node merging strategy from the paper:
    - Nodes within merge_threshold distance are merged by averaging positions
    - Edges are updated to reference merged nodes
    - Duplicate edges are removed

    Args:
        tile_results: List of (nodes, edges, edge_probs) from each tile
        merge_threshold: Distance threshold for merging nodes (pixels)
        logger: Logger instance

    Returns:
        Dictionary with 'nodes' [N, 2] and 'edges' [M, 2]
    """
    if not tile_results:
        return {'nodes': [], 'edges': []}

    # Concatenate all nodes and edges
    all_nodes = []
    all_edges = []
    node_offset = 0

    for nodes, edges, edge_probs in tile_results:
        if len(nodes) > 0:
            all_nodes.append(nodes)
            if len(edges) > 0:
                # Offset edge indices
                offset_edges = edges + node_offset
                all_edges.append(offset_edges)
            node_offset += len(nodes)

    if not all_nodes:
        logger.info("No nodes detected in any tile")
        return {'nodes': [], 'edges': []}

    all_nodes = np.vstack(all_nodes)  # [N_total, 2]
    all_edges = np.vstack(all_edges) if all_edges else np.empty((0, 2), dtype=np.int64)

    logger.info(f"Before merging: {len(all_nodes)} nodes, {len(all_edges)} edges")

    # Node deduplication via spatial clustering
    # Build a mapping from old node indices to new (merged) node indices
    node_mapping = {}  # old_idx -> new_idx
    merged_nodes = []
    used = set()

    for i in range(len(all_nodes)):
        if i in used:
            continue

        # Find all nodes within merge_threshold of node i
        pos_i = all_nodes[i]
        distances = np.sqrt(np.sum((all_nodes - pos_i) ** 2, axis=1))
        cluster = np.where(distances <= merge_threshold)[0]

        # Average positions of clustered nodes
        merged_pos = all_nodes[cluster].mean(axis=0)
        new_idx = len(merged_nodes)
        merged_nodes.append(merged_pos)

        # Map all nodes in cluster to new merged node
        for idx in cluster:
            node_mapping[idx] = new_idx
            used.add(idx)

    merged_nodes = np.array(merged_nodes)
    logger.info(f"After merging: {len(merged_nodes)} nodes (merged {len(all_nodes) - len(merged_nodes)} duplicates)")

    # Remap edges to merged node indices
    remapped_edges = []
    edge_set = set()  # To remove duplicate edges

    for src, dst in all_edges:
        new_src = node_mapping[src]
        new_dst = node_mapping[dst]

        # Skip self-loops
        if new_src == new_dst:
            continue

        # Canonical edge representation (smaller index first)
        edge = tuple(sorted([new_src, new_dst]))

        if edge not in edge_set:
            edge_set.add(edge)
            remapped_edges.append([new_src, new_dst])

    remapped_edges = np.array(remapped_edges) if remapped_edges else np.empty((0, 2), dtype=np.int64)
    logger.info(f"After edge deduplication: {len(remapped_edges)} edges")

    return {
        'nodes': merged_nodes.tolist(),
        'edges': remapped_edges.tolist()
    }


def graph_to_adjacency_dict(graph: Dict) -> Dict[Tuple[float, float], List[Tuple[float, float]]]:
    """
    Convert graph to adjacency dictionary format.

    Args:
        graph: Dictionary with 'nodes' [N, 2] and 'edges' [M, 2]

    Returns:
        Dictionary where keys are node coordinates (x, y) as tuples and values
        are lists of neighboring node coordinates
    """
    nodes = graph['nodes']
    edges = graph['edges']

    # Build adjacency dictionary
    adj_dict = {}

    # Initialize all nodes with empty neighbor lists
    # NOTE: Sat2Graph format uses (Y, X) not (X, Y)!
    for node in nodes:
        # Swap x,y to y,x for Sat2Graph compatibility
        node_tuple = (node[1], node[0])
        adj_dict[node_tuple] = []

    # Add edges (undirected graph - add both directions)
    for src_idx, dst_idx in edges:
        # Swap x,y to y,x for Sat2Graph compatibility
        src_node = (nodes[src_idx][1], nodes[src_idx][0])
        dst_node = (nodes[dst_idx][1], nodes[dst_idx][0])

        # Add bidirectional edges
        if dst_node not in adj_dict[src_node]:
            adj_dict[src_node].append(dst_node)
        if src_node not in adj_dict[dst_node]:
            adj_dict[dst_node].append(src_node)

    return adj_dict


def save_graph(graph: Dict, output_path: str, logger):
    """
    Save graph to JSON file and pickle file.

    Args:
        graph: Dictionary with 'nodes' and 'edges'
        output_path: Output file path (will save both .json and .p)
        logger: Logger instance
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save JSON format
    with open(output_path, 'w') as f:
        json.dump(graph, f, indent=2)

    logger.info(f"Saved graph to {output_path}")
    logger.info(f"  Nodes: {len(graph['nodes'])}")
    logger.info(f"  Edges: {len(graph['edges'])}")

    # Save pickle format (adjacency dictionary)
    pickle_path = output_path.with_suffix('.p')
    adj_dict = graph_to_adjacency_dict(graph)

    with open(pickle_path, 'wb') as f:
        pickle.dump(adj_dict, f)

    logger.info(f"Saved adjacency dictionary to {pickle_path}")
    logger.info(f"  Format: {{(x, y): [(x1, y1), (x2, y2), ...], ...}}")


def visualize_graph(img: np.ndarray, graph: Dict, output_path: str,
                    node_color: str, edge_color: str, node_size: int, edge_width: int,
                    logger):
    """
    Visualize graph overlaid on image and save.

    Args:
        img: Original image [H, W, 3] in RGB uint8
        graph: Dictionary with 'nodes' and 'edges'
        output_path: Output image path
        node_color: Color for nodes (matplotlib color name)
        edge_color: Color for edges (matplotlib color name)
        node_size: Radius for nodes
        edge_width: Line width for edges
        logger: Logger instance
    """
    logger.info(f"Creating visualization...")

    # Convert color names to BGR for OpenCV
    color_map = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'cyan': (255, 255, 0),
        'magenta': (255, 0, 255),
        'white': (255, 255, 255),
        'lime': (0, 255, 0),
    }

    node_bgr = color_map.get(node_color.lower(), (0, 0, 255))
    edge_bgr = color_map.get(edge_color.lower(), (0, 255, 255))

    # Create visualization image (BGR for OpenCV)
    vis_img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)

    nodes = np.array(graph['nodes'])
    edges = np.array(graph['edges'])

    # Draw edges first (so nodes appear on top)
    if len(edges) > 0 and len(nodes) > 0:
        for src_idx, dst_idx in edges:
            if src_idx < len(nodes) and dst_idx < len(nodes):
                pt1 = tuple(nodes[src_idx].astype(int))
                pt2 = tuple(nodes[dst_idx].astype(int))
                cv2.line(vis_img, pt1, pt2, edge_bgr, edge_width, cv2.LINE_AA)

    # Draw nodes
    if len(nodes) > 0:
        for x, y in nodes:
            cv2.circle(vis_img, (int(x), int(y)), node_size, node_bgr, -1, cv2.LINE_AA)
            # White outline for better visibility
            cv2.circle(vis_img, (int(x), int(y)), node_size + 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_img)

    logger.info(f"Saved visualization to {output_path}")


def main():
    """Main inference pipeline."""
    args = parse_args()
    logger = setup_logging(args.verbose)

    # Setup device
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
    elif args.device == 'mps':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        if not torch.backends.mps.is_available():
            logger.warning("MPS not available, falling back to CPU")
    else:
        device = torch.device('cpu')

    logger.info(f"Using device: {device}")

    # Load model
    model, normalize_mean, normalize_std = load_model(args.weights, args.k, args.backbone, device, logger)

    # Load image
    img = load_image(args.image, logger)

    # Extract tiles
    tiles = extract_tiles(img, args.img_size, args.stride, logger)

    # Process each tile
    logger.info("Processing tiles...")
    tile_results = []

    for tile, y_offset, x_offset in tqdm(tiles, desc="Processing tiles"):
        nodes, edges, edge_probs = process_tile(
            tile, model, device,
            args.junction_thresh, args.edge_thresh, args.max_nodes,
            y_offset, x_offset, normalize_mean, normalize_std
        )
        tile_results.append((nodes, edges, edge_probs))

    # Merge graphs with node deduplication
    logger.info("Merging graphs and deduplicating nodes...")
    graph = merge_graphs(tile_results, args.merge_threshold, logger)

    # Save graph
    save_graph(graph, args.output, logger)

    # Optionally save visualization
    if args.output_image:
        visualize_graph(img, graph, args.output_image,
                       args.node_color, args.edge_color,
                       args.node_size, args.edge_width, logger)

    logger.info("Inference complete!")


if __name__ == '__main__':
    main()
