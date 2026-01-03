"""
Inference script for SpaceNet road graph extraction.

Handles single 400×400 images resized to 512×512 for inference,
then scales predictions back to original 400×400 resolution.
"""
import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch

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
        description='OpenSERGE SpaceNet inference (single 400×400 image)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output
    ap.add_argument('--weights', type=str, required=True,
                    help='Path to model checkpoint (.pt file)')
    ap.add_argument('--image', type=str, required=True,
                    help='Path to input image (400×400 PNG)')
    ap.add_argument('--output', type=str, default='output_graph.p',
                    help='Output path for extracted graph (pickle format)')
    ap.add_argument('--output_json', type=str, default=None,
                    help='Optional: output path for JSON format graph')
    ap.add_argument('--output_image', type=str, default=None,
                    help='Optional: output path for visualization image')

    # Model parameters
    ap.add_argument('--img_size', type=int, default=512,
                    help='Model input size (image will be resized to this)')
    ap.add_argument('--junction_thresh', type=float, default=0.5,
                    help='Threshold for junction detection (0-1)')
    ap.add_argument('--edge_thresh', type=float, default=0.5,
                    help='Threshold for edge prediction (0-1)')
    ap.add_argument('--k', type=int, default=None,
                    help='k for k-NN graph prior (None = complete graph)')
    ap.add_argument('--max_nodes', type=int, default=2000,
                    help='Maximum nodes to process')

    # Visualization parameters
    ap.add_argument('--node_color', type=str, default='red',
                    help='Node color for visualization')
    ap.add_argument('--edge_color', type=str, default='yellow',
                    help='Edge color for visualization')
    ap.add_argument('--node_size', type=int, default=3,
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


def load_model(checkpoint_path: str, k: Optional[int], device: torch.device, logger) -> OpenSERGE:
    """Load model from checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config from checkpoint
    config = checkpoint.get('config', {})

    # Use checkpoint config with command-line overrides
    model_k = k if k is not None else config.get('k')
    model_backbone = config.get('backbone', 'resnet50')
    model_use_fpn = config.get('use_fpn', False)
    model_use_pos_encoding = config.get('use_pos_encoding', False)
    model_img_size = config.get('img_size', 512)

    logger.info(f"Model configuration:")
    logger.info(f"  Backbone: {model_backbone}")
    logger.info(f"  k: {model_k if model_k is not None else 'complete graph'}")
    logger.info(f"  FPN: {model_use_fpn}")
    logger.info(f"  Position encoding: {model_use_pos_encoding}")
    logger.info(f"  Image size: {model_img_size}")

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

    return model


def load_and_preprocess_image(image_path: str, target_size: int, logger) -> tuple:
    """
    Load image and resize to target size.

    Returns:
        tuple: (resized_image, original_size, scale_factor)
    """
    logger.info(f"Loading image from {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_h, original_w = img.shape[:2]
    logger.info(f"Original image size: {original_h}×{original_w}")

    # Resize to target size
    img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    scale_factor = target_size / original_w
    logger.info(f"Resized to {target_size}×{target_size} (scale factor: {scale_factor:.3f})")

    return img_resized, (original_h, original_w), scale_factor


def run_inference(model: OpenSERGE, img: np.ndarray, junction_thresh: float,
                  edge_thresh: float, max_nodes: int, device: torch.device, logger) -> Dict:
    """
    Run model inference on a single image.

    Returns:
        dict with keys: 'nodes' ([N, 2] array in x,y), 'edges' ([M, 2] array), 'edge_probs' ([M] array)
    """
    logger.info("Running inference...")

    # Prepare image tensor
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]

    # Run model
    with torch.no_grad():
        output = model(img_tensor, j_thr=junction_thresh, e_thr=edge_thresh, max_nodes=max_nodes)

    # Extract graph from batch
    graph = output['graphs'][0]  # First (and only) image in batch

    nodes = graph['nodes'].cpu().numpy()  # [N, 2] in (x, y)
    edges = graph['edges'].cpu().numpy()  # [M, 2] edge indices
    edge_probs = graph['edge_probs'].cpu().numpy()  # [M]

    logger.info(f"Detected {len(nodes)} junctions, {len(edges)} edges")

    return {'nodes': nodes, 'edges': edges, 'edge_probs': edge_probs}


def scale_graph_to_original(graph: Dict, scale_factor: float, logger) -> Dict:
    """
    Scale node coordinates back to original image resolution.

    Args:
        graph: Graph with nodes in resized image coordinates
        scale_factor: Scale factor used for resizing

    Returns:
        Graph with nodes in original image coordinates
    """
    nodes_scaled = graph['nodes'] / scale_factor
    logger.info(f"Scaled nodes back to original resolution (scale factor: {1/scale_factor:.3f})")

    return {
        'nodes': nodes_scaled,
        'edges': graph['edges'],
        'edge_probs': graph['edge_probs']
    }


def graph_to_adjacency_dict(graph: Dict, flip_y: bool = False, img_height: Optional[int] = None) -> Dict:
    """
    Convert graph to Sat2Graph adjacency dictionary format.

    Args:
        graph: Graph dict with 'nodes' [N,2] (x,y) and 'edges' [M,2]
        flip_y: Whether to flip Y coordinates (for SpaceNet compatibility)
        img_height: Image height for Y-flipping

    Returns:
        Adjacency dict: {(y, x): [(y1, x1), (y2, x2), ...]}
    """
    nodes = graph['nodes']  # [N, 2] in (x, y)
    edges = graph['edges']  # [M, 2]

    # Build adjacency list
    adj_dict = {}

    for i in range(len(nodes)):
        x, y = nodes[i]

        # Flip Y if needed (SpaceNet uses bottom-left origin)
        if flip_y and img_height is not None:
            y = img_height - y

        # Initialize node in dict
        node_key = (y, x)  # Sat2Graph format uses (y, x)
        if node_key not in adj_dict:
            adj_dict[node_key] = []

    # Add edges
    for edge_idx in edges:
        i, j = edge_idx
        xi, yi = nodes[i]
        xj, yj = nodes[j]

        # Flip Y if needed
        if flip_y and img_height is not None:
            yi = img_height - yi
            yj = img_height - yj

        node_i = (yi, xi)
        node_j = (yj, xj)

        # Add bidirectional edges
        if node_j not in adj_dict[node_i]:
            adj_dict[node_i].append(node_j)
        if node_i not in adj_dict[node_j]:
            adj_dict[node_j].append(node_i)

    return adj_dict


def save_graph(graph: Dict, output_path: str, logger, flip_y: bool = True, img_height: int = None):
    """Save graph to pickle file (Sat2Graph adjacency dict format)."""
    logger.info(f"Saving graph to {output_path}")

    # Convert to adjacency dict
    adj_dict = graph_to_adjacency_dict(graph, flip_y=flip_y, img_height=img_height)

    # Save as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(adj_dict, f)

    logger.info(f"Saved {len(adj_dict)} nodes")


def save_graph_json(graph: Dict, output_path: str, logger):
    """Save graph to JSON file (simple node/edge list format)."""
    logger.info(f"Saving graph to JSON: {output_path}")

    # Convert to JSON-serializable format
    graph_json = {
        'nodes': graph['nodes'].tolist(),  # [N, 2] (x, y)
        'edges': graph['edges'].tolist(),  # [M, 2]
        'edge_probs': graph['edge_probs'].tolist()  # [M]
    }

    with open(output_path, 'w') as f:
        json.dump(graph_json, f, indent=2)

    logger.info(f"Saved {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")


def visualize_graph(img: np.ndarray, graph: Dict, output_path: str,
                    node_color: str, edge_color: str, node_size: int, edge_width: int, logger):
    """Create visualization of extracted graph overlaid on image."""
    logger.info(f"Creating visualization: {output_path}")

    # Convert to BGR for OpenCV
    vis_img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)

    nodes = graph['nodes']
    edges = graph['edges']

    # Color mapping
    color_map = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'cyan': (255, 255, 0),
        'magenta': (255, 0, 255),
        'white': (255, 255, 255)
    }

    node_bgr = color_map.get(node_color.lower(), (0, 0, 255))
    edge_bgr = color_map.get(edge_color.lower(), (0, 255, 255))

    # Draw edges first (so nodes are on top)
    for edge_idx in edges:
        i, j = edge_idx
        pt1 = tuple(nodes[i].astype(int))
        pt2 = tuple(nodes[j].astype(int))
        cv2.line(vis_img, pt1, pt2, edge_bgr, edge_width)

    # Draw nodes
    for node in nodes:
        pt = tuple(node.astype(int))
        cv2.circle(vis_img, pt, node_size, node_bgr, -1)

    # Save
    cv2.imwrite(output_path, vis_img)
    logger.info(f"Saved visualization to {output_path}")


def main():
    args = parse_args()
    logger = setup_logging(args.verbose)

    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Load model
    model = load_model(args.weights, args.k, device, logger)

    # Load and preprocess image
    img_resized, (original_h, original_w), scale_factor = load_and_preprocess_image(
        args.image, args.img_size, logger
    )

    # Run inference on resized image
    graph = run_inference(model, img_resized, args.junction_thresh, args.edge_thresh,
                         args.max_nodes, device, logger)

    # Scale graph back to original resolution
    graph_original = scale_graph_to_original(graph, scale_factor, logger)

    # Save graph (pickle format with Y-flip for SpaceNet compatibility)
    save_graph(graph_original, args.output, logger, flip_y=True, img_height=original_h)

    # Save JSON if requested
    if args.output_json:
        save_graph_json(graph_original, args.output_json, logger)

    # Create visualization if requested
    if args.output_image:
        # Load original image for visualization
        img_original = cv2.imread(args.image, cv2.IMREAD_COLOR)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

        visualize_graph(img_original, graph_original, args.output_image,
                       args.node_color, args.edge_color, args.node_size, args.edge_width, logger)

    logger.info("Inference complete!")


if __name__ == '__main__':
    main()
