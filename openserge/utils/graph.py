import torch

def knn_graph(x, k: int):
    # x: [N, C]
    N = x.size(0)
    # If we have fewer nodes than k+1, use all other nodes
    k_actual = min(k, N - 1)
    if k_actual <= 0:
        # Not enough nodes for any edges
        return torch.empty(0, dtype=torch.long, device=x.device), torch.empty(0, dtype=torch.long, device=x.device)

    with torch.no_grad():
        dist = torch.cdist(x, x)  # [N, N]
        topk = dist.topk(k_actual+1, largest=False).indices[:, 1:]  # exclude self
    # edge list (i -> j) for undirected: add both directions
    src = torch.arange(N, device=x.device).unsqueeze(1).repeat(1, k_actual).reshape(-1)
    dst = topk.reshape(-1)
    return src, dst

def complete_graph(n: int, device):
    # all pairs i<j, undirected as two directed edges
    idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=device)
    src = torch.cat([idx_i, idx_j], dim=0)
    dst = torch.cat([idx_j, idx_i], dim=0)
    return src, dst

def collate_fn(batch):
    """Custom collate function to handle variable-length edge lists."""
    images = torch.stack([item['image'] for item in batch])
    junction_maps = torch.stack([item['junction_map'] for item in batch])
    offset_maps = torch.stack([item['offset_map'] for item in batch])
    offset_masks = torch.stack([item['offset_mask'] for item in batch])
    edges = [item['edges'] for item in batch]  # Keep as list
    meta = [item['meta'] for item in batch]

    return {
        'image': images,
        'junction_map': junction_maps,
        'offset_map': offset_maps,
        'offset_mask': offset_masks,
        'edges': edges,
        'meta': meta
    }

def create_edge_labels(junction_map, edges_list, device='cuda'):
    """
    DEPRECATED: Use create_edge_labels_from_model() instead for better performance.

    Create edge labels for training the edge classifier.

    Args:
        junction_map: [B, 1, h, w] ground truth junction map
        edges_list: List of edge lists, one per batch item.
                    Each edge list is [((i1,j1), (i2,j2)), ...] in grid coordinates
        device: torch device

    Returns:
        List of (src, dst, labels) tuples, one per batch item
    """
    B = junction_map.size(0)
    edge_labels_batch = []

    for b in range(B):
        j_map = junction_map[b, 0]  # [h, w]
        edges_gt = edges_list[b]  # List of ((i1,j1), (i2,j2))

        # Extract grid coordinates where junctions exist
        mask = j_map > 0.5
        if mask.sum() == 0:
            # No junctions, return empty
            edge_labels_batch.append((
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.float, device=device)
            ))
            continue

        idx = mask.nonzero(as_tuple=False)  # [N, 2] where each row is (i, j)
        N = idx.size(0)

        if N <= 1:
            # Need at least 2 nodes to have edges
            edge_labels_batch.append((
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.float, device=device)
            ))
            continue

        # Create mapping from (i,j) grid coords to node index
        coord_to_idx = {}
        for node_idx in range(N):
            i, j = idx[node_idx].tolist()
            coord_to_idx[(i, j)] = node_idx

        # Build graph structure (same as model)
        # Note: For k-NN we'd need node features, but ground truth doesn't have them
        # So we always use complete graph for ground truth edge labels
        src, dst = complete_graph(N, device)

        # Create ground truth edge set from edges_list
        gt_edge_set = set()
        for (i1, j1), (i2, j2) in edges_gt:
            if (i1, j1) in coord_to_idx and (i2, j2) in coord_to_idx:
                idx1 = coord_to_idx[(i1, j1)]
                idx2 = coord_to_idx[(i2, j2)]
                # Add both directions since graph is typically undirected
                gt_edge_set.add((idx1, idx2))
                gt_edge_set.add((idx2, idx1))

        # Label each edge in the graph
        labels = torch.zeros(src.size(0), dtype=torch.float, device=device)
        for e in range(src.size(0)):
            s, d = src[e].item(), dst[e].item()
            if (s, d) in gt_edge_set:
                labels[e] = 1.0

        edge_labels_batch.append((src, dst, labels))

    return edge_labels_batch


def create_edge_labels_from_model(model_graphs, gt_edges_list, junction_logits, stride, device='cuda'):
    """
    Create edge labels aligned with model's predicted graph structure (VECTORIZED & OPTIMIZED).

    This function creates edge labels that are already aligned with the model's edge predictions,
    eliminating the need for expensive alignment in the loss function.

    Args:
        model_graphs: List of graph dicts from model output, one per batch item
        gt_edges_list: List of GT edge lists, each is [((i1,j1), (i2,j2)), ...]
        junction_logits: [B, 1, h, w] junction logits from model (for extracting node positions)
        stride: Downsampling stride (e.g., 32)
        device: torch device

    Returns:
        List of labels tensors, one per batch item, aligned with model's edge_src/edge_dst
    """
    labels_batch = []

    for b, graph in enumerate(model_graphs):
        edge_src = graph.get('edge_src')
        edge_dst = graph.get('edge_dst')
        nodes = graph.get('nodes')  # [N, 2] in pixel coordinates

        # Handle empty graphs
        if edge_src is None or edge_src.numel() == 0 or nodes.numel() == 0:
            labels_batch.append(torch.empty(0, dtype=torch.float32, device=device))
            continue

        num_edges = edge_src.size(0)
        num_nodes = nodes.size(0)

        # Convert node pixel coordinates back to grid coordinates
        # nodes are in pixel coords, convert to grid: (i, j) = (y // stride, x // stride)
        nodes_grid = (nodes / stride).long()  # [N, 2] where each row is [x_grid, y_grid]
        # Note: nodes format is [x, y] so we need [j, i] for grid coords
        nodes_grid_ji = nodes_grid.flip(1)  # [N, 2] -> [[j0, i0], [j1, i1], ...]

        gt_edges = gt_edges_list[b]

        # Handle no GT edges
        if len(gt_edges) == 0:
            labels_batch.append(torch.zeros(num_edges, dtype=torch.float32, device=device))
            continue

        # Convert GT edges to tensor format: [[i1, j1, i2, j2], ...]
        gt_edges_tensor = torch.tensor(
            [[i1, j1, i2, j2] for (i1, j1), (i2, j2) in gt_edges],
            dtype=torch.long,
            device=device
        )  # [E_gt, 4]

        # Create GT edge matrix for fast lookup (vectorized)
        # For each GT edge, find the corresponding node indices
        gt_src_coords = gt_edges_tensor[:, :2]  # [E_gt, 2] = [[i1, j1], ...]
        gt_dst_coords = gt_edges_tensor[:, 2:]  # [E_gt, 2] = [[i2, j2], ...]

        # Find matching nodes for GT edges using broadcasting
        # nodes_grid_ji is [N, 2], gt_src_coords is [E_gt, 2]
        # We want to find which node index corresponds to each GT coord

        # Match source nodes: nodes_grid_ji[node_idx] == gt_src_coords[edge_idx]
        src_matches = (nodes_grid_ji.unsqueeze(0) == gt_src_coords.unsqueeze(1)).all(dim=2)  # [E_gt, N]
        dst_matches = (nodes_grid_ji.unsqueeze(0) == gt_dst_coords.unsqueeze(1)).all(dim=2)  # [E_gt, N]

        # Get node indices for each GT edge
        src_indices = src_matches.long().argmax(dim=1)  # [E_gt]
        dst_indices = dst_matches.long().argmax(dim=1)  # [E_gt]

        # Verify that we found valid matches (handle edges outside detected junctions)
        src_valid = src_matches.any(dim=1)  # [E_gt]
        dst_valid = dst_matches.any(dim=1)  # [E_gt]
        edge_valid = src_valid & dst_valid  # [E_gt]

        # Filter to valid GT edges only
        if edge_valid.any():
            valid_gt_src = src_indices[edge_valid]  # [E_valid]
            valid_gt_dst = dst_indices[edge_valid]  # [E_valid]

            # Create edge pairs: [E_valid, 2]
            gt_edge_pairs = torch.stack([valid_gt_src, valid_gt_dst], dim=1)

            # Also add reverse direction (undirected graph)
            gt_edge_pairs = torch.cat([
                gt_edge_pairs,
                torch.stack([valid_gt_dst, valid_gt_src], dim=1)
            ], dim=0)  # [2*E_valid, 2]

            # Create predicted edge pairs: [E_pred, 2]
            pred_edge_pairs = torch.stack([edge_src, edge_dst], dim=1)  # [E_pred, 2]

            # Vectorized edge matching using broadcasting
            # Check if each predicted edge exists in GT edges
            # pred_edge_pairs[i] == gt_edge_pairs[j] for some j
            matches = (pred_edge_pairs.unsqueeze(1) == gt_edge_pairs.unsqueeze(0)).all(dim=2)  # [E_pred, 2*E_valid]
            labels = matches.any(dim=1).float()  # [E_pred]
        else:
            # No valid GT edges found
            labels = torch.zeros(num_edges, dtype=torch.float32, device=device)

        labels_batch.append(labels)

    return labels_batch
