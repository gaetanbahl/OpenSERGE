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
