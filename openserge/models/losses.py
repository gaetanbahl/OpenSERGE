from typing import Dict
import torch, torch.nn.functional as F

def sigmoid_focal_loss(inputs, targets, alpha: float=0.25, gamma: float=2.0, reduction='mean'):
    p = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = p*targets + (1-p)*(1-targets)
    loss = ce * ((1-p_t)**gamma)
    if alpha >= 0:
        alpha_t = alpha*targets + (1-alpha)*(1-targets)
        loss = alpha_t * loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def masked_mse(pred, target, mask):
    # pred, target: [..., D], mask: [..., 1] or [...]
    mask = mask.float()
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    diff = (pred - target) ** 2
    num = (diff * mask).sum()
    den = mask.sum() * pred.shape[-1]
    return num / (den.clamp_min(1.0))

def openserge_losses(outputs, targets, j_alpha=0.25, j_gamma=2.0):
    """Compute junctionness focal loss, offset masked MSE, and edge BCE (OPTIMIZED).

    targets (per-batch dict) must contain:
      - 'junction_map': [B,1,h,w] in {0,1}
      - 'offset_map':   [B,2,h,w] with valid vectors in [-0.5,0.5]
      - 'offset_mask':  [B,1,h,w] mask where 1 means positive cell
      - 'edge_labels':  list of pre-aligned label tensors (one per batch item)
                       Each tensor has shape [E] matching model's edge_logits

    Note: Use create_edge_labels_from_model() to create pre-aligned edge_labels.
          This eliminates expensive alignment operations during loss computation.
    """
    j_logits = outputs['cnn']['junction_logits']
    off = outputs['cnn']['offset']

    j_tgt = targets['junction_map'].to(j_logits.device)
    off_tgt = targets['offset_map'].to(off.device)
    off_mask = targets['offset_mask'].to(off.device)

    # Junction loss
    Lj = sigmoid_focal_loss(j_logits, j_tgt, alpha=j_alpha, gamma=j_gamma)

    # Offset loss (masked MSE over positive cells)
    Lo = masked_mse(off, off_tgt, off_mask)

    # Edge loss: OPTIMIZED - labels are pre-aligned, no expensive lookup needed!
    edge_labels_list = targets.get('edge_labels', [])

    if len(edge_labels_list) == 0:
        # Fallback to old method if edge_labels not provided
        Le = _compute_edge_loss_legacy(outputs, targets)
    else:
        # FAST PATH: Pre-aligned labels, vectorized computation
        Le = _compute_edge_loss_vectorized(outputs, edge_labels_list)

    return {'L_junction': Lj, 'L_offset': Lo, 'L_edge': Le}


def _compute_edge_loss_vectorized(outputs, edge_labels_list):
    """Vectorized edge loss computation with pre-aligned labels (FAST)."""
    graphs = outputs['graphs']

    # Collect all edge logits and labels from batch
    all_logits = []
    all_labels = []

    for b, graph in enumerate(graphs):
        if b >= len(edge_labels_list):
            continue

        edge_logits = graph.get('edge_logits')
        labels = edge_labels_list[b]

        # Skip empty graphs
        if edge_logits is None or edge_logits.numel() == 0 or labels.numel() == 0:
            continue

        # Ensure labels match logits size
        if edge_logits.size(0) != labels.size(0):
            continue

        all_logits.append(edge_logits)
        all_labels.append(labels)

    # Compute loss over entire batch in one operation (vectorized!)
    if len(all_logits) > 0:
        batch_logits = torch.cat(all_logits, dim=0)  # [total_edges]
        batch_labels = torch.cat(all_labels, dim=0)  # [total_edges]
        Le = F.binary_cross_entropy_with_logits(batch_logits, batch_labels, reduction='mean')
    else:
        Le = torch.tensor(0.0, device=graphs[0]['nodes'].device if len(graphs) > 0 else 'cpu')

    return Le


def _compute_edge_loss_legacy(outputs, targets):
    """Legacy edge loss with alignment (SLOW - kept for backward compatibility)."""
    graphs = outputs['graphs']
    edge_lists = targets.get('edge_lists', [])

    Le = 0.0
    num_edges = 0
    device = graphs[0]['nodes'].device if len(graphs) > 0 else 'cpu'

    for b, (src_gt, dst_gt, labels_gt) in enumerate(edge_lists):
        if b >= len(graphs):
            continue

        # Check if this batch item has any nodes/edges
        if graphs[b]['nodes'].numel() == 0 or labels_gt.numel() == 0:
            continue

        # Get edge info from model
        edge_logits = graphs[b].get('edge_logits')
        edge_src = graphs[b].get('edge_src')
        edge_dst = graphs[b].get('edge_dst')

        if edge_logits is None or edge_logits.numel() == 0:
            continue

        # Create a lookup dict from ground truth edges to labels
        gt_edge_dict = {}
        for i in range(src_gt.size(0)):
            s, d = src_gt[i].item(), dst_gt[i].item()
            gt_edge_dict[(s, d)] = labels_gt[i].item()

        # Align labels with model's edges
        aligned_labels = torch.zeros_like(edge_logits)
        for i in range(edge_src.size(0)):
            s, d = edge_src[i].item(), edge_dst[i].item()
            aligned_labels[i] = gt_edge_dict.get((s, d), 0.0)

        # Compute BCE loss
        Le = Le + F.binary_cross_entropy_with_logits(edge_logits, aligned_labels, reduction='sum')
        num_edges += edge_logits.numel()

    # Average over all edges in the batch
    if num_edges > 0:
        Le = Le / num_edges
    else:
        Le = torch.tensor(0.0, device=device)

    return Le
