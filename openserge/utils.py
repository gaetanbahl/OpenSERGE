import math, json, os, torch, torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np

def build_grid(h: int, w: int, device=None):
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    return x, y  # each [H, W]

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

def to_image_coords(u_off, v_off, cell_xy, stride, in_hw, out_hw):
    # u_off, v_off in [-0.5, 0.5], cell centers = (cell_xy + 0.5) * stride
    # Return x, y in input image coords
    cx = (cell_xy[..., 0] + 0.5 + u_off) * stride
    cy = (cell_xy[..., 1] + 0.5 + v_off) * stride
    # clamp to image size
    H, W = in_hw
    cx = cx.clamp(0, W-1)
    cy = cy.clamp(0, H-1)
    return cx, cy

def knn_graph(x, k: int):
    # x: [N, C]
    with torch.no_grad():
        dist = torch.cdist(x, x)  # [N, N]
        topk = dist.topk(k+1, largest=False).indices[:, 1:]  # exclude self
    # edge list (i -> j) for undirected: add both directions
    src = torch.arange(x.size(0), device=x.device).unsqueeze(1).repeat(1, k).reshape(-1)
    dst = topk.reshape(-1)
    return src, dst

def complete_graph(n: int, device):
    # all pairs i<j, undirected as two directed edges
    idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=device)
    src = torch.cat([idx_i, idx_j], dim=0)
    dst = torch.cat([idx_j, idx_i], dim=0)
    return src, dst
