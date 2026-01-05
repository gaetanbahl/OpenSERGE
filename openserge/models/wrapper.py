from typing import Dict, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from .net import SingleShotRoadGraphNet
from .gnn import RoadGraphGNN
from ..utils.graph import knn_graph, complete_graph

class OpenSERGE(nn.Module):
    def __init__(self, backbone='resnet50', nfeat=256, gnn_layers=(256,256,256), scorer_hidden=128, k: int=None, use_fpn=False, use_pos_encoding=False, img_size=512):
        super().__init__()
        self.ss = SingleShotRoadGraphNet(backbone=backbone, nfeat=nfeat, use_fpn=use_fpn)
        self.k = k  # if None -> complete graph, else k-NN prior
        self.proj = nn.Identity()  # if you want extra projection on node feature map per-paper
        self.gnn = RoadGraphGNN(c_in=nfeat, layers=gnn_layers, scorer_hidden=scorer_hidden,
                                use_pos_encoding=use_pos_encoding, img_size=img_size)

    @property
    def normalize_mean(self):
        """Get normalization mean from backbone's pretrained config."""
        return self.ss.backbone.normalize_mean

    @property
    def normalize_std(self):
        """Get normalization std from backbone's pretrained config."""
        return self.ss.backbone.normalize_std

    def forward(self, images, j_thr=0.5, e_thr=0.5, max_nodes=2000,
                use_gt_junctions=False, gt_junction_map=None, gt_offset_map=None):
        # Step 1: CNN
        out = self.ss(images)
        j_logits = out['junction_logits']  # [B,1,h,w]
        offs = out['offset']               # [B,2,h,w]
        nmap = out['node_feats_map']      # [B,C,h,w]
        stride = out['stride']

        B, _, h, w = j_logits.shape
        results = []
        for b in range(B):
            # Conditional junction extraction: GT or predicted
            if use_gt_junctions and gt_junction_map is not None and gt_offset_map is not None:
                # Extract junctions from ground truth
                j_map = gt_junction_map[b, 0]
                mask = j_map > 0.5
                if mask.sum() == 0:
                    results.append({'nodes': torch.empty((0,2), device=images.device), 'edges': torch.empty((0,2), dtype=torch.long, device=images.device)})
                    continue
                idx = mask.nonzero(as_tuple=False)  # [N,2] (i,j) grid coords
                if idx.size(0) > max_nodes:
                    # Prioritize junctions by GT heatmap values
                    vals = j_map[idx[:,0], idx[:,1]]
                    topk = torch.topk(vals, max_nodes).indices
                    idx = idx[topk]
                # Use GT offsets
                y_off = gt_offset_map[b, 0][idx[:,0], idx[:,1]]  # Channel 0 = y-offset
                x_off = gt_offset_map[b, 1][idx[:,0], idx[:,1]]  # Channel 1 = x-offset
            else:
                # Extract junctions from predictions (original logic)
                j_log = j_logits[b,0]
                j_prob = torch.sigmoid(j_log)
                mask = j_prob > j_thr
                if mask.sum() == 0:
                    results.append({'nodes': torch.empty((0,2), device=images.device), 'edges': torch.empty((0,2), dtype=torch.long, device=images.device)})
                    continue
                idx = mask.nonzero(as_tuple=False)  # [N,2] (y,x)
                if idx.size(0) > max_nodes:
                    # keep top-K by prob
                    vals = j_prob[idx[:,0], idx[:,1]]
                    topk = torch.topk(vals, max_nodes).indices
                    idx = idx[topk]
                y_off = offs[b,0][idx[:,0], idx[:,1]]  # Channel 0 = y-offset
                x_off = offs[b,1][idx[:,0], idx[:,1]]  # Channel 1 = x-offset
            # Map to image coords
            x = (idx[:,1].float() + 0.5 + x_off) * stride
            y = (idx[:,0].float() + 0.5 + y_off) * stride
            nodes_xy = torch.stack([x, y], dim=-1)  # [N,2]

            # Node features from nmap at cell locations
            nfeat = nmap[b,:,idx[:,0], idx[:,1]].transpose(0,1)  # [N,C]

            # Build prior edges
            if nodes_xy.size(0) <= 1:
                edges = torch.empty((0,2), dtype=torch.long, device=images.device)
                results.append({'nodes': nodes_xy, 'node_feats': nfeat, 'edges': edges})
                continue
            if self.k is None:
                src, dst = complete_graph(nodes_xy.size(0), images.device)
            else:
                src, dst = knn_graph(nfeat, self.k)  # k-NN in feature space
            # GNN message passing
            x_emb = self.gnn(nfeat, src, dst, nodes_xy=nodes_xy)
            # Edge scoring
            logits = self.gnn.score_edges(x_emb, src, dst)
            probs = torch.sigmoid(logits)
            # Retain edges with p>0.5 (tunable)
            keep = probs > e_thr
            edges = torch.stack([src[keep], dst[keep]], dim=-1)
            results.append({
                'nodes': nodes_xy,
                'node_feats': x_emb,
                'edges': edges,
                'edge_probs': probs[keep],
                'edge_logits': logits,  # All edge logits before filtering
                'edge_src': src,  # Source indices for all edges
                'edge_dst': dst   # Destination indices for all edges
            })
        return {'cnn': out, 'graphs': results}
