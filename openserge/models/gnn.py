import torch, torch.nn as nn, torch.nn.functional as F

class EdgeConv(nn.Module):
    """Dynamic Graph CNN EdgeConv operator.
    e_ij = ReLU( theta(x_j - x_i) + phi(x_i) ), with aggregation max_j e_ij
    """
    def __init__(self, c_in, c_out):
        super().__init__()
        self.theta = nn.Linear(c_in, c_out, bias=False)
        self.phi = nn.Linear(c_in, c_out, bias=False)
        self.bn = nn.BatchNorm1d(c_out)
    def forward(self, x, src, dst):
        # x: [N, C], edges as directed lists src, dst: [E]
        xi = x[src]  # [E, C]
        xj = x[dst]  # [E, C]
        e = F.relu(self.theta(xj - xi) + self.phi(xi))
        # aggregate per node with scatter_max
        N = x.size(0)
        out = x.new_zeros((N, e.size(1)))
        # use scatter max
        max_vals = out - 1e9
        max_vals.index_put_((src,), e, accumulate=True)  # this is SUM; implement max manually
        # Manual max aggregation
        # We'll do segment-wise max via torch_scatter-free trick:
        out = out - 1e9
        out.index_put_((src,), e, accumulate=True)  # sum placeholder
        # Fallback: use max by bucket (approx). For clarity & minimal deps, use scatter_reduce in PyTorch>=2.0
        if hasattr(torch, 'scatter_reduce'):
            out = torch.zeros((N, e.size(1)), device=x.device, dtype=e.dtype)
            out.scatter_reduce_(0, src.unsqueeze(-1).expand_as(e), e, reduce='amax', include_self=False)
        else:
            # approximate with max per node using loop (small graphs)
            out = torch.full((N, e.size(1)), -1e9, device=x.device, dtype=e.dtype)
            for i in range(e.size(0)):
                s = src[i]
                out[s] = torch.maximum(out[s], e[i])
        out = self.bn(out)
        return F.relu(out)

class MLPScore(nn.Module):
    def __init__(self, c_in, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*c_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
    def forward(self, xi, xj):
        z = torch.cat([xi, xj], dim=-1)
        return self.net(z).squeeze(-1)  # logits

class RoadGraphGNN(nn.Module):
    def __init__(self, c_in=256, layers=(256,256,256), scorer_hidden=128):
        super().__init__()
        convs = []
        c_prev = c_in
        for c in layers:
            convs.append(EdgeConv(c_prev, c))
            c_prev = c
        self.convs = nn.ModuleList(convs)
        self.scorer = MLPScore(c_prev, scorer_hidden)
    def forward(self, x, edges_src, edges_dst):
        for conv in self.convs:
            x = conv(x, edges_src, edges_dst)
        return x  # node embeddings
    def score_edges(self, x, edges_src, edges_dst):
        return self.scorer(x[edges_src], x[edges_dst])
