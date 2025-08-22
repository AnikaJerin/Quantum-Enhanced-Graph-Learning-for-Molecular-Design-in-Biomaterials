from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool

class SmallGIN(nn.Module):
    def __init__(self, in_dim=9, hidden=64, out_dim=32):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        nn2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.g1 = GINConv(nn1)
        self.g2 = GINConv(nn2)
        self.lin = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        h = self.g1(x, edge_index).relu()
        h = self.g2(h, edge_index).relu()
        h = global_mean_pool(h, batch)
        return self.lin(h)

class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x): return self.net(x)
