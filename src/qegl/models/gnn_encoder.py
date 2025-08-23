from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINConv, global_mean_pool


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GNNEncoder(nn.Module):
    """
    Simple yet strong GIN encoder with global mean pooling.
    Outputs a fixed-size graph embedding of dimension `hidden`.
    """
    def __init__(self, in_dim: int, hidden: int = 128, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        dim_in = in_dim
        for _ in range(layers):
            mlp = MLP(dim_in, hidden, hidden, dropout)
            self.layers.append(GINConv(mlp))
            dim_in = hidden

        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden = hidden

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.layers:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
        # global pooling to get graph embedding
        g = global_mean_pool(x, batch)
        return g  # shape [batch, hidden]
