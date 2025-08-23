from __future__ import annotations

import torch
from torch import nn


class HybridGNNQ(nn.Module):
    """
    Hybrid model: GNN encoder -> Quantum head (reg + clf).
    """
    def __init__(self, gnn_encoder: nn.Module, quantum_head: nn.Module):
        super().__init__()
        self.gnn = gnn_encoder
        self.qhead = quantum_head

    def forward(self, data):
        # Graph encoder to get per-graph embeddings
        g = self.gnn(data)  # [B, hidden]
        y_reg, y_logit = self.qhead(g)
        return y_reg, y_logit
