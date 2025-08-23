# qegl/models/classical_head.py
from __future__ import annotations
import torch
from torch import nn
from typing import Tuple

class ClassicalHead(nn.Module):
    """
    Simple MLP head that takes GNN embedding and outputs (regression, classification).
    Use as a baseline to compare to the quantum head.
    tasks: (do_reg, do_clf)
    """
    def __init__(self, embedding_dim: int, hidden: int = 64, tasks: Tuple[bool, bool] = (True, True)):
        super().__init__()
        self.do_reg, self.do_clf = tasks
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2),
            nn.ReLU()
        )
        if self.do_reg:
            self.head_reg = nn.Linear(hidden//2, 1)
        if self.do_clf:
            self.head_clf = nn.Linear(hidden//2, 1)

    def forward(self, z: torch.Tensor):
        h = self.mlp(z)
        y_reg = self.head_reg(h).squeeze(-1) if self.do_reg else None
        y_logit = self.head_clf(h).squeeze(-1) if self.do_clf else None
        return y_reg, y_logit
