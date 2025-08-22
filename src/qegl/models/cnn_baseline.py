from __future__ import annotations
import io
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import numpy as np

def smiles_to_image(smiles: str, size: int=224) -> Image.Image:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return Image.new("RGB", (size, size), color=(0,0,0))
    d2d = rdMolDraw2D.MolDraw2DCairo(size, size)
    rdMolDraw2D.PrepareAndDrawMolecule(d2d, m)
    d2d.FinishDrawing()
    png = d2d.GetDrawingText()
    return Image.open(io.BytesIO(png))

class ResNetImageBaseline(nn.Module):
    def __init__(self, out_dim=1):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, out_dim)

    def forward(self, x):
        return self.backbone(x)
