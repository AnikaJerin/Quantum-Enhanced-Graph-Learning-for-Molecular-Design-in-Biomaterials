from __future__ import annotations
import csv, numpy as np
from typing import Optional, Tuple, List
from qegl.featurize import mol_to_rdkit_features, pad_or_truncate

def load_tox21_like(csv_path: str, endpoint: str = "SR-p53", num_features: int = 32, max_rows: Optional[int] = None):
    smiles, y = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            if endpoint in row:
                lbl = row[endpoint]
            else:
                lbl = row["label"]
            if lbl == "" or lbl is None:
                continue
            lbl = int(float(lbl))
            if lbl < 0:
                continue
            s = row["smiles"]
            smiles.append(s)
            y.append(lbl)
    X = []
    for s in smiles:
        feats = mol_to_rdkit_features(s)
        X.append(pad_or_truncate(feats, num_features))
    return np.array(X, dtype=float), np.array(y, dtype=int), smiles
