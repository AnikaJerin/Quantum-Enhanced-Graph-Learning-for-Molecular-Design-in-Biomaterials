# # scripts/train_hybrid_gnn_quantum.py

# import argparse
# import yaml
# import torch
# import torch.nn.functional as F
# from torch_geometric.loader import DataLoader
# from sklearn.model_selection import KFold
# from torch.utils.data import Subset

# from qegl.data.mol_dataset import MoleculeDataset
# from qegl.models.gnn_encoder import GNNEncoder
# from qegl.models.hybrid_model import HybridGNNQ
# from qegl.models.classical_head import ClassicalHead as QuantumHead  # Using classical head

# # ---------------------------
# # Training & evaluation loops
# # ---------------------------

# def train_epoch(model, loader, optimizer, task, device):
#     model.train()
#     total_loss = 0
#     for batch in loader:
#         batch = batch.to(device)
#         optimizer.zero_grad()
#         pred = model(batch)

#         if task == "regression":
#             loss = F.mse_loss(pred.squeeze(), batch.y.float())
#         elif task == "classification":
#             loss = F.binary_cross_entropy_with_logits(pred.squeeze(), batch.y.float())
#         elif task == "multitask":
#             gap_pred, tox_pred = pred
#             loss_gap = F.mse_loss(gap_pred.squeeze(), batch.y[:, 0].float())
#             loss_tox = F.binary_cross_entropy_with_logits(tox_pred.squeeze(), batch.y[:, 1].float())
#             loss = loss_gap + loss_tox
#         else:
#             raise ValueError(f"Unknown task: {task}")

#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     return total_loss / len(loader)


# def eval_epoch(model, loader, task, device):
#     model.eval()
#     preds, targets = [], []
#     with torch.no_grad():
#         for batch in loader:
#             batch = batch.to(device)
#             pred = model(batch)

#             if task == "regression":
#                 preds.append(pred.squeeze().cpu().numpy())
#                 targets.append(batch.y.cpu().numpy())
#             elif task == "classification":
#                 preds.append(torch.sigmoid(pred).squeeze().cpu().numpy())
#                 targets.append(batch.y.cpu().numpy())
#             elif task == "multitask":
#                 gap_pred, tox_pred = pred
#                 preds.append((gap_pred.squeeze().cpu().numpy(), torch.sigmoid(tox_pred).squeeze().cpu().numpy()))
#                 targets.append((batch.y[:, 0].cpu().numpy(), batch.y[:, 1].cpu().numpy()))

#     return preds, targets

# # ---------------------------
# # K-Fold Runner
# # ---------------------------

# def run_fold(cfg, csv_file, task, device, train_index, test_index):
#     # Load full dataset
#     full_ds = MoleculeDataset(root="data", csv_file=csv_file, task=task)

#     # Subsets for train/test
#     train_ds = Subset(full_ds, train_index)
#     test_ds = Subset(full_ds, test_index)

#     train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 32), shuffle=True)
#     test_loader = DataLoader(test_ds, batch_size=cfg.get("batch_size", 32))

#     # Build model
#     hidden = int(cfg.get("hidden", 128))
#     layers = int(cfg.get("layers", 3))
#     dropout = float(cfg.get("dropout", 0.1))
#     gnn = GNNEncoder(in_dim=full_ds.num_node_features, hidden=hidden, layers=layers, dropout=dropout)

#     tasks_tuple = (
#         task in ("regression", "multitask"),
#         task in ("classification", "multitask"),
#     )

#     qhead = QuantumHead(embedding_dim=hidden, tasks=tasks_tuple)
#     model = HybridGNNQ(gnn, qhead).to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-3)))

#     epochs = int(cfg.get("epochs", 20))
#     for epoch in range(epochs):
#         train_loss = train_epoch(model, train_loader, optimizer, task, device)
#         print(f"[Fold] Epoch {epoch+1}/{epochs}, Loss={train_loss:.4f}")

#     preds, targets = eval_epoch(model, test_loader, task, device)
#     return preds, targets

# # ---------------------------
# # Main
# # ---------------------------

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, required=True, help="YAML config file")
#     parser.add_argument("--csv", type=str, required=True, help="CSV with SMILES + properties")
#     parser.add_argument("--task", type=str, choices=["regression", "classification", "multitask"], required=True)
#     args = parser.parse_args()

#     # Load config
#     with open(args.config) as f:
#         cfg = yaml.safe_load(f)

#     # Device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Read CSV
#     from pandas import read_csv
#     df = read_csv(args.csv)

#     # K-Fold
#     kf = KFold(n_splits=int(cfg.get("folds", 3)), shuffle=True, random_state=42)
#     all_preds, all_targets = [], []
#     for fold, (train_index, test_index) in enumerate(kf.split(df)):
#         print(f"=== Fold {fold+1} ===")
#         preds, targets = run_fold(cfg, args.csv, args.task, device, train_index, test_index)
#         all_preds.append(preds)
#         all_targets.append(targets)

# if __name__ == "__main__":
#     main()




# ******** HYBRID WITH QUANTUM METHOD *******

# #!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.pool import global_mean_pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem
import yaml

# local modules
from qegl.models.gnn_encoder import GNNEncoder
# from qegl.models.classical_head import ClassicalHead as QuantumHead
from qegl.models.quantum_head_qiskit import QuantumHead
from qegl.models.hybrid_model import HybridGNNQ


# -----------------------------
# Utilities
# -----------------------------
_ALLOWED_ATOMS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H, C, N, O, F, P, S, Cl, Br, I

def _atom_features(atom: Chem.Atom) -> np.ndarray:
    """
    Minimal but useful atom featurization: atom type one-hot + degree + aromatic + formal charge.
    """
    onehot = np.zeros(len(_ALLOWED_ATOMS), dtype=np.float32)
    Z = atom.GetAtomicNum()
    if Z in _ALLOWED_ATOMS:
        onehot[_ALLOWED_ATOMS.index(Z)] = 1.0
    degree = atom.GetDegree()
    aromatic = 1.0 if atom.GetIsAromatic() else 0.0
    charge = float(atom.GetFormalCharge())
    return np.concatenate([onehot, np.array([degree, aromatic, charge], dtype=np.float32)])


class MolCSVDataset(InMemoryDataset):
    """
    CSV -> PyG graphs.
    Required columns:
      - 'smiles'
      - targets: any subset of {'gap', 'toxicity'}
    """
    def __init__(self, csv_path: str, smiles_col: str = "smiles",
                 target_cols: List[str] = ["gap", "toxicity"]) -> None:
        self.csv_path = csv_path
        self.smiles_col = smiles_col
        self.target_cols = target_cols

        df = pd.read_csv(csv_path)
        if smiles_col not in df.columns:
            raise ValueError(f"CSV must contain column '{smiles_col}'")
        missing = [t for t in target_cols if t not in df.columns]
        if len(missing) > 0:
            # allow training with subset of tasks
            df = df.copy()
            for m in missing:
                df[m] = np.nan

        df = df.dropna(subset=[smiles_col])

        data_list: List[Data] = []
        for _, r in df.iterrows():
            smi = str(r[smiles_col])
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            # Node features
            x = torch.tensor(np.stack([_atom_features(a) for a in mol.GetAtoms()]), dtype=torch.float32)

            # Edges (undirected)
            row, col = [], []
            for b in mol.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                row += [i, j]
                col += [j, i]
            edge_index = torch.tensor([row, col], dtype=torch.long) if len(row) else torch.zeros((2, 0), dtype=torch.long)

            d = Data(x=x, edge_index=edge_index)
            if "gap" in df.columns and pd.notnull(r.get("gap", np.nan)):
                d.y_gap = torch.tensor([float(r["gap"])], dtype=torch.float32)
            if "toxicity" in df.columns and pd.notnull(r.get("toxicity", np.nan)):
                d.y_tox = torch.tensor([float(r["toxicity"])], dtype=torch.float32)
            data_list.append(d)

        super().__init__(root=None)
        self.data, self.slices = self.collate(data_list)

    @property
    def num_node_features(self) -> int:
        # one-hot len + degree + aromatic + charge
        return len(_ALLOWED_ATOMS) + 3


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collect_targets(ds: Subset, attr: str) -> np.ndarray:
    vals = []
    for i in ds.indices:
        y = getattr(ds.dataset[i], attr, None)
        if y is not None:
            vals.append(float(y))
    return np.array(vals, dtype=np.float32)


def train_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer,
                device: torch.device, task: str, loss_weights: Tuple[float, float],
                bce_pos_weight: torch.Tensor | None, gap_scaler: StandardScaler | None) -> float:
    model.train()
    mae_loss = nn.L1Loss()
    if bce_pos_weight is not None:
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight.to(device))
    else:
        bce_loss = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        y_reg, y_logit = model(batch)

        loss = 0.0
        # Regression: scale targets before L1
        if task in ("regression", "multitask") and hasattr(batch, "y_gap") and y_reg is not None:
            y_true = batch.y_gap.view(-1).detach().cpu().numpy()
            if gap_scaler is not None and y_true.size > 0:
                y_true_scaled = torch.tensor(gap_scaler.transform(y_true.reshape(-1, 1)).reshape(-1),
                                             dtype=torch.float32, device=device)
            else:
                y_true_scaled = torch.tensor(y_true, dtype=torch.float32, device=device)
            loss = loss + loss_weights[0] * mae_loss(y_reg.view(-1), y_true_scaled)

        # Classification
        if task in ("classification", "multitask") and hasattr(batch, "y_tox") and y_logit is not None:
            loss = loss + loss_weights[1] * bce_loss(y_logit.view(-1), batch.y_tox.view(-1))

        loss.backward()
        opt.step()

        bs = batch.num_graphs
        total_loss += float(loss.item()) * bs
        n_samples += bs

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device, task: str,
               gap_scaler: StandardScaler | None) -> dict:
    model.eval()
    y_true_reg, y_pred_reg = [], []
    y_true_cls, y_pred_cls = [], []

    for batch in loader:
        batch = batch.to(device)
        y_reg, y_logit = model(batch)

        if task in ("regression", "multitask") and hasattr(batch, "y_gap") and y_reg is not None:
            pred = y_reg.view(-1).detach().cpu().numpy()
            if gap_scaler is not None and pred.size > 0:
                pred = gap_scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1)
            y_true = batch.y_gap.view(-1).detach().cpu().numpy()
            y_true_reg.append(y_true)
            y_pred_reg.append(pred)

        if task in ("classification", "multitask") and hasattr(batch, "y_tox") and y_logit is not None:
            y_true_cls.append(batch.y_tox.view(-1).detach().cpu().numpy())
            y_pred_cls.append(torch.sigmoid(y_logit).view(-1).detach().cpu().numpy())

    metrics = {}
    if len(y_true_reg):
        yt = np.concatenate(y_true_reg)
        yp = np.concatenate(y_pred_reg)
        metrics["mae"] = float(mean_absolute_error(yt, yp))
        with np.errstate(all="ignore"):
            metrics["r2"] = float(r2_score(yt, yp)) if yt.shape[0] >= 2 else float("nan")
    if len(y_true_cls):
        yt = np.concatenate(y_true_cls)
        yp = np.concatenate(y_pred_cls)
        try:
            metrics["auc"] = float(roc_auc_score(yt, yp))
        except Exception:
            metrics["auc"] = float("nan")
    return metrics


def run_fold(df_path: str, cfg: dict, fold_idx: int, train_idx, val_idx, task: str, outdir: str) -> None:
    # Which targets to include
    if task == "regression":
        targets = ["gap"]
    elif task == "classification":
        targets = ["toxicity"]
    else:
        targets = ["gap", "toxicity"]

    ds = MolCSVDataset(df_path, smiles_col=cfg.get("smiles_col", "smiles"), target_cols=targets)
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    dl_train = DataLoader(train_ds, batch_size=int(cfg.get("batch_size", 16)), shuffle=True)
    dl_val = DataLoader(val_ds, batch_size=int(cfg.get("batch_size", 16)), shuffle=False)

    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.get("use_cuda", True)) else "cpu")

    # ---- Build model ----
    hidden = int(cfg.get("hidden", 128))
    layers = int(cfg.get("layers", 3))
    dropout = float(cfg.get("dropout", 0.1))
    gnn = GNNEncoder(in_dim=ds.num_node_features, hidden=hidden, layers=layers, dropout=dropout)

    tasks_tuple = (
        task in ("regression", "multitask"),
        task in ("classification", "multitask"),
    )

    qhead = QuantumHead(
        embedding_dim=hidden,
        n_qubits=int(cfg.get("n_qubits", 6)),
        reps_feature=int(cfg.get("reps_feature", 1)),
        reps_ansatz=int(cfg.get("reps_ansatz", 2)),
        entanglement=str(cfg.get("entanglement", "linear")),
        backend_name=str(cfg.get("quantum_backend", "aer_simulator_statevector")),
        tasks=tasks_tuple,
    )
    model = HybridGNNQ(gnn, qhead).to(device)

    # ---- Optimizer ----
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 2e-3)), weight_decay=float(cfg.get("weight_decay", 1e-5)))
    loss_weights = (float(cfg.get("reg_weight", 1.0)), float(cfg.get("clf_weight", 1.0)))

    # ---- Target scaler (regression) ----
    gap_scaler: StandardScaler | None = None
    if "gap" in targets:
        y_gap_train = _collect_targets(train_ds, "y_gap")
        if y_gap_train.size > 1:
            gap_scaler = StandardScaler().fit(y_gap_train.reshape(-1, 1))

    # ---- Class imbalance handling for tox ----
    bce_pos_weight = None
    if "toxicity" in targets:
        y_tox_train = _collect_targets(train_ds, "y_tox")
        if y_tox_train.size > 0:
            pos = float((y_tox_train > 0.5).sum())
            neg = float((y_tox_train <= 0.5).sum())
            if pos > 0 and neg > 0:
                bce_pos_weight = torch.tensor([neg / pos], dtype=torch.float32)

    # ---- Train ----
    epochs = int(cfg.get("epochs", 120))
    best = {}
    best_key = np.inf

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, dl_train, opt, device, task, loss_weights, bce_pos_weight, gap_scaler)
        val_metrics = eval_epoch(model, dl_val, device, task, gap_scaler)

        # Selection metric per task
        if task == "regression":
            select = val_metrics.get("mae", np.inf)  # lower better
        elif task == "classification":
            select = -val_metrics.get("auc", 0.0)    # higher auc -> lower key
        else:
            mae = val_metrics.get("mae", np.inf)
            auc = val_metrics.get("auc", 0.0)
            select = mae - auc

        if select < best_key:
            best_key = select
            best = dict(val_metrics)

        msg = f"[Fold {fold_idx:02d}][{epoch:03d}] train_loss={tr_loss:.4f}"
        if "mae" in val_metrics: msg += f"  MAE={val_metrics['mae']:.4f}"
        if "r2" in val_metrics and not np.isnan(val_metrics["r2"]): msg += f"  R2={val_metrics['r2']:.4f}"
        if "auc" in val_metrics: msg += f"  AUC={val_metrics['auc']:.4f}"
        print(msg)


    # Save best metrics
    os.makedirs(outdir, exist_ok=True)
    out_json = os.path.join(outdir, f"fold_{fold_idx:02d}_metrics.json")
    with open(out_json, "w") as f:
        import json as _json
        _json.dump(best, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--task", type=str, default="multitask",
                        choices=["regression", "classification", "multitask"])
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))

    df = pd.read_csv(args.csv)
    print(f"Loading data from {args.csv} ... found {len(df)} rows")

    kf = KFold(n_splits=int(cfg.get("n_folds", 5)), shuffle=True, random_state=int(cfg.get("seed", 42)))
    outdir = cfg.get("outdir", "results")

    for i, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(df)))):
        print("\n" + "=" * 30)
        print(f"FOLD {i + 1}")
        print("=" * 30)
        run_fold(args.csv, cfg, i + 1, train_idx, val_idx, args.task, outdir)


if __name__ == "__main__":
    main()
