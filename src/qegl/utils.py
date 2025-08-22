from __future__ import annotations
import os, random, yaml
import numpy as np

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed: int = 42):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def standardize_features(X: np.ndarray):
    """Return standardized features and (mean, std) to reuse on test."""
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mean) / std, (mean, std)
