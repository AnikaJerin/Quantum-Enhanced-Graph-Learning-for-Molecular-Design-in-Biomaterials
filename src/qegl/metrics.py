from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, accuracy_score, f1_score

def regression_metrics(y_true, y_pred):
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2":  float(r2_score(y_true, y_pred))}

def classification_metrics(y_true, y_score, threshold=0.5):
    y_prob = y_score
    if y_prob.ndim > 1 and y_prob.shape[1] == 2:
        y_prob = y_prob[:,1]
    y_hat = (y_prob >= threshold).astype(int)
    out = {
        "ACC": float(accuracy_score(y_true, y_hat)),
        "F1":  float(f1_score(y_true, y_hat)),
    }
    try:
        out["AUROC"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["AUROC"] = float("nan")
    return out
