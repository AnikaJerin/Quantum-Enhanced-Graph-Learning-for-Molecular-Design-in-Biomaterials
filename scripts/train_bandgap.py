#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
import numpy as np
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from qegl.data.qm9 import load_qm9_like
from qegl.quantum.qkernels import make_quantum_kernel, kernel_matrix
from qegl.utils import load_yaml, set_seed, ensure_dir


def center_kernel(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    one = np.ones((n, n)) / n
    return K - one @ K - K @ one + one @ K @ one


def evaluate_baseline(X, y, n_folds=3, seed=42):
    print(f"\n{'='*30}")
    print("BASELINES")
    print(f"{'='*30}")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Linear baseline
    lin_scores = []
    for fold, (tr, te) in enumerate(kf.split(X), start=1):
        model = LinearRegression()
        model.fit(X[tr], y[tr])
        y_hat = model.predict(X[te])
        lin_scores.append({
            "MAE": mean_absolute_error(y[te], y_hat),
            "R2": r2_score(y[te], y_hat),
        })
    print(f"Linear MAE={np.mean([s['MAE'] for s in lin_scores]):.4f}, R2={np.mean([s['R2'] for s in lin_scores]):.4f}")

    # RBF SVR baseline with tiny grid
    svr = SVR(kernel='rbf')
    grid = GridSearchCV(
        svr,
        param_grid={"C": [1.0, 10.0, 100.0], "gamma": [0.1, 0.01, 0.001], "epsilon": [0.1, 0.5]},
        cv=n_folds,
        scoring='neg_mean_absolute_error',
        n_jobs=1,
    )
    grid.fit(X, y)
    print(f"SVR best params: {grid.best_params_}")
    print(f"SVR CV MAE: {-grid.best_score_:.4f}")

    return {"linear": lin_scores, "svr": grid.best_params_}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="runs/bandgap")
    parser.add_argument("--baseline", action="store_true", help="Run classical baselines for comparison")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))
    ensure_dir(args.outdir)

    # Load data
    print(f"Loading data from {args.csv}...")
    X, y, smiles = load_qm9_like(args.csv, num_features=cfg.get("num_features", 4), max_rows=cfg.get("max_molecules"))

    if X.size == 0 or y.size == 0:
        raise RuntimeError("No valid molecules found after cleaning. Check your CSV.")

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Target mean: {y.mean():.2f}, std: {y.std():.2f}")

    # Scale to zero-mean/unit-var, then map to [0, pi]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    x_min = np.percentile(X_scaled, 1, axis=0)
    x_max = np.percentile(X_scaled, 99, axis=0)
    den = np.where((x_max - x_min) == 0, 1.0, (x_max - x_min))
    X_angles = (np.clip(X_scaled, x_min, x_max) - x_min) / den * np.pi

    # Baselines
    if args.baseline:
        evaluate_baseline(X_scaled, y, n_folds=cfg.get("outer_folds", 3), seed=cfg.get("seed", 42))

    # Quantum kernel
    print(f"\nCreating quantum kernel with {cfg.get('num_features', 4)} features, {cfg.get('reps', 1)} reps...")
    qk = make_quantum_kernel(
        num_features=cfg.get("num_features", 4),
        reps=cfg.get("reps", 1),
        entanglement=cfg.get("entanglement", "linear"),
    )

    n_folds = cfg.get("outer_folds", 3)
    if n_folds > len(X):
        n_folds = max(2, len(X) - 1)
        print(f"Adjusted folds to {n_folds} due to tiny dataset")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=cfg.get("seed", 42))

    # Expanded alpha grid for KRR with precomputed kernel
    alpha_grid = [0.1, 1.0, 10.0, 100.0, 1000.0]
    fold_scores = []

    print(f"\n{'='*30}")
    print("QUANTUM KERNEL MODEL")
    print(f"{'='*30}")

    for fold, (tr, te) in enumerate(kf.split(X_angles), start=1):
        print(f"\n[Fold {fold}] Training on {len(tr)} samples, testing on {len(te)} samples...")
        K_train = kernel_matrix(qk, X_angles[tr])
        K_train = center_kernel(K_train)
        K_test = kernel_matrix(qk, X_angles[te], X_angles[tr])
        # Center test using train centering terms
        n_tr = K_train.shape[0]
        one_tr = np.ones((n_tr, n_tr)) / n_tr
        K_test = K_test - (one_tr @ K_train)[:K_test.shape[0], :] - K_test @ one_tr + (one_tr @ K_train @ one_tr)[:K_test.shape[0], :]

        # Center targets on the training fold and add mean back at prediction
        y_tr = y[tr]
        y_tr_mean = float(np.mean(y_tr))
        y_tr_centered = y_tr - y_tr_mean

        best_mae, best_r2, best_alpha = 1e9, -1e9, None
        for a in alpha_grid:
            model = KernelRidge(alpha=a, kernel="precomputed")
            model.fit(K_train, y_tr_centered)
            y_hat_centered = model.predict(K_test)
            y_hat = y_hat_centered + y_tr_mean
            mae = mean_absolute_error(y[te], y_hat)
            r2 = r2_score(y[te], y_hat)
            if mae < best_mae:
                best_mae, best_r2, best_alpha = mae, r2, a
        print(f"[Fold {fold}] MAE={best_mae:.4f}  R2={best_r2:.4f}  (alpha={best_alpha})")
        fold_scores.append({"MAE": best_mae, "R2": best_r2, "alpha": best_alpha})

    mae_scores = [s["MAE"] for s in fold_scores]
    r2_scores = [s["R2"] for s in fold_scores]

    print(f"\n{'='*50}")
    print("QUANTUM KERNEL RESULTS")
    print(f"{'='*50}")
    print(f"MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"R²:  {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"Best MAE: {np.min(mae_scores):.4f}")
    print(f"Best R²:  {np.max(r2_scores):.4f}")

    # Save
    import json
    results = {
        "config": cfg,
        "quantum_scores": fold_scores,
        "summary": {
            "mean_mae": float(np.mean(mae_scores)),
            "std_mae": float(np.std(mae_scores)),
            "mean_r2": float(np.mean(r2_scores)),
            "std_r2": float(np.std(r2_scores)),
        }
    }
    with open(f"{args.outdir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.outdir}/results.json")


if __name__ == "__main__":
    main()

