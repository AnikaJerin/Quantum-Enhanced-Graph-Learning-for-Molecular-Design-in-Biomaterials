from __future__ import annotations
import argparse, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from qegl.utils import load_yaml, set_seed, ensure_dir
from qegl.data.qm9 import load_qm9_like
from qegl.quantum.qkernels import make_quantum_kernel, kernel_matrix
from qegl.metrics import classification_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--csv", type=str, default="data/tox_small.csv")
    ap.add_argument("--outdir", type=str, default="runs/toxicity")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))
    ensure_dir(args.outdir)

    X, y, smiles = load_qm9_like(args.csv, num_features=cfg.get("num_features", 8), max_rows=cfg.get("max_molecules"))
    y = (y > 0.5).astype(int)  

    qk = make_quantum_kernel(
        num_features=cfg.get("num_features", 8),
        reps=cfg.get("reps", 2),
        entanglement=cfg.get("entanglement", "linear"),
    )

    skf = StratifiedKFold(n_splits=cfg.get("outer_folds", 3), shuffle=True, random_state=cfg.get("seed", 42))
    scores = []
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        K_train = kernel_matrix(qk, X[tr])
        K_test = kernel_matrix(qk, X[te], X[tr])
        clf = SVC(kernel="precomputed", C=cfg.get("C", 1.0), probability=True)
        clf.fit(K_train, y[tr])
        y_prob = clf.predict_proba(K_test)[:,1]
        m = classification_metrics(y[te], y_prob)
        print(f"[Fold {fold}] ACC={m['ACC']:.3f}  F1={m['F1']:.3f}  AUROC={m['AUROC']:.3f}")
        scores.append(m)

    def mean(k): return float(np.nanmean([s[k] for s in scores]))
    print(f"[Mean] ACC={mean('ACC'):.3f}  F1={mean('F1'):.3f}  AUROC={mean('AUROC'):.3f}")

if __name__ == "__main__":
    main()
