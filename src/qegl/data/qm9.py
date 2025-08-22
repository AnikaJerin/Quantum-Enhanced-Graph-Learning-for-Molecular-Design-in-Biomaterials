import numpy as np
import pandas as pd

from ..featurize import mol_to_rdkit_features

def _detect_target_column(df: pd.DataFrame) -> str:
    for cand in ["bandgap", "target", "y", "gap", "Egap", "egap"]:
        if cand in df.columns:
            return cand
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c.lower() != "smiles"]
    if not numeric_cols:
        raise ValueError("Could not find a numeric target column in CSV.")
    return numeric_cols[0]

def load_qm9_like(csv_path: str, num_features: int = 16, max_rows=None):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    if max_rows:
        df = df.head(max_rows)

    if "smiles" not in df.columns:
        raise ValueError("CSV must have a 'smiles' column.")

    target_col = _detect_target_column(df)
    print(f"Using target column: {target_col}")

    X_list, y_list, smiles_list = [], [], []
    total = len(df)
    invalid = 0

    for _, row in df.iterrows():
        s = str(row["smiles"])
        feats = mol_to_rdkit_features(s, num_features=num_features)
        if feats is None:
            invalid += 1
            continue
        X_list.append(feats)
        y_list.append(float(row[target_col]))
        smiles_list.append(s)

    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=float)

    print(f"Loaded {len(smiles_list)} valid molecules out of {total} (skipped {invalid} invalid).")
    return X, y, smiles_list
