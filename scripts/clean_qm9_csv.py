#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from qegl.featurize import mol_from_smiles


def is_valid_smiles(s: str) -> bool:
    try:
        mol_from_smiles(s)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Input CSV with at least a 'smiles' column")
    ap.add_argument("--out_csv", default="data/qm9_clean.csv", help="Output cleaned CSV path")
    ap.add_argument("--target_col", default="gap", help="Target column name to keep (e.g., gap or toxic)")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if "smiles" not in df.columns:
        raise ValueError("Input CSV must contain a 'smiles' column")

    keep_cols = [c for c in ["smiles", args.target_col] if c in df.columns]
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in CSV")

    valid_mask = df["smiles"].apply(is_valid_smiles)
    df_clean = df.loc[valid_mask, keep_cols].reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df_clean.to_csv(args.out_csv, index=False)

    print(f"Cleaned CSV written: {args.out_csv}")
    print(f"Valid rows: {len(df_clean)} / {len(df)}")


if __name__ == "__main__":
    main()