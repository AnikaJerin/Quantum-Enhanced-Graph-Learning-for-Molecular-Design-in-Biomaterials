from typing import Optional
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

# Silence RDKit parse errors (we handle invalids ourselves)
RDLogger.DisableLog("rdApp.error")

def mol_from_smiles(s: str) -> Optional[Chem.Mol]:
    """Convert SMILES -> RDKit Mol with sanitization. Return None if invalid."""
    try:
        mol = Chem.MolFromSmiles(s, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None

# Chosen 16 robust, continuous descriptors (order is fixed)
_DESC_FUNCS = [
    Descriptors.MolWt,
    Descriptors.ExactMolWt,
    Descriptors.MolLogP,
    Descriptors.NumHDonors,
    Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds,
    Descriptors.TPSA,
    Descriptors.RingCount,
    Descriptors.NumAromaticRings,
    Descriptors.NumAliphaticRings,
    Descriptors.HeavyAtomCount,
    Descriptors.FractionCSP3,
    Descriptors.BalabanJ,
    Descriptors.BertzCT,
    Descriptors.HallKierAlpha,
    Descriptors.NumValenceElectrons,
]

def _descriptor_vector(mol: Chem.Mol) -> np.ndarray:
    vals = [float(f(mol)) for f in _DESC_FUNCS]
    arr = np.asarray(vals, dtype=float)
    # Ensure no NaN/inf leaks into the model
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

def mol_to_rdkit_features(smiles: str, num_features: int = 16) -> Optional[np.ndarray]:
    """
    Return exactly `num_features` features for the molecule as a 1D numpy array.
    Uses 16 continuous RDKit descriptors; pads/truncates if num_features != 16.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None

    base = _descriptor_vector(mol)  # length 16
    if num_features == base.shape[0]:
        return base

    # Truncate or zero-pad to requested size (keeps determinism; no data leakage)
    if num_features < base.shape[0]:
        return base[:num_features]
    pad = np.zeros(num_features - base.shape[0], dtype=float)
    return np.concatenate([base, pad], axis=0)
