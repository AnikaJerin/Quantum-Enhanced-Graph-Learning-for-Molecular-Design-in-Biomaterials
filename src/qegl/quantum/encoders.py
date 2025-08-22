from __future__ import annotations
from typing import Literal
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit.circuit import QuantumCircuit

def make_feature_map(name: Literal["ZZFeatureMap", "PauliFeatureMap"]="ZZFeatureMap",
                     num_features: int=32, reps: int=2, entanglement="linear") -> QuantumCircuit:
    if name == "ZZFeatureMap":
        return ZZFeatureMap(feature_dimension=num_features, reps=reps, entanglement=entanglement)
    elif name == "PauliFeatureMap":
        return PauliFeatureMap(feature_dimension=num_features, reps=reps, entanglement=entanglement, paulis=["ZI","IZ","ZZ"])
    else:
        raise ValueError(f"Unknown feature map: {name}")
