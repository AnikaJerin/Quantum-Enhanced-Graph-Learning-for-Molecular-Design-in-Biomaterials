from __future__ import annotations
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

def _make_feature_map(num_features: int, reps: int = 1, entanglement: str | list[str] = "linear"):
    """Create a ZZFeatureMap with specified parameters."""
    return ZZFeatureMap(feature_dimension=num_features, reps=reps, entanglement=entanglement)

def _feature_state(x: np.ndarray, fmap: ZZFeatureMap) -> Statevector:
    """Convert input features to quantum state vector."""
    try:
        # Ensure input is properly formatted
        x_clean = np.asarray(x, dtype=np.float64)
        if x_clean.size != fmap.feature_dimension:
            raise ValueError(f"Expected {fmap.feature_dimension} features, got {x_clean.size}")
        
        # Bind parameters and get statevector
        bound = fmap.assign_parameters(x_clean.tolist(), inplace=False)
        return Statevector.from_instruction(bound)
    except Exception as e:
        print(f"Warning: Error in feature state conversion: {e}")
        # Return a default state if conversion fails
        return Statevector.from_instruction(fmap.assign_parameters([0.0] * fmap.feature_dimension))

def make_quantum_kernel(num_features=16, reps=1, entanglement="linear"):
    """
    Returns a callable K(x, z) = |<phi(x)|phi(z)>|^2 (fidelity kernel).
    
    Args:
        num_features: Number of input features
        reps: Number of repetitions in the feature map
        entanglement: Entanglement pattern ('linear', 'circular', 'full', or custom)
    
    Returns:
        Callable kernel function
    """
    fmap = _make_feature_map(num_features=num_features, reps=reps, entanglement=entanglement)

    def kernel(x: np.ndarray, z: np.ndarray) -> float:
        """Compute quantum kernel between two feature vectors."""
        try:
            sx = _feature_state(x, fmap)
            sz = _feature_state(z, fmap)
            
            # Compute fidelity (squared magnitude of inner product)
            inner_product = np.vdot(sx.data, sz.data)
            fidelity = float(np.abs(inner_product) ** 2)
            
            # Ensure numerical stability
            if np.isnan(fidelity) or np.isinf(fidelity):
                return 0.0
            
            return max(0.0, min(1.0, fidelity))  # Clamp to [0, 1]
            
        except Exception as e:
            print(f"Warning: Kernel computation failed: {e}")
            return 0.0
    
    return kernel

def kernel_matrix(kernel_fn, X1: np.ndarray, X2: np.ndarray | None = None, batch: int = 128) -> np.ndarray:
    """
    Compute Gram matrix using the provided kernel_fn.
    Uses symmetry to save time when X2 is None.
    
    Args:
        kernel_fn: Kernel function to use
        X1: First set of feature vectors
        X2: Second set of feature vectors (if None, uses X1)
        batch: Batch size for computation (not used in current implementation)
    
    Returns:
        Kernel matrix K where K[i,j] = kernel_fn(X1[i], X2[j])
    """
    X1 = np.asarray(X1, dtype=np.float64)
    X2 = X1 if X2 is None else np.asarray(X2, dtype=np.float64)
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2), dtype=np.float64)

    if X2 is X1:
        # Symmetric case - compute only upper triangle
        for i in range(n1):
            K[i, i] = 1.0  # Diagonal elements are always 1 for normalized states
            for j in range(i+1, n2):
                kij = kernel_fn(X1[i], X2[j])
                K[i, j] = K[j, i] = kij
    else:
        # General case - compute full matrix
        for i in range(n1):
            for j in range(n2):
                K[i, j] = kernel_fn(X1[i], X2[j])
    
    return K
