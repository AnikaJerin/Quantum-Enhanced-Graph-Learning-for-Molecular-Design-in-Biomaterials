from __future__ import annotations

import warnings
import torch
from torch import nn

# Try to import Qiskit ML. If missing, raise a clear error at init.
_QISKIT_OK = True
try:
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.connectors import TorchConnector
except Exception:
    _QISKIT_OK = False


class QuantumHead(nn.Module):
    """
    Variational Quantum Head using Qiskit EstimatorQNN + TorchConnector.

    Pipeline:
      embedding (dim=embedding_dim)
         -> Linear downmap to n_qubits features
         -> QNN (angle encoding + EfficientSU2 style trainable ansatz)
         -> Expectation values <Z_i> for i=1..n_qubits
         -> Linear heads: (regression scalar), (classification logit)

    tasks: tuple[use_regression, use_classification]
    """
    def __init__(
        self,
        embedding_dim: int,
        n_qubits: int = 6,
        reps_feature: int = 1,
        reps_ansatz: int = 2,
        entanglement: str = "linear",
        backend_name: str = "aer_simulator_statevector",
        tasks: tuple[bool, bool] = (True, True),
    ):
        super().__init__()
        if not _QISKIT_OK:
            raise ImportError(
                "qiskit-machine-learning (and qiskit) not found. "
                "Please install: pip install 'qiskit-machine-learning>=0.6' qiskit"
            )

        self.use_reg, self.use_clf = tasks
        self.n_qubits = n_qubits

        # Map classical embedding to n_qubits angles
        self.down = nn.Linear(embedding_dim, n_qubits)

        # Build parameterized circuit
        self._build_qnn(n_qubits, reps_feature, reps_ansatz, entanglement)

        # TorchConnector wraps the QNN as a torch Module
        self.qnn_torch = TorchConnector(self.qnn)

        # Heads on top of expectation vector (size n_qubits)
        if self.use_reg:
            self.reg_head = nn.Linear(n_qubits, 1)
        else:
            self.reg_head = None

        if self.use_clf:
            self.cls_head = nn.Linear(n_qubits, 1)
        else:
            self.cls_head = None

    def _build_qnn(self, n_qubits: int, reps_feature: int, reps_ansatz: int, entanglement: str):
        # Inputs and weights as ParameterVectors
        x = ParameterVector("x", n_qubits)
        theta = ParameterVector("θ", n_qubits * reps_ansatz * 2)

        qc = QuantumCircuit(n_qubits)

        # Feature map: Ry rotations repeated
        for r in range(reps_feature):
            for i in range(n_qubits):
                qc.ry(x[i], i)
            # simple entanglement
            if entanglement in ("linear", "full"):
                for i in range(n_qubits - 1):
                    qc.cx(i, i + 1)
                if entanglement == "full" and n_qubits > 2:
                    qc.cx(n_qubits - 1, 0)

        # Ansatz: alternating RZ, RX with entanglement
        t_idx = 0
        for r in range(reps_ansatz):
            for i in range(n_qubits):
                qc.rz(theta[t_idx], i); t_idx += 1
                qc.rx(theta[t_idx], i); t_idx += 1
            # entangle
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)

        # Observables: Z on each qubit
        observables = [SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1 - i) if i == 0 else
                                                "I" * i + "Z" + "I" * (n_qubits - i - 1), 1.0)])
                       for i in range(n_qubits)]

        # Build EstimatorQNN
        self.qnn = EstimatorQNN(
            circuit=qc,
            input_params=list(x),
            weight_params=list(theta),
            observables=observables,
        )

    def forward(self, batch_embed: torch.Tensor):
        """
        batch_embed: [B, embedding_dim]
        returns: (y_reg, y_logit) where either can be None if the task is disabled
        """
        # Map to angles and clamp to a sane range
        angles = self.down(batch_embed)  # [B, n_qubits]
        # QNN expects float64 numpy via TorchConnector, but the connector handles types.
        q_out = self.qnn_torch(angles)   # [B, n_qubits], expectation values in [-1, 1]

        y_reg, y_logit = None, None
        if self.reg_head is not None:
            y_reg = self.reg_head(q_out).squeeze(-1)  # [B]
        if self.cls_head is not None:
            y_logit = self.cls_head(q_out).squeeze(-1)  # [B]
        return y_reg, y_logit
