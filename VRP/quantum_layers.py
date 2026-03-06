import importlib.util
import os
from pathlib import Path

import torch
import torch.nn as nn


def decoder_config_from_env():
    backend = os.getenv("QGAT_DECODER_BACKEND", "classical").strip().lower()
    config = {
        "ansatz_name": os.getenv("QGAT_QNN_ANSATZ", "pce").strip().lower(),
        "n_qubits": int(os.getenv("QGAT_QNN_QUBITS", "8")),
        "n_layers": int(os.getenv("QGAT_QNN_LAYERS", "4")),
        "rotation": os.getenv("QGAT_QNN_ROTATION", "RXRYRZ").strip(),
        "topology": os.getenv("QGAT_QNN_TOPOLOGY", "brickwall").strip(),
    }
    return backend, config


def _load_pce_ansatz():
    ansatz_path = Path(__file__).resolve().parents[3] / "ansatz" / "pce.py"
    spec = importlib.util.spec_from_file_location("qgat_ansatz_pce", ansatz_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load ansatz module from {ansatz_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "ansatz"):
        raise ImportError(f"Ansatz module {ansatz_path} does not define ansatz()")
    return module.ansatz


def _build_ansatz(ansatz_name):
    if ansatz_name == "pce":
        return _load_pce_ansatz()
    raise ValueError(f"Unsupported ansatz '{ansatz_name}'")


class HybridQuantumLinear(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_qubits=8,
        n_layers=4,
        rotation="RXRYRZ",
        topology="brickwall",
        ansatz_name="pce",
        bias=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rotation = rotation
        self.topology = topology
        self.ansatz = _build_ansatz(ansatz_name)

        try:
            import pennylane as qml
        except ImportError as exc:
            raise ImportError(
                "PennyLane is required when decoder_backend='qnn'"
            ) from exc

        self.qml = qml
        try:
            self.device = qml.device("lightning.qubit", wires=n_qubits)
        except Exception:
            self.device = qml.device("default.qubit", wires=n_qubits)

        self.input_proj = nn.Linear(input_dim, n_qubits, bias=bias)
        self.output_proj = nn.Linear(n_qubits, output_dim, bias=bias)
        self.theta = nn.Parameter(torch.zeros(n_layers, n_qubits, len(_parse_rot_sequence(rotation))))
        self._qnode = qml.QNode(self._circuit, self.device, interface="torch")

    def _circuit(self, encoded_inputs, theta):
        for wire in range(self.n_qubits):
            self.qml.RY(encoded_inputs[wire], wires=wire)

        self.ansatz(
            theta,
            R_type=self.rotation,
            n_qubits=self.n_qubits,
            topology=self.topology,
        )
        return [self.qml.expval(self.qml.PauliZ(wire)) for wire in range(self.n_qubits)]

    def forward(self, x):
        encoded = torch.tanh(self.input_proj(x)) * torch.pi
        outputs = []
        for sample in encoded:
            q_out = self._qnode(sample, self.theta)
            outputs.append(torch.stack(q_out) if isinstance(q_out, (list, tuple)) else q_out)
        stacked = torch.stack(outputs, dim=0).to(dtype=x.dtype, device=x.device)
        return self.output_proj(stacked)


class SwitchableLinear(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        bias=False,
        backend="classical",
        qnn_config=None,
    ):
        super().__init__()
        self.backend = backend
        qnn_config = qnn_config or {}
        if backend == "classical":
            self.layer = nn.Linear(input_dim, output_dim, bias=bias)
        elif backend == "qnn":
            self.layer = HybridQuantumLinear(
                input_dim=input_dim,
                output_dim=output_dim,
                bias=bias,
                **qnn_config,
            )
        else:
            raise ValueError(f"Unsupported decoder backend '{backend}'")

    def forward(self, x):
        return self.layer(x)


def _parse_rot_sequence(rotation):
    cleaned = rotation.upper().replace(" ", "")
    if len(cleaned) % 2 != 0:
        raise ValueError(f"Invalid rotation sequence '{rotation}'")
    return [cleaned[i:i + 2] for i in range(0, len(cleaned), 2)]
