import os

import torch
import torch.nn as nn


def _qnn_config_from_env():
    return {
        "ansatz_name": os.getenv("QGAT_QNN_ANSATZ", "pce").strip().lower(),
        "n_qubits": int(os.getenv("QGAT_QNN_QUBITS", "8")),
        "n_layers": int(os.getenv("QGAT_QNN_LAYERS", "4")),
        "rotation": os.getenv("QGAT_QNN_ROTATION", "RXRYRZ").strip(),
        "topology": os.getenv("QGAT_QNN_TOPOLOGY", "brickwall").strip(),
    }


def decoder_config_from_env():
    backend = os.getenv("QGAT_DECODER_BACKEND", "classical").strip().lower()
    return backend, _qnn_config_from_env()


def encoder_attn_config_from_env(conv_layers):
    backend = os.getenv("QGAT_ENCODER_ATTN_BACKEND", "classical").strip().lower()
    config = _qnn_config_from_env()
    raw_selector = os.getenv("QGAT_ENCODER_ATTN_LAYERS", "0").strip()
    selected_layers = parse_layer_selector(raw_selector, conv_layers)
    return backend, config, selected_layers


def _load_pce_ansatz():
    from ansatz.pce import ansatz

    return ansatz


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
        self.ansatz_name = ansatz_name
        self.ansatz = None
        self.qml = None
        self.device = None
        self.q_layer = None

        self.input_proj = nn.Linear(input_dim, n_qubits, bias=bias)
        self.output_proj = nn.Linear(n_qubits, output_dim, bias=bias)
        self._weight_shape = (n_layers, n_qubits, len(_parse_rot_sequence(rotation)))

    def _ensure_runtime(self):
        if self.q_layer is not None:
            return

        try:
            import pennylane as qml
        except ImportError as exc:
            raise ImportError(
                "PennyLane is required when decoder_backend='qnn'"
            ) from exc

        self.qml = qml
        self.ansatz = _build_ansatz(self.ansatz_name)
        diff_method = "backprop"
        try:
            self.device = qml.device("lightning.qubit", wires=self.n_qubits)
            diff_method = "adjoint"
        except Exception:
            self.device = qml.device("default.qubit", wires=self.n_qubits)
        qnode = qml.QNode(self._circuit, self.device, interface="torch", diff_method=diff_method)
        self.q_layer = qml.qnn.TorchLayer(
            qnode,
            weight_shapes={"theta": self._weight_shape},
            init_method={"theta": torch.nn.init.zeros_},
        )

    def _circuit(self, inputs, theta):
        for wire in range(self.n_qubits):
            self.qml.RY(inputs[..., wire], wires=wire)

        self.ansatz(
            theta,
            R_type=self.rotation,
            n_qubits=self.n_qubits,
            topology=self.topology,
        )
        return [self.qml.expval(self.qml.PauliZ(wire)) for wire in range(self.n_qubits)]

    def forward(self, x):
        self._ensure_runtime()
        encoded = torch.tanh(self.input_proj(x)) * torch.pi
        stacked = self.q_layer(encoded).to(dtype=x.dtype, device=x.device)
        return self.output_proj(stacked)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["qml"] = None
        state["device"] = None
        state["ansatz"] = None
        state["q_layer"] = None
        return state


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


def parse_layer_selector(selector, num_layers):
    if selector == "":
        return set()

    selected = set()
    for raw_part in selector.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" not in part:
            selected.add(_parse_layer_index(part, num_layers))
            continue

        if part == "-":
            selected.update(range(num_layers))
            continue

        start_raw, end_raw = part.split("-", 1)
        start = 0 if start_raw == "" else _parse_layer_index(start_raw, num_layers)
        end = num_layers - 1 if end_raw == "" else _parse_layer_index(end_raw, num_layers)
        if start > end:
            raise ValueError(f"Invalid layer range '{part}' for {num_layers} layers")
        selected.update(range(start, end + 1))

    return selected


def _parse_layer_index(token, num_layers):
    index = int(token)
    if index < 0 or index >= num_layers:
        raise ValueError(f"Layer index {index} out of range for {num_layers} layers")
    return index
