import pennylane as qml


def parse_rot_sequence(R_type: str):
    if not isinstance(R_type, str) or len(R_type.strip()) == 0:
        raise ValueError("R_type must be a non-empty string like 'RY' or 'RXRYRZ'.")

    s = R_type.upper().replace(" ", "")
    if len(s) % 2 != 0:
        raise ValueError(f"R_type must be concatenation of 'RX','RY','RZ'. Got: {R_type}")

    seq = [s[i:i + 2] for i in range(0, len(s), 2)]
    allowed = {"RX", "RY", "RZ"}
    if any(tok not in allowed for tok in seq):
        raise ValueError(f"R_type tokens must be in {allowed}. Got: {seq}")
    return seq


def apply_rot_sequence(params_1q, wire, rot_seq):
    if len(params_1q) != len(rot_seq):
        raise ValueError(
            f"params_1q must have length {len(rot_seq)} for rot_seq={rot_seq}, got {len(params_1q)}"
        )

    for theta, gate in zip(params_1q, rot_seq):
        if gate == "RX":
            qml.RX(theta, wires=wire)
        elif gate == "RY":
            qml.RY(theta, wires=wire)
        else:
            qml.RZ(theta, wires=wire)


def ansatz(
    params,
    R_type="RXRYRZ",
    n_qubits=8,
    topology="brickwall",
):
    rot_seq = parse_rot_sequence(R_type)
    k = len(rot_seq)

    if len(params) == 0:
        raise ValueError("params must have L>=1 layers.")
    if any(len(params[ell]) != n_qubits for ell in range(len(params))):
        raise ValueError(f"Each layer must have first dim = {n_qubits}.")
    for ell in range(len(params)):
        for w in range(n_qubits):
            if len(params[ell][w]) != k:
                raise ValueError(f"params[{ell}][{w}] must have length {k} (R_type={R_type}).")

    def _ent(u, v):
        qml.CNOT(wires=[u, v])

    topo = topology.lower().replace("-", "").replace("_", "")

    for ell in range(len(params)):
        for w in range(n_qubits):
            apply_rot_sequence(params[ell][w], wire=w, rot_seq=rot_seq)

        if topo == "brickwall":
            for u in range(0, n_qubits - 1, 2):
                _ent(u, u + 1)
            for u in range(1, n_qubits - 1, 2):
                _ent(u, u + 1)
        elif topo == "chain":
            for u in range(n_qubits - 1):
                _ent(u, u + 1)
        elif topo == "lambda":
            for d in range(n_qubits - 1):
                left = d
                right = n_qubits - 2 - d
                if left <= right:
                    _ent(left, left + 1)
                if right > left:
                    _ent(right, right + 1)
                if left >= right:
                    break
        else:
            raise ValueError("topology must be 'brickwall', 'chain', or 'lambda'.")
