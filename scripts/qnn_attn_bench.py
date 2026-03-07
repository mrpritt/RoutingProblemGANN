import time
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from VRP.VRP_Actor import Model


def main():
    cfg = {
        "ansatz_name": "pce",
        "n_qubits": 8,
        "n_layers": 4,
        "rotation": "RXRYRZ",
        "topology": "brickwall",
    }

    classical = Model(3, 128, 1, 16, 4)
    quantum = Model(
        3,
        128,
        1,
        16,
        4,
        encoder_attn_backend="qnn",
        encoder_attn_qnn_config=cfg,
        encoder_attn_qnn_layers={0},
    )

    x = torch.randn(4, 272)

    t0 = time.time()
    classical.encoder.convs1[0].attn(x)
    t1 = time.time()
    quantum.encoder.convs1[0].attn(x)
    t2 = time.time()

    print(
        {
            "classical_s": t1 - t0,
            "qnn_s": t2 - t1,
            "slowdown": (t2 - t1) / max(t1 - t0, 1e-9),
        }
    )


if __name__ == "__main__":
    main()
