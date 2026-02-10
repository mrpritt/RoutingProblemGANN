# Notes

## Installation
```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Training

```
CUDA_VISIBLE_DEVICES=1 python -m VRP.PPO_train.py --n_node 21
```

