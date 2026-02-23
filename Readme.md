# Notes

## Installation
```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Training

```
CUDA_VISIBLE_DEVICES=1 python -m VRP.PPO_train.py
```
# --n_node 21 is ineffective

## Evaluation

Need folder "./trained".

### Using a self-trained checkpoint

`PPO_train.py` saves `Actor_critic.state_dict()`, whose keys have an `actor.`
prefix incompatible with the plain `Model` expected by `test_vrp.py`.
Convert before placing the checkpoint in `trained/<n>/actor.pt`:

```
python scripts/fix_actor_keys.py vrp-21-GAT/20201125/99/actor.pt trained/21.mine/actor.pt
```

```
CUDA_VISIBLE_DEVICES=1 python -m VRP.test_vrp.py
```
