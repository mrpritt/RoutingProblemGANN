# Notes

## Installation
```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Training

The paper reports results using the Rollout algorithm (Section 4.2), noting PPO performs
slightly less well. Rollout checkpoints are directly compatible with the evaluator (no key
conversion needed). `n_nodes` is hardcoded to 21 in both trainers; there is no CLI argument.

### Rollout (recommended)

```
CUDA_VISIBLE_DEVICES=1 python -m VRP.VRP_Rollout_train
```

Saves each epoch to `Vrp-21-GAT/rollout/{epoch}/actor.pt`.

### PPO

```
CUDA_VISIBLE_DEVICES=1 python -m VRP.PPO_train
```

Saves each epoch to `vrp-21-GAT/20201125/{epoch}/actor.pt`.
PPO checkpoints need key conversion before use with the evaluator (see below).

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
