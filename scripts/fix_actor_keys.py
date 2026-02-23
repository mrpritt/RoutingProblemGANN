"""
Convert a checkpoint saved from Actor_critic (PPO training) to one loadable
by the plain Model used in test_vrp.py.

Training saves old_polic.state_dict() which is an Actor_critic, so keys have
the form  actor.<layer>  and  critic.<layer>.
test_vrp.py loads into Model which expects  <layer>  directly.

Usage:
    python scripts/fix_actor_keys.py <src.pt> <dst.pt>

Example:
    python scripts/fix_actor_keys.py vrp-21-GAT/20201125/99/actor.pt trained/21.mine/actor.pt
"""
import sys
import collections
import torch


def fix_keys(src_path: str, dst_path: str) -> None:
    src = torch.load(src_path, map_location='cpu')
    fixed = collections.OrderedDict(
        (k[len('actor.'):], v)
        for k, v in src.items()
        if k.startswith('actor.')
    )
    torch.save(fixed, dst_path)
    print(f"Saved {len(fixed)} keys to {dst_path}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    fix_keys(sys.argv[1], sys.argv[2])
