"""Train a SplitAdapter (or a baseline / ablation) on top of the frozen carry-box base policy.

  python scripts/train_adapter.py --variant split --base <base.pt> --seed 1 --headless
  variants: split | anyadapter | no_split | no_hier_film | no_grl
  --resume_adapter     continue from a checkpoint (use the same --log_dir)
"""
import os
import sys
import json
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splitadapter.utils import parse, make_env, task_registry, BASE_CKPT, ROOT  # noqa: E402

from splitadapter.algo.adapter_ppo import AdapterRunner, class_to_dict  # noqa: E402


def main():
    ours, gym_args = parse([
        ('--variant', dict(type=str, default='split')),
        ('--iters', dict(type=int, default=None)),
        ('--name', dict(type=str, default=None)),
        ('--resume_adapter', dict(type=str, default=None)),
        ('--envs', dict(type=int, default=None)),
        ('--base', dict(type=str, default=BASE_CKPT)),
        ('--log_dir', dict(type=str, default=None)),
        ('--save_interval', dict(type=int, default=10)),
    ])
    env_cfg, train_cfg = task_registry.get_cfgs('carrybox_split')
    if ours.envs is not None:
        env_cfg.env.num_envs = ours.envs
    train_cfg.adapter.variant = ours.variant
    train_cfg.runner.save_interval = ours.save_interval
    iters = ours.iters or train_cfg.runner.max_iterations
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    name = ours.name or ours.variant
    log_dir = ours.log_dir or os.path.join(ROOT, 'logs', f'{stamp}_{name}')
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump({'env': class_to_dict(env_cfg), 'train': class_to_dict(train_cfg), 'iters': iters, 'base': ours.base}, f, indent=1, default=str)

    env, _ = make_env(gym_args, env_cfg)
    runner = AdapterRunner(env, train_cfg, log_dir, device=env.device, base_ckpt_path=ours.base)
    if ours.resume_adapter:
        runner.load(ours.resume_adapter)
        print(f'resumed from {ours.resume_adapter} at iteration {runner.it}', flush=True)
    print('log dir:', log_dir, flush=True)
    if runner.it < iters:
        runner.learn(iters - runner.it)


if __name__ == '__main__':
    main()
