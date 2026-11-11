"""Shared helpers: task registration, argument parsing, environment construction."""
import os
import sys
import argparse

import isaacgym  # noqa: F401  (must be imported before torch)
import legged_gym.envs  # noqa: F401  registers PhysHSI tasks (must precede legged_gym.utils)
from legged_gym.utils import get_args, task_registry

from splitadapter.envs.carrybox_split import SplitCarryBox
from splitadapter.envs.carrybox_split_config import SplitCarryBoxCfg, SplitCarryBoxCfgPPO, BaseFinetuneCfgPPO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHYSHSI = os.path.join(ROOT, 'third_party', 'PhysHSI')
BASE_CKPT = os.path.join(PHYSHSI, 'legged_gym', 'resources', 'ckpt', 'carrybox.pt')

if 'carrybox_split' not in task_registry.task_classes:
    task_registry.register('carrybox_split', SplitCarryBox, SplitCarryBoxCfg(), SplitCarryBoxCfgPPO())
    task_registry.register('carrybox_split_base', SplitCarryBox, SplitCarryBoxCfg(), BaseFinetuneCfgPPO())


def parse(extra):
    """Parse our own flags; the rest goes to the Isaac Gym / legged_gym argument parser."""
    p = argparse.ArgumentParser(add_help=False)
    for name, kw in extra:
        p.add_argument(name, **kw)
    ours, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    gym_args = get_args()
    return ours, gym_args


def make_env(gym_args, env_cfg, name='carrybox_split'):
    env, env_cfg = task_registry.make_env(name=name, args=gym_args, env_cfg=env_cfg)
    return env, env_cfg
