"""Evaluate base policy / adapters on the paper's grid: mass {2,4,6} kg x pickup/placement height {0,30,60} cm.

  python scripts/eval_isaac.py --ckpt base --trials 10 --headless
  python scripts/eval_isaac.py --ckpt logs/<run>/model_last.pt --trials 10 --headless
Writes a JSON with per-condition Lift-up / Full-task success counts.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splitadapter.utils import parse, make_env, task_registry, BASE_CKPT, ROOT  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from splitadapter.modules.split_adapter import AdapterActorCritic  # noqa: E402

MASSES = [2.0, 4.0, 6.0]
HEIGHTS = [0.0, 0.3, 0.6]


class Dict2Obj:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def load_policy(ckpt, env, device, base_path=None):
    if ckpt != 'base' and base_path is None:
        base_path = torch.load(ckpt, map_location='cpu').get('base_ckpt', BASE_CKPT)
    base = torch.load(base_path or BASE_CKPT, map_location=device)['model_state_dict']
    if ckpt == 'base':
        cfg = task_registry.get_cfgs('carrybox_split')[1].adapter
        ac = AdapterActorCritic(base, cfg, env.num_privileged_obs).to(device)  # zero-init FiLM == base policy
        return ac, 'base'
    d = torch.load(ckpt, map_location=device)
    cfg = Dict2Obj(d['adapter_cfg'])
    ac = AdapterActorCritic(base, cfg, env.num_privileged_obs).to(device)
    ac.load_state_dict(d['ac'])
    return ac, cfg.variant


def main():
    ours, gym_args = parse([
        ('--ckpt', dict(type=str, default='base')),
        ('--trials', dict(type=int, default=10)),
        ('--seed', dict(type=int, default=0)),
        ('--out', dict(type=str, default=None)),
        ('--no_dr', dict(action='store_true')),
        ('--trace_envs', dict(type=int, default=0)),
        ('--base', dict(type=str, default=None)),
        ('--tag', dict(type=str, default=None)),
        ('--force', dict(action='store_true')),
    ])
    if ours.tag and not ours.force:
        out0 = ours.out or os.path.join(ROOT, 'results', f"isaac_{ours.tag}_s{ours.seed}.json")
        if os.path.exists(out0):
            print('skip (exists):', out0)
            return
    env_cfg, _ = task_registry.get_cfgs('carrybox_split')
    conds = [(m, h) for m in MASSES for h in HEIGHTS]
    masses, heights, cond_id = [], [], []
    for ci, (m, h) in enumerate(conds):
        masses += [m] * ours.trials
        heights += [h] * ours.trials
        cond_id += [ci] * ours.trials
    N = len(masses)
    env_cfg.env.num_envs = N
    env_cfg.env.test = True
    env_cfg.env.episode_length_s = 20
    env_cfg.evaluation.enabled = True
    env_cfg.evaluation.masses = masses
    env_cfg.evaluation.pickup_heights = heights
    env_cfg.evaluation.place_heights = heights
    env_cfg.evaluation.seed = ours.seed
    env_cfg.seed = ours.seed
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.push_robots = False
    if ours.no_dr:
        dr = env_cfg.domain_rand
        for k in ['randomize_actuation_offset', 'randomize_motor_strength', 'randomize_com_displacement',
                  'randomize_link_mass', 'randomize_kp', 'randomize_kd', 'delay', 'randomize_friction',
                  'randomize_restitution', 'randomize_initial_joint_pos']:
            setattr(dr, k, False)
    env, _ = make_env(gym_args, env_cfg)
    device = env.device
    ac, variant = load_policy(ours.ckpt, env, device, ours.base)
    if ours.tag:
        variant = ours.tag
    ac.eval()

    env.reset()
    obs = env.get_observations()
    T = int(env.max_episode_length) + 5
    traces = {'mass': [], 'loaded_p': [], 'gt_loaded': [], 'gt_mass': None}
    for t in range(T):
        with torch.no_grad():
            hist = env.get_adapter_inputs()
            mean, enc = ac.adapter(obs, hist)
            if ours.trace_envs > 0 and 'mass' in enc:
                k = ours.trace_envs
                traces['mass'].append(enc['mass'][:k].cpu().numpy().tolist())
                traces['loaded_p'].append(enc['loaded_p'][:k].cpu().numpy().tolist())
                traces['gt_loaded'].append(env.lbl_loaded_soft[:k].cpu().numpy().tolist())
        obs, _, _, _, _, _, _, _ = env.step(mean)
        if bool(env.ev_finished.all()):
            break
    lifted = env.ev_lifted.cpu().numpy()
    succ = env.ev_success.cpu().numpy()
    cond_id = np.array(cond_id)
    res = {'ckpt': ours.ckpt, 'variant': variant, 'trials': ours.trials, 'seed': ours.seed, 'conditions': []}
    for ci, (m, h) in enumerate(conds):
        sel = cond_id == ci
        res['conditions'].append({'mass': m, 'height': h, 'full': int(succ[sel].sum()), 'lift': int(lifted[sel].sum())})
    res['full_total'] = int(succ.sum())
    res['lift_total'] = int(lifted.sum())
    res['n'] = N
    if ours.trace_envs > 0:
        traces['gt_mass'] = masses[:ours.trace_envs]
        res['traces'] = traces
    print(f"== {variant}  ({ours.ckpt})")
    print('mass  h    full  lift')
    for c in res['conditions']:
        print(f"{c['mass']:.0f}kg {c['height']*100:3.0f}cm  {c['full']:3d}/{ours.trials}  {c['lift']:3d}/{ours.trials}")
    print(f"Full-task {res['full_total']}/{N}   Lift-up {res['lift_total']}/{N}")
    out = ours.out or os.path.join(ROOT, 'results', f"isaac_{variant}_s{ours.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(res, f)
    print('saved', out)


if __name__ == '__main__':
    main()
