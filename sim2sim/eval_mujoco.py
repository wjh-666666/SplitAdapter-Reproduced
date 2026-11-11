"""MuJoCo sim-to-sim evaluation (paper Table 1 protocol).

  python sim2sim/eval_mujoco.py --ckpt base --base <base.pt> --trials 10 --workers 18
  python sim2sim/eval_mujoco.py --ckpt logs/<run>/model_last.pt --trials 10
Mass {2,4,6} kg x pickup/placement height {0,30,60} cm, Lift-up / Full-task success.
"""
import os
import sys
import json
import time
import argparse
import multiprocessing as mp

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DEFAULT_BASE = os.path.join(ROOT, 'third_party', 'PhysHSI', 'legged_gym', 'resources', 'ckpt', 'carrybox.pt')
MASSES = [2.0, 4.0, 6.0]
HEIGHTS = [0.0, 0.3, 0.6]


class Obj:
    def __init__(self, d):
        self.__dict__.update(d)


DEFAULT_ADAPTER_CFG = dict(variant='split', latent_obj=16, latent_dyn=16, latent_unified=32, film_hidden=128,
                           grl_coef=0.1, grl_warmup_steps=10000)


def load_policy(ckpt, base_path):
    import torch
    from splitadapter.modules.split_adapter import AdapterActorCritic
    if ckpt != 'base':
        d = torch.load(ckpt, map_location='cpu')
        base_path = base_path or d.get('base_ckpt', DEFAULT_BASE)
        cfg = Obj(d['adapter_cfg'])
    else:
        d, cfg = None, Obj(DEFAULT_ADAPTER_CFG)
    base = torch.load(base_path or DEFAULT_BASE, map_location='cpu')['model_state_dict']
    ac = AdapterActorCritic(base, cfg, 126)
    if d is not None:
        ac.load_state_dict(d['ac'])
    ac.eval()
    return ac, (cfg.variant if ckpt != 'base' else 'base')


def worker(job):
    import torch
    torch.set_num_threads(1)
    from sim2sim.mujoco_carrybox import MujocoCarryBox
    ckpt, base_path, items, opts = job
    ac, _ = load_policy(ckpt, base_path)
    env = MujocoCarryBox(box_size=opts['box_size'], add_noise=not opts['no_noise'],
                         motor_strength=opts['motor_strength'])
    out = []
    for (ci, m, h, trial) in items:
        seed = 1000 * ci + trial + 100000 * opts['seed']
        obs, hist = env.reset(m, h, h, seed)
        traj, est = [], []
        done = False
        while not done:
            with torch.no_grad():
                o = torch.from_numpy(obs).unsqueeze(0)
                hh = torch.from_numpy(hist).unsqueeze(0)
                act, enc = ac.adapter(o, hh)
            if 'mass' in enc and opts['trace']:
                est.append((float(enc['mass'][0]), float(enc['loaded_p'][0]), env.loaded_soft))
            obs, hist, done, info = env.step(act[0].numpy())
            if opts['trace']:
                traj.append(info['box_pos'][:2].tolist() + info['robot_pos'][:2].tolist())
        rec = dict(ci=ci, mass=m, height=h, trial=trial, lift=bool(env.lifted), full=bool(env.ever_success),
                   fell=bool(env.fell), t_success=env.success_time, goal=env.goal.tolist())
        if opts['trace']:
            rec['traj'] = traj[::5]
            rec['est'] = est[::2]
        out.append(rec)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='base')
    p.add_argument('--base', default=None)
    p.add_argument('--trials', type=int, default=10)
    p.add_argument('--workers', type=int, default=18)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tag', default=None)
    p.add_argument('--out', default=None)
    p.add_argument('--trace', action='store_true')
    p.add_argument('--no_noise', action='store_true')
    p.add_argument('--motor_strength', type=float, default=1.0)
    p.add_argument('--box_size', type=float, nargs=3, default=[0.30, 0.43, 0.30])
    p.add_argument('--force', action='store_true', help='re-run even if the output file exists')
    a = p.parse_args()
    tag0 = a.tag or ('base' if a.ckpt == 'base' else os.path.basename(os.path.dirname(a.ckpt)))
    out0 = a.out or os.path.join(ROOT, 'results', f'mujoco_{tag0}_s{a.seed}.json')
    if os.path.exists(out0) and not a.force:
        print(f'skip (exists): {out0}')
        return
    opts = dict(box_size=a.box_size, no_noise=a.no_noise, motor_strength=a.motor_strength, seed=a.seed, trace=a.trace)
    items = [(ci, m, h, t) for ci, (m, h) in enumerate([(m, h) for m in MASSES for h in HEIGHTS]) for t in range(a.trials)]
    chunks = [items[i::a.workers] for i in range(a.workers)]
    jobs = [(a.ckpt, a.base, c, opts) for c in chunks if c]
    t0 = time.time()
    with mp.get_context('spawn').Pool(len(jobs)) as pool:
        recs = [r for rs in pool.map(worker, jobs) for r in rs]
    recs.sort(key=lambda r: (r['ci'], r['trial']))
    tag = a.tag or ('base' if a.ckpt == 'base' else os.path.basename(os.path.dirname(a.ckpt)))
    res = dict(ckpt=a.ckpt, base=a.base, tag=tag, trials=a.trials, seed=a.seed, records=recs, conditions=[])
    print(f'== MuJoCo sim2sim: {tag}   ({time.time() - t0:.0f}s)')
    print('mass  h    full  lift')
    for ci, (m, h) in enumerate([(m, h) for m in MASSES for h in HEIGHTS]):
        rs = [r for r in recs if r['ci'] == ci]
        full, lift = sum(r['full'] for r in rs), sum(r['lift'] for r in rs)
        res['conditions'].append(dict(mass=m, height=h, full=full, lift=lift, n=len(rs)))
        print(f'{m:.0f}kg {h * 100:3.0f}cm  {full:3d}/{len(rs)}  {lift:3d}/{len(rs)}')
    res['full_total'] = sum(c['full'] for c in res['conditions'])
    res['lift_total'] = sum(c['lift'] for c in res['conditions'])
    print(f"Full-task {res['full_total']}/{len(recs)}   Lift-up {res['lift_total']}/{len(recs)}")
    out = a.out or os.path.join(ROOT, 'results', f'mujoco_{tag}_s{a.seed}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'))
    print('saved', out)


if __name__ == '__main__':
    main()
