"""Checks that Storage.window() reproduces env.adapt_hist exactly across resets."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splitadapter.utils import parse, make_env, task_registry, BASE_CKPT
import torch
from splitadapter.algo.adapter_ppo import AdapterRunner

ours, gym_args = parse([])
env_cfg, train_cfg = task_registry.get_cfgs('carrybox_split')
env_cfg.env.num_envs = 64
train_cfg.runner.num_steps_per_env = 120
env, _ = make_env(gym_args, env_cfg)
r = AdapterRunner(env, train_cfg, '/tmp/sa_test', device=env.device, base_ckpt_path=BASE_CKPT)
S = r.storage
env.reset()
obs = env.get_observations()
true_h, max_err, n_resets = [], 0.0, 0
with torch.no_grad():
    for t in range(S.T):
        hist = env.get_adapter_inputs().clone()
        if t == 0:
            S.frames[:S.H - 1] = hist[:, :-1].transpose(0, 1)
        S.frames[S.H - 1 + t] = hist[:, -1]
        S.hist_count[t] = env.hist_count
        true_h.append(hist)
        a, _ = r.ac.act_dist(obs, hist)
        obs, _, _, dones, *_ = env.step(a)
        n_resets += int(dones.sum())
    for t in range(S.T):
        envs = torch.arange(env.num_envs, device=env.device)
        w = S.window(torch.full_like(envs, t), envs, S.hist_count[t])
        max_err = max(max_err, (w - true_h[t]).abs().max().item())
print(f'resets during test: {n_resets}   max |window - env hist| = {max_err:.2e}')
assert max_err < 1e-6
print('OK')
