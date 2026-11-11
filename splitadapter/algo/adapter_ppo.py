"""PPO(+AMP) adapter training with the SplitAdapter auxiliary objectives.

The base actor is frozen.  Trainable parts:
  * RL group  (adaptive lr, KL-controlled): FiLM generators, action log-std, critic
  * Aux group (fixed lr 1e-3):              history normaliser/encoder, branches, heads, world models, adversaries
  * AMP discriminator (initialised from the base checkpoint, keeps training as in the base objective)
"""
import os
import time
import statistics
from collections import deque

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.modules import AMP
from splitadapter.modules.split_adapter import AdapterActorCritic


def class_to_dict(obj):
    if not hasattr(obj, '__dict__'):
        return obj
    out = {}
    for k in dir(obj):
        if k.startswith('_'):
            continue
        v = getattr(obj, k)
        if callable(v) and not isinstance(v, type):
            continue
        out[k] = class_to_dict(v) if isinstance(v, type) else v
    return out


class Storage:
    def __init__(self, N, T, dims, device):
        self.N, self.T, self.device = N, T, device
        f = lambda *s, dt=torch.float: torch.zeros(T, N, *s, device=device, dtype=dt)
        self.obs = f(dims['obs'])
        self.critic_obs = f(dims['critic'])
        # frames: 49 frames preceding the rollout + T rollout frames; histories are rebuilt as windows
        self.H = dims['hist_len']
        self.frames = torch.zeros(T + self.H - 1, N, dims['one_step'], device=device)
        self.hist_count = f(dt=torch.long)
        self.actions = f(dims['act'])
        self.values = f(1)
        self.logp = f(1)
        self.mu = f(dims['act'])
        self.sigma = f(dims['act'])
        self.rewards = f(1)
        self.dones = f(1)
        self.amp_obs = f(dims['amp'])
        self.robot_next = f(79)
        self.obj_delta = f(6)
        self.wm_valid = f(dt=torch.bool)
        self.mass = f()
        self.loaded_soft = f()
        self.loaded_hard = f()
        self.lbl_valid = f(dt=torch.bool)
        self.returns = f(1)
        self.adv = f(1)
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        adv = 0
        for t in reversed(range(self.T)):
            nv = last_values if t == self.T - 1 else self.values[t + 1]
            nt = 1.0 - self.dones[t]
            delta = self.rewards[t] + nt * gamma * nv - self.values[t]
            adv = delta + nt * gamma * lam * adv
            self.returns[t] = adv + self.values[t]
        self.adv = self.returns - self.values
        self.adv = (self.adv - self.adv.mean()) / (self.adv.std() + 1e-8)

    def minibatches(self, num_mb, epochs):
        B = self.N * self.T
        mb = B // num_mb
        flat = lambda x: x.reshape(B, *x.shape[2:])
        names = ['obs', 'critic_obs', 'hist_count', 'actions', 'values', 'logp', 'mu', 'sigma', 'returns', 'adv',
                 'amp_obs', 'robot_next', 'obj_delta', 'wm_valid', 'mass', 'loaded_soft', 'loaded_hard', 'lbl_valid']
        data = {n: flat(getattr(self, n)) for n in names}
        ar = torch.arange(self.H, device=self.device)
        for _ in range(epochs):
            perm = torch.randperm(B, device=self.device)
            for i in range(num_mb):
                idx = perm[i * mb:(i + 1) * mb]
                out = {n: v[idx] for n, v in data.items()}
                out['hist'] = self.window(idx // self.N, idx % self.N, out['hist_count'], ar)
                yield out

    def window(self, t, env, count, ar=None):
        if ar is None:
            ar = torch.arange(self.H, device=self.device)
        h = self.frames[t.unsqueeze(1) + ar.unsqueeze(0), env.unsqueeze(1)]  # (B, H, D)
        valid = ar.unsqueeze(0) >= (self.H - count.unsqueeze(1))
        return h * valid.unsqueeze(-1)


class AdapterRunner:
    def __init__(self, env, train_cfg, log_dir, device='cuda:0', base_ckpt_path=None):
        self.env = env
        self.device = device
        self.cfg = train_cfg
        self.alg_cfg = train_cfg.algorithm
        self.ad_cfg = train_cfg.adapter
        self.log_dir = log_dir
        self.base_ckpt_path = base_ckpt_path
        ck = torch.load(base_ckpt_path, map_location=device)
        H = env.cfg.adapter.history_length
        D = env.cfg.adapter.one_step_obs
        self.ac = AdapterActorCritic(ck['model_state_dict'], self.ad_cfg, env.num_privileged_obs, D, H).to(device)
        self.amp = AMP(train_cfg.amp.num_obs, train_cfg.amp.amp_coef, device=device).to(device)
        if 'amp_state_dict' in ck:
            self.amp.load_state_dict(ck['amp_state_dict'])
        ad = self.ac.adapter
        self.rl_params = ad.film_parameters() + [self.ac.log_std] + list(self.ac.critic.parameters())
        self.aux_params = ad.aux_parameters()
        self.lr = self.alg_cfg.learning_rate
        self.opt_rl = torch.optim.Adam(self.rl_params, lr=self.lr)
        self.opt_aux = torch.optim.Adam(self.aux_params, lr=self.alg_cfg.aux_learning_rate)
        self.opt_amp = torch.optim.Adam([{'params': self.amp.trunk.parameters(), 'weight_decay': 1e-3},
                                         {'params': self.amp.amp_linear.parameters(), 'weight_decay': 1e-1}], lr=1e-4)
        dims = dict(obs=env.num_obs, critic=env.num_privileged_obs, hist_len=H, one_step=D,
                    act=env.num_actions, amp=env.num_amp_obs)
        self.T = train_cfg.runner.num_steps_per_env
        self.storage = Storage(env.num_envs, self.T, dims, device)
        self.aux_steps = 0
        self.it = 0
        n_film = sum(p.numel() for p in ad.film_parameters())
        n_aux = sum(p.numel() for p in self.aux_params)
        print(f'[SplitAdapter] variant={ad.variant}  FiLM params={n_film}  aux params={n_aux}')

    # ------------------------------------------------------------------
    def save(self, path):
        tmp = path + '.tmp'   # atomic: a crash / pre-emption mid-write never leaves a corrupt checkpoint
        torch.save({'ac': self.ac.state_dict(), 'amp': self.amp.state_dict(),
                    'opt_rl': self.opt_rl.state_dict(), 'opt_aux': self.opt_aux.state_dict(),
                    'opt_amp': self.opt_amp.state_dict(), 'lr': self.lr, 'it': self.it,
                    'aux_steps': self.aux_steps, 'adapter_cfg': class_to_dict(self.ad_cfg),
                    'base_ckpt': self.base_ckpt_path}, tmp)
        os.replace(tmp, path)

    def load(self, path):
        d = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(d['ac'])
        self.amp.load_state_dict(d['amp'])
        self.opt_rl.load_state_dict(d['opt_rl'])
        self.opt_aux.load_state_dict(d['opt_aux'])
        self.opt_amp.load_state_dict(d['opt_amp'])
        self.lr, self.it, self.aux_steps = d['lr'], d['it'], d['aux_steps']

    # ------------------------------------------------------------------
    def learn(self, num_iterations):
        env, S, ac = self.env, self.storage, self.ac
        writer = SummaryWriter(self.log_dir)
        gamma, lam = self.alg_cfg.gamma, self.alg_cfg.lam
        env.reset()
        obs = env.get_observations().to(self.device)
        critic_obs = env.get_privileged_observations().to(self.device)
        rew_buf, len_buf, succ_buf = deque(maxlen=200), deque(maxlen=200), deque(maxlen=200)
        cur_rew = torch.zeros(env.num_envs, device=self.device)
        cur_len = torch.zeros(env.num_envs, device=self.device)
        ep_succ = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
        tot_steps, t_start = 0, time.time()
        for it in range(self.it, self.it + num_iterations):
            t0 = time.time()
            with torch.no_grad():
                for t in range(self.T):
                    hist = env.get_adapter_inputs()
                    mass, l_soft, l_hard, l_valid = env.get_labels()
                    ac.adapter.normalizer.update(hist[:, -1])
                    mean, _ = ac.act_dist(obs, hist)
                    a = ac.distribution.sample()
                    S.obs[t] = obs
                    S.critic_obs[t] = critic_obs
                    if t == 0:
                        S.frames[:S.H - 1] = hist[:, :-1].transpose(0, 1)
                    S.frames[S.H - 1 + t] = hist[:, -1]
                    S.hist_count[t] = env.hist_count
                    S.actions[t] = a
                    S.values[t] = ac.evaluate(critic_obs)
                    S.logp[t] = ac.distribution.log_prob(a).sum(-1, keepdim=True)
                    S.mu[t] = mean
                    S.sigma[t] = ac.distribution.stddev
                    S.mass[t], S.loaded_soft[t], S.loaded_hard[t], S.lbl_valid[t] = mass, l_soft, l_hard, l_valid
                    obs, priv, raw_rew, dones, infos, _, _, amp_obs = env.step(a)
                    critic_obs = priv
                    robot_next, obj_delta, wm_valid = env.get_wm_targets()
                    amp_rew = self.amp.predict_reward(amp_obs, normalizer=None).squeeze(1) * 0.5
                    rew = self.amp.combine_reward(amp_rew, raw_rew)
                    if 'time_outs' in infos:
                        rew = rew + gamma * S.values[t].squeeze(1) * infos['time_outs'].float()
                    S.rewards[t] = rew.unsqueeze(1)
                    S.dones[t] = dones.float().unsqueeze(1)
                    S.amp_obs[t] = amp_obs
                    S.robot_next[t], S.obj_delta[t], S.wm_valid[t] = robot_next, obj_delta, wm_valid
                    # book keeping
                    ep_succ |= env.success_buf
                    cur_rew += rew
                    cur_len += 1
                    done_ids = dones.nonzero(as_tuple=False).flatten()
                    if len(done_ids) > 0:
                        rew_buf.extend(cur_rew[done_ids].tolist())
                        len_buf.extend(cur_len[done_ids].tolist())
                        succ_buf.extend(ep_succ[done_ids].float().tolist())
                        cur_rew[done_ids] = 0
                        cur_len[done_ids] = 0
                        ep_succ[done_ids] = False
                S.compute_returns(ac.evaluate(critic_obs), gamma, lam)
            t_collect = time.time() - t0
            t1 = time.time()
            logs = self.update()
            t_learn = time.time() - t1
            self.it = it + 1
            tot_steps += self.T * env.num_envs
            # ---- logging
            fps = self.T * env.num_envs / (t_collect + t_learn)
            for k, v in logs.items():
                writer.add_scalar(f'Loss/{k}', v, it)
            writer.add_scalar('Policy/lr', self.lr, it)
            writer.add_scalar('Policy/mean_std', ac.log_std.exp().mean().item(), it)
            writer.add_scalar('Perf/fps', fps, it)
            if len(rew_buf) > 0:
                writer.add_scalar('Train/mean_reward', statistics.mean(rew_buf), it)
                writer.add_scalar('Train/mean_episode_length', statistics.mean(len_buf), it)
                writer.add_scalar('Train/episode_success', statistics.mean(succ_buf), it)
            if it % 10 == 0:
                msg = (f'it {it:5d} | fps {fps:7.0f} | col {t_collect:5.1f}s learn {t_learn:5.1f}s | '
                       f'rew {statistics.mean(rew_buf) if rew_buf else 0:7.2f} len {statistics.mean(len_buf) if len_buf else 0:6.1f} '
                       f'succ {statistics.mean(succ_buf) if succ_buf else 0:.3f} | lr {self.lr:.1e} std {ac.log_std.exp().mean():.3f} | '
                       + ' '.join(f'{k} {v:.4f}' for k, v in logs.items()))
                print(msg, flush=True)
            if it % self.cfg.runner.save_interval == 0:
                self.save(os.path.join(self.log_dir, f'model_{it}.pt'))
        self.save(os.path.join(self.log_dir, f'model_{self.it - 1}.pt'))
        self.save(os.path.join(self.log_dir, 'model_last.pt'))
        writer.close()

    # ------------------------------------------------------------------
    def update(self):
        a_cfg, c = self.alg_cfg, self.ad_cfg
        ac, S = self.ac, self.storage
        sums = {}
        n = 0
        for b in S.minibatches(a_cfg.num_mini_batches, a_cfg.num_learning_epochs):
            hist = b['hist']
            mean, enc = ac.act_dist(b['obs'], hist)
            logp = ac.distribution.log_prob(b['actions']).sum(-1)
            sigma = ac.distribution.stddev
            entropy = ac.distribution.entropy().sum(-1).mean()
            value = ac.evaluate(b['critic_obs'])
            # adaptive lr from KL
            with torch.no_grad():
                kl = torch.sum(torch.log(sigma / b['sigma'] + 1e-5) +
                               (b['sigma'].pow(2) + (b['mu'] - mean).pow(2)) / (2 * sigma.pow(2)) - 0.5, -1).mean()
                if a_cfg.schedule == 'adaptive':
                    if kl > a_cfg.desired_kl * 2.0:
                        self.lr = max(1e-6, self.lr / 1.5)
                    elif 0.0 < kl < a_cfg.desired_kl / 2.0:
                        self.lr = min(1e-3, self.lr * 1.5)
                    for g in self.opt_rl.param_groups:
                        g['lr'] = self.lr
            ratio = torch.exp(logp - b['logp'].squeeze(-1))
            adv = b['adv'].squeeze(-1)
            surr = torch.max(-adv * ratio, -adv * ratio.clamp(1 - a_cfg.clip_param, 1 + a_cfg.clip_param)).mean()
            v_clip = b['values'] + (value - b['values']).clamp(-a_cfg.clip_param, a_cfg.clip_param)
            v_loss = torch.max((value - b['returns']).pow(2), (v_clip - b['returns']).pow(2)).mean()
            rl_loss = surr + a_cfg.value_loss_coef * v_loss - a_cfg.entropy_coef * entropy
            # ---- auxiliary objectives
            o_t = b['obs'][:, -123:]
            grl_active = self.aux_steps >= c.grl_warmup_steps
            L, alog = ac.adapter.aux_losses(enc, o_t, b['actions'], b['robot_next'], b['obj_delta'], b['wm_valid'],
                                            b['mass'], b['loaded_soft'], b['loaded_hard'], b['lbl_valid'], grl_active)
            loss = rl_loss + sum(L.values())
            self.opt_rl.zero_grad()
            self.opt_aux.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.rl_params, a_cfg.max_grad_norm)
            nn.utils.clip_grad_norm_(self.aux_params, a_cfg.max_grad_norm)
            self.opt_rl.step()
            self.opt_aux.step()
            self.aux_steps += 1
            # ---- AMP discriminator
            expert = self.env.motionlib.get_expert_obs(batch_size=b['amp_obs'].shape[0]).to(self.device)
            amp_loss, exp_l, pol_l = self.amp.compute_loss(b['amp_obs'], expert)
            self.opt_amp.zero_grad()
            amp_loss.backward()
            self.opt_amp.step()
            # ---- logs
            vals = dict(surrogate=surr.item(), value=v_loss.item(), kl=kl.item(), amp=amp_loss.item(),
                        amp_expert=exp_l.item(), amp_policy=pol_l.item())
            vals.update({f'aux_{k}': v.item() for k, v in L.items()})
            vals.update({k: float(v) for k, v in alog.items()})
            for k, v in vals.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
        return {k: v / n for k, v in sums.items()}
