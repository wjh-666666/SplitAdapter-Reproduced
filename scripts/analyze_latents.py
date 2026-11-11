"""Latent analysis (paper Fig. 3a, Fig. 5): collect rollouts in the training distribution and compute
  * t-SNE of the adaptation representation coloured by box mass (loaded samples)
  * world-model transition losses (object / robot target) on held-out data
  * linear-probe predictability gap: R^2(matched branch -> target) - R^2(cross branch -> target)

  python scripts/analyze_latents.py --ckpts logs/A/model_last.pt logs/B/model_last.pt --headless
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splitadapter.utils import parse, make_env, task_registry, ROOT  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scripts.eval_isaac import load_policy  # noqa: E402


def probe_r2(X, Y, Xte, Yte, lam=1e-2):
    X = np.concatenate([X, np.ones((len(X), 1))], 1)
    Xte = np.concatenate([Xte, np.ones((len(Xte), 1))], 1)
    W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
    P = Xte @ W
    ss_res = ((Yte - P) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-9
    return float(np.mean(1 - ss_res / ss_tot))


def collect(env, ac, steps):
    env.reset()
    obs = env.get_observations()
    buf = {k: [] for k in ['z_obj', 'z_dyn', 'z', 'm_eff', 'mass', 'loaded', 'robot_next', 'robot_cur', 'obj_delta',
                           'valid', 'obj_wm', 'dyn_wm']}
    for t in range(steps):
        with torch.no_grad():
            hist = env.get_adapter_inputs()
            mean, enc = ac.adapter(obs, hist)
            mass, l_soft, l_hard, l_valid = env.get_labels()
            o_t = obs[:, -123:].clone()
        obs, *_ = env.step(mean)
        rn, od, wv = env.get_wm_targets()
        if t < 50:
            continue
        for k in ['z_obj', 'z_dyn', 'z', 'm_eff']:
            if k in enc:
                buf[k].append(enc[k].cpu().numpy())
        buf['mass'].append(mass.cpu().numpy())
        buf['loaded'].append(l_hard.cpu().numpy())
        buf['robot_next'].append(rn.cpu().numpy())
        buf['robot_cur'].append(o_t[:, :79].cpu().numpy())
        buf['obj_delta'].append(od.cpu().numpy())
        buf['valid'].append((wv & l_valid).cpu().numpy())
        # per-sample world-model errors
        ad = ac.adapter
        with torch.no_grad():
            if ad.split:
                mh = enc['mass'].unsqueeze(-1)
                lh = enc['loaded_p'].unsqueeze(-1)
                me = enc['m_eff'].unsqueeze(-1)
                po = ad.obj_wm(torch.cat([o_t, mean, mh, lh, enc['z_obj']], -1))
                pr = o_t[:, :79] + ad.dyn_wm(torch.cat([o_t, mean, enc['z_dyn'], me], -1))
            else:
                inp = [o_t, mean, enc['z']] + ([enc['m_eff'].unsqueeze(-1)] if ad.use_mass else [])
                out = ad.uni_wm(torch.cat(inp, -1))
                pr, po = o_t[:, :79] + out[:, :79], out[:, 79:]
        buf['obj_wm'].append((po - od).abs().mean(-1).cpu().numpy())
        buf['dyn_wm'].append((pr - rn).abs().mean(-1).cpu().numpy())
    return {k: np.concatenate(v, 0) for k, v in buf.items() if len(v) > 0}


def main():
    ours, gym_args = parse([('--ckpts', dict(type=str, nargs='+', required=True)),
                            ('--tags', dict(type=str, nargs='+', default=None)),
                            ('--envs', dict(type=int, default=1024)),
                            ('--steps', dict(type=int, default=400)),
                            ('--out', dict(type=str, default=os.path.join(ROOT, 'results', 'latent_analysis')))])
    env_cfg, _ = task_registry.get_cfgs('carrybox_split')
    env_cfg.env.num_envs = ours.envs
    env_cfg.domain_rand.disturbance = False
    env, _ = make_env(gym_args, env_cfg)
    os.makedirs(ours.out, exist_ok=True)
    summary = {}
    for i, ck in enumerate(ours.ckpts):
        tag = ours.tags[i] if ours.tags else os.path.basename(os.path.dirname(ck))
        ac, variant = load_policy(ck, env, env.device)
        ac.eval()
        D = collect(env, ac, ours.steps)
        v = D['valid']
        idx = np.where(v)[0]
        rng = np.random.default_rng(0)
        rng.shuffle(idx)
        n = min(len(idx), 200000)
        tr, te = idx[:n // 2], idx[n // 2:n]
        obj_t = D['obj_delta']
        rob_t = D['robot_next'] - D['robot_cur']
        res = {'variant': variant, 'obj_wm_l1': float(D['obj_wm'][v].mean()), 'dyn_wm_l1': float(D['dyn_wm'][v].mean())}
        if 'z_obj' in D:
            zo, zd = D['z_obj'], D['z_dyn']
            r = {}
            for zn, Z in [('obj', zo), ('dyn', zd)]:
                for tn, Y in [('obj', obj_t), ('robot', rob_t)]:
                    r[f'{zn}->{tn}'] = probe_r2(Z[tr], Y[tr], Z[te], Y[te])
            res['probe_r2'] = r
            res['gap_object'] = r['obj->obj'] - r['dyn->obj']
            res['gap_robot'] = r['dyn->robot'] - r['obj->robot']
            rep = np.concatenate([zo, zd], 1)
        else:
            rep = D['z']
        # t-SNE on loaded samples (mass-dependent organisation)
        try:
            from sklearn.manifold import TSNE
            li = np.where(v & (D['loaded'] > 0.5))[0]
            li = rng.choice(li, size=min(3000, len(li)), replace=False)
            emb = TSNE(n_components=2, perplexity=40, init='pca', random_state=0).fit_transform(rep[li])
            np.savez(os.path.join(ours.out, f'tsne_{tag}.npz'), emb=emb, mass=D['mass'][li])
            # how well does the representation organise mass?  (linear probe R^2 on loaded samples)
            half = len(li) // 2
            res['mass_probe_r2_loaded'] = probe_r2(rep[li[:half]], D['mass'][li[:half], None], rep[li[half:]], D['mass'][li[half:], None])
        except Exception as e:  # noqa: BLE001
            print('t-SNE failed:', e)
        summary[tag] = res
        print(tag, json.dumps(res, indent=1))
    json.dump(summary, open(os.path.join(ours.out, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
