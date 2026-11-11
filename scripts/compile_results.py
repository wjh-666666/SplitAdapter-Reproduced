"""Compile evaluation JSONs into Table-1-style markdown tables and figures.

  python scripts/compile_results.py
Reads results/mujoco_<tag>_s<seed>.json, results/isaac_<tag>_s0.json, results/latent_analysis/*.
Writes results/REPORT_TABLES.md and results/figs/*.png
"""
import os
import glob
import json
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(RES, 'figs')
ORDER = ['physhsi_release', 'base_ft', 'anyadapter@500', 'anyadapter', 'no_split', 'no_hier_film', 'no_grl', 'split']
NAMES = {'base_candidates': 'Base fine-tune candidates (3 seeds)', 'physhsi_release': 'PhysHSI (released ckpt)',
         'base_ft': 'Base (PhysHSI, fine-tuned)', 'anyadapter@500': 'AnyAdapter-style (it 500)',
         'anyadapter': 'AnyAdapter-style (it 1500)', 'no_split': 'w/o split latent',
         'no_hier_film': 'w/o hierarchical FiLM', 'no_grl': 'w/o GRL', 'split': 'SplitAdapter'}
# methods shown in the bar figures (<= 8 series: the validated categorical palette has 8 slots)
FIG_METHODS = ['physhsi_release', 'base_ft', 'anyadapter@500', 'no_split', 'no_hier_film', 'no_grl', 'split']
# validated categorical palette (fixed order, light surface); text stays in ink colors
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, SURF, GRID = '#0b0b0b', '#52514e', '#fcfcfb', '#e4e3df'


def style(ax):
    ax.set_facecolor(SURF)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def load(prefix, folder=None):
    """{method: {(train_seed, eval_seed): result}}; tags like 'split_s2' -> method 'split', train seed 2."""
    out = defaultdict(dict)
    folder = folder or RES
    for f in glob.glob(os.path.join(folder, f'{prefix}_*_s*.json')):
        d = json.load(open(f))
        tag = (d.get('tag') or d.get('variant'))
        m = re.match(r'^(.*)_s(\d+)$', tag)
        method, ts = (m.group(1), int(m.group(2))) if m else (tag, 1)
        if method == 'base_ft' and m:   # the three fine-tune candidates (selection by Isaac eval), not the selected base
            method = 'base_candidates'
        out[method][(ts, d['seed'])] = d
    return out


def table(data, title, protocol=False):
    lines = [f'### {title}', '',
             '| Method | 2kg 0 | 2kg 30 | 2kg 60 | 4kg 0 | 4kg 30 | 4kg 60 | 6kg 0 | 6kg 30 | 6kg 60 | Full-task | Lift-up |',
             '|---|' + '---|' * 11]
    for tag in [t for t in ORDER if t in data] + [t for t in data if t not in ORDER]:
        runs = data[tag]
        ss = [k for k in sorted(runs) if (not protocol or k == (1, 0))]
        if not ss:
            continue
        conds = np.array([[c['full'] for c in runs[s]['conditions']] for s in ss])
        lifts = np.array([[c['lift'] for c in runs[s]['conditions']] for s in ss])
        n = runs[ss[0]]['conditions'][0].get('n', runs[ss[0]]['trials'])
        if len(ss) == 1:
            cells = [str(int(x)) for x in conds[0]]
            ft, lt = f'{conds.sum()}/{9 * n}', f'{lifts.sum()}/{9 * n}'
        else:
            cells = [f'{x:.1f}' for x in conds.mean(0)]
            ft = f'{conds.sum(1).mean():.1f}±{conds.sum(1).std():.1f}/{9 * n}'
            lt = f'{lifts.sum(1).mean():.1f}/{9 * n}'
        lines.append(f'| {NAMES.get(tag, tag)} | ' + ' | '.join(cells) + f' | {ft} | {lt} |')
    return '\n'.join(lines) + '\n'


def fig_success(data, fname, title, methods=None):
    tags = [t for t in (methods or ORDER) if t in data][:len(COLORS)]
    if not tags:
        return
    full, full6, lift = [], [], []
    for t in tags:
        rs = list(data[t].values())
        full.append(np.mean([sum(c['full'] for c in r['conditions']) for r in rs]) / 90 * 100)
        full6.append(np.mean([sum(c['full'] for c in r['conditions'] if c['mass'] == 6.0) for r in rs]) / 30 * 100)
        lift.append(np.mean([sum(c['lift'] for c in r['conditions']) for r in rs]) / 90 * 100)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURF)
    for ax, vals, sub in zip(axes, [lift, full, full6], ['Lift-up success (all)', 'Full-task success (all)',
                                                          'Full-task success (6 kg, OOD)']):
        style(ax)
        x = np.arange(len(tags))
        bars = ax.bar(x, vals, width=0.72, color=COLORS[:len(tags)], edgecolor=SURF, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.0f}', ha='center', va='bottom', fontsize=8.5, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([NAMES[t] for t in tags], rotation=35, ha='right', fontsize=8, color=INK2)
        ax.set_ylim(0, 108)
        ax.set_title(sub, fontsize=10.5, color=INK, loc='left')
        ax.set_ylabel('% of trials', fontsize=9, color=INK2)
    fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=150, facecolor=SURF)
    plt.close(fig)


def fig_mass_trace(fname):
    f = os.path.join(RES, 'mujoco_split_trace.json')
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    picks = {}
    for r in d['records']:
        if r['height'] == 0.0 and r['full'] and r['mass'] not in picks and r.get('est'):
            picks[r['mass']] = r
    if not picks:
        return
    fig, axes = plt.subplots(2, len(picks), figsize=(4.4 * len(picks), 4.8), facecolor=SURF, sharex=False, squeeze=False)
    for j, (m, r) in enumerate(sorted(picks.items())):
        est = np.array(r['est'])
        t = np.arange(len(est)) * 0.04
        ax = axes[0, j]
        style(ax)
        ax.plot(t, est[:, 0] * est[:, 1], color=COLORS[0], lw=2, label='estimated m_eff = p̂·m̂')
        ax.axhline(m, color=INK2, lw=1.2, ls='--', label=f'GT mass {m:.0f} kg')
        ax.set_ylim(0, 7.5)
        ax.set_title(f'{m:.0f} kg payload', fontsize=10.5, color=INK, loc='left')
        ax.set_ylabel('mass (kg)', fontsize=9, color=INK2)
        ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
        ax = axes[1, j]
        style(ax)
        ax.plot(t, est[:, 1], color=COLORS[0], lw=2, label='predicted loaded prob. p̂')
        ax.plot(t, est[:, 2], color=COLORS[1], lw=2, label='GT loaded (soft)')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('time (s)', fontsize=9, color=INK2)
        ax.set_ylabel('loaded prob.', fontsize=9, color=INK2)
        ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    fig.suptitle('Online mass / loaded-state estimation (MuJoCo sim2sim, 0 cm pickup)', fontsize=12, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=150, facecolor=SURF)
    plt.close(fig)


def fig_traj(fname):
    files = [(t, os.path.join(RES, f'mujoco_{t}_trace.json')) for t in ['base_ft', 'split']]
    files = [(t, f) for t, f in files if os.path.exists(f)]
    if len(files) < 2:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=SURF)
    for j, h in enumerate([0.0, 0.3, 0.6]):
        ax = axes[j]
        style(ax)
        ax.yaxis.grid(False)
        for k, (t, f) in enumerate(files):
            d = json.load(open(f))
            first = True
            for r in d['records']:
                if r['mass'] != 6.0 or r['height'] != h:
                    continue
                tr = np.array(r['traj'])
                ax.plot(tr[:, 0], tr[:, 1], color=COLORS[k], lw=1.6, alpha=0.85,
                        label=NAMES[t] if first else None)
                first = False
                if not r['full']:
                    ax.plot(tr[-1, 0], tr[-1, 1], 'x', color=COLORS[k], ms=8, mew=2)
                if k == 0:
                    ax.plot(r['goal'][0], r['goal'][1], '*', color=INK2, ms=10)
        ax.set_aspect('equal')
        ax.set_title(f'6 kg, {h * 100:.0f} cm   (box path; × = failure, ★ = goal)', fontsize=10, color=INK, loc='left')
        ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=150, facecolor=SURF)
    plt.close(fig)


def fig_tsne(fname):
    fs = {os.path.basename(f)[5:-4]: f for f in glob.glob(os.path.join(RES, 'latent_analysis', 'tsne_*.npz'))}
    tags = [t for t in ['anyadapter', 'split'] if t in fs]
    if not tags:
        return
    fig, axes = plt.subplots(1, len(tags), figsize=(5.2 * len(tags), 4.6), facecolor=SURF, squeeze=False)
    for j, t in enumerate(tags):
        d = np.load(fs[t])
        ax = axes[0, j]
        style(ax)
        ax.yaxis.grid(False)
        sc = ax.scatter(d['emb'][:, 0], d['emb'][:, 1], c=d['mass'], cmap='Blues', vmin=0, vmax=5.2, s=6,
                        edgecolors='none')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{NAMES[t]}: {'[z_obj; z_dyn]' if t == 'split' else 'unified z'}", fontsize=10, color=INK, loc='left')
        cb = fig.colorbar(sc, ax=ax, fraction=0.046)
        cb.set_label('box mass (kg)', color=INK2, fontsize=9)
    fig.suptitle('t-SNE of adaptation latents (loaded samples)', fontsize=12, color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=150, facecolor=SURF)
    plt.close(fig)


def latent_table():
    f = os.path.join(RES, 'latent_analysis', 'summary.json')
    if not os.path.exists(f):
        return ''
    s = json.load(open(f))
    lines = ['### World-model losses and predictability gap (Isaac Gym rollouts, training distribution)', '',
             '| Variant | obj WM L1 | robot WM L1 | R² z_obj→obj | R² z_dyn→obj | R² z_dyn→robot | R² z_obj→robot | gap obj | gap robot | mass probe R² |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for t in [t for t in ORDER if t in s]:
        r = s[t]
        p = r.get('probe_r2', {})
        g = lambda k: f'{p[k]:.3f}' if k in p else '–'  # noqa: E731
        lines.append(f"| {NAMES[t]} | {r['obj_wm_l1']:.4f} | {r['dyn_wm_l1']:.4f} | {g('obj->obj')} | {g('dyn->obj')} | "
                     f"{g('dyn->robot')} | {g('obj->robot')} | {r.get('gap_object', float('nan')):.3f} | "
                     f"{r.get('gap_robot', float('nan')):.3f} | {r.get('mass_probe_r2_loaded', float('nan')):.3f} |")
    return '\n'.join(lines) + '\n'


def main():
    os.makedirs(FIG, exist_ok=True)
    mj, isa = load('mujoco'), load('isaac')
    md = ['# SplitAdapter reproduction - result tables', '',
          'Cells: Full-task successes per condition (heights in cm). 10 trials per condition.', '']
    md.append(table(mj, 'MuJoCo sim-to-sim, paper protocol (training seed 1, 10 trials / condition)', protocol=True))
    md.append(table(mj, 'MuJoCo sim-to-sim, mean ± std over all training seeds x 3 evaluation seeds (per-run counts out of 90)'))
    md.append(table(isa, 'Isaac Gym (training simulator, with domain randomisation), mean over training seeds'))
    md.append(latent_table())
    open(os.path.join(RES, 'REPORT_TABLES.md'), 'w').write('\n'.join(md))
    fig_success(mj, 'success_mujoco.png', 'MuJoCo sim-to-sim success (mean over seeds)', FIG_METHODS)
    fig_success(isa, 'success_isaac.png', 'Isaac Gym success', FIG_METHODS)
    fig_mass_trace('mass_estimation.png')
    fig_traj('trajectories_6kg.png')
    fig_tsne('tsne_latents.png')
    print('\n'.join(md))
    print('figures:', sorted(os.listdir(FIG)))


if __name__ == '__main__':
    main()
