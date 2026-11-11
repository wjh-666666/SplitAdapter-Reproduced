"""Dependency-aware multi-GPU job scheduler that runs the whole reproduction.

Every training job is an independent single-GPU process with checkpoint-based auto-resume (Isaac Gym Preview 4
can segfault at random on recent GPUs; a crashed job is relaunched from its last checkpoint).  Runnable jobs are
dispatched by priority so that all GPUs stay busy.

  nohup python scripts/scheduler.py --gpus 0,1,2,3,4,5,6,7 > logs/scheduler.log 2>&1 &
  cat logs/scheduler_status.txt

A GPU id may be listed twice (e.g. --gpus 0,1,2,3,4,5,6,7,7) to host two jobs.  Plan (see build_jobs):
  1. fine-tune the released PhysHSI base with 3 seeds; select the best one by Isaac Gym evaluation
  2. 5 methods x 3 training seeds on the selected base (1500 iterations each)
  3. Isaac Gym + MuJoCo (3 evaluation seeds) evaluation of every model; AnyAdapter is additionally evaluated
     at iteration 500 (it diverges later in training, see README)
"""
import os
import re
import sys
import json
import glob
import time
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = sys.executable
MAX_CPU_JOBS = 2
MAX_ATTEMPTS = 60
RELEASED = 'third_party/PhysHSI/legged_gym/resources/ckpt/carrybox.pt'
BASE_ITERS, ADAPTER_ITERS = 1500, 1500
VARIANTS = ['split', 'anyadapter', 'no_split', 'no_grl', 'no_hier_film']
ENV = dict(os.environ, LD_LIBRARY_PATH=os.path.join(sys.prefix, 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''),
           CUDA_CACHE_MAXSIZE='4294967296')   # keep PTX-JIT-compiled kernels across processes
SELECTED = 'logs/selected_base.txt'
SLOTS = list(range(8))


def log(msg):
    print(time.strftime('%m-%d %H:%M:%S'), msg, flush=True)


def latest_ckpt(pattern):
    best = None
    for f in glob.glob(pattern):
        m = re.search(r'model_(\d+)\.pt$', f)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), f)
    return best


def selected_base():
    return open(SELECTED).read().strip() if os.path.exists(SELECTED) else None


# ----------------------------------------------------------------------------- job definitions
class Job:
    def __init__(self, name, prio, gpu=True, deps=()):
        self.name, self.prio, self.gpu, self.deps = name, prio, gpu, list(deps)
        self.proc, self.slot, self.attempts, self.state = None, None, 0, 'pending'

    def done(self):
        raise NotImplementedError

    def command(self):
        raise NotImplementedError


class BaseJob(Job):
    def __init__(self, seed):
        super().__init__(f'base_s{seed}', 100)
        self.seed, self.root = seed, f'logs/base_s{seed}'

    def final(self):
        c = latest_ckpt(f'{self.root}/*/model_{BASE_ITERS - 1}.pt')
        return c[1] if c else None

    def done(self):
        return self.final() is not None

    def command(self):
        c = latest_ckpt(f'{self.root}/*/model_*.pt')
        init, start = (RELEASED, 0) if c is None else (c[1], c[0] + 1)
        return (f'CUDA_VISIBLE_DEVICES={SLOTS[self.slot]} {PY} scripts/train_base.py --iters {BASE_ITERS} '
                f'--start_iter {start} --init {init} --seed {self.seed} --log_root {self.root} --headless '
                f'>> logs/train_base_s{self.seed}.log 2>&1')


class AdapterJob(Job):
    def __init__(self, variant, seed):
        super().__init__(f'adapter_{variant}_s{seed}', 80, deps=['select_base'])
        self.variant, self.seed, self.dir = variant, seed, f'logs/adapter_{variant}_s{seed}'

    def done(self):
        return os.path.exists(f'{self.dir}/model_last.pt')

    def command(self):
        os.makedirs(self.dir, exist_ok=True)
        c = latest_ckpt(f'{self.dir}/model_*.pt')
        res = f'--resume_adapter {c[1]}' if c else ''
        return (f'CUDA_VISIBLE_DEVICES={SLOTS[self.slot]} {PY} scripts/train_adapter.py --variant {self.variant} '
                f'--seed {self.seed} --iters {ADAPTER_ITERS} --base {selected_base()} --log_dir {self.dir} {res} '
                f'--headless >> logs/train_{self.name}.log 2>&1')


class IsaacEvalJob(Job):
    def __init__(self, tag, ckpt_fn, base_fn, deps, prio=95):
        super().__init__(f'isaac_{tag}', prio, deps=deps)
        self.tag, self.ckpt_fn, self.base_fn = tag, ckpt_fn, base_fn

    def done(self):
        return os.path.exists(f'results/isaac_{self.tag}_s0.json')

    def command(self):
        return (f'CUDA_VISIBLE_DEVICES={SLOTS[self.slot]} {PY} scripts/eval_isaac.py --ckpt {self.ckpt_fn()} '
                f'--base {self.base_fn()} --tag {self.tag} --trials 10 --headless > logs/eval_isaac_{self.tag}.log 2>&1')


class MujocoEvalJob(Job):
    """MuJoCo sim-to-sim evaluation with 3 evaluation seeds (CPU)."""

    def __init__(self, tag, ckpt_fn, base_fn, deps, prio=60):
        super().__init__(f'mujoco_{tag}', prio, gpu=False, deps=deps)
        self.tag, self.ckpt_fn, self.base_fn = tag, ckpt_fn, base_fn

    def outs(self):
        return [f'results/mujoco_{self.tag}_s{s}.json' for s in range(3)]

    def done(self):
        return all(os.path.exists(o) for o in self.outs())

    def command(self):
        cmds = [f'{PY} sim2sim/eval_mujoco.py --ckpt {self.ckpt_fn()} --base {self.base_fn()} --tag {self.tag} '
                f'--seed {s} --trials 10 --workers 50 --out {o}' for s, o in enumerate(self.outs())]
        return '{ ' + ' && '.join(cmds) + f'; }} > logs/eval_mujoco_{self.tag}.log 2>&1'


class SelectJob(Job):
    """Pick the fine-tuned base with the best Isaac Gym (training simulator) Full-task score."""

    def __init__(self, seeds):
        super().__init__('select_base', 99, gpu=False, deps=[f'isaac_base_ft_s{s}' for s in seeds])
        self.seeds = seeds

    def done(self):
        return selected_base() is not None

    def run_inline(self, jobs):
        best = None
        for s in self.seeds:
            r = json.load(open(f'results/isaac_base_ft_s{s}_s0.json'))
            score = (r['full_total'], r['lift_total'])
            log(f'  base seed {s}: Isaac Full-task {score[0]}/{r["n"]}  Lift-up {score[1]}/{r["n"]}')
            if best is None or score > best[0]:
                best = (score, s)
        path = jobs[f'base_s{best[1]}'].final()
        open(SELECTED, 'w').write(path + '\n')
        log(f'selected base: seed {best[1]} -> {path}')


def build_jobs():
    J = []
    seeds = [1, 2, 3]
    for s in seeds:
        b = BaseJob(s)
        J.append(b)
        J.append(IsaacEvalJob(f'base_ft_s{s}', (lambda: 'base'), (lambda b=b: b.final()), deps=[b.name], prio=98))
    J.append(SelectJob(seeds))
    for tag, base_fn, deps in [('base_ft', selected_base, ['select_base']), ('physhsi_release', lambda: RELEASED, [])]:
        J.append(IsaacEvalJob(tag, lambda: 'base', base_fn, deps=deps))
        J.append(MujocoEvalJob(tag, lambda: 'base', base_fn, deps=deps))
    for v in VARIANTS:
        for s in seeds:
            a = AdapterJob(v, s)
            J.append(a)
            ckpts = [('model_last.pt', f'{v}_s{s}')] + ([('model_500.pt', f'{v}@500_s{s}')] if v == 'anyadapter' else [])
            for ck, tag in ckpts:
                J.append(IsaacEvalJob(tag, (lambda a=a, ck=ck: f'{a.dir}/{ck}'), selected_base, deps=[a.name]))
                J.append(MujocoEvalJob(tag, (lambda a=a, ck=ck: f'{a.dir}/{ck}'), selected_base, deps=[a.name]))
    return {j.name: j for j in J}


# ----------------------------------------------------------------------------- scheduler loop
def launch(j, slot):
    j.slot, j.attempts, j.state = slot, j.attempts + 1, 'running'
    j.proc = subprocess.Popen(j.command(), shell=True, env=ENV, cwd=ROOT, start_new_session=True,
                              executable='/bin/bash')
    log(f'start  {j.name:32s} slot={slot if j.gpu else "-"} attempt={j.attempts}')


def write_status(jobs):
    lines = [time.strftime('%Y-%m-%d %H:%M:%S')]
    for g in range(len(SLOTS)):
        r = [j.name for j in jobs.values() if j.state == 'running' and j.gpu and j.slot == g]
        lines.append(f'slot{g} (gpu{SLOTS[g]}): {", ".join(r) or "IDLE"}')
    lines.append('cpu : ' + ', '.join(j.name for j in jobs.values() if j.state == 'running' and not j.gpu))
    for st in ['pending', 'done', 'failed']:
        n = [j.name for j in jobs.values() if j.state == st]
        lines.append(f'{st} ({len(n)}): ' + ', '.join(n))
    open('logs/scheduler_status.txt', 'w').write('\n'.join(lines) + '\n')


def main():
    global SLOTS
    p = argparse.ArgumentParser()
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7', help='comma separated GPU ids; repeat an id for 2 jobs/GPU')
    SLOTS = [int(g) for g in p.parse_args().gpus.split(',')]
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    jobs = build_jobs()
    for j in jobs.values():
        if j.done():
            j.state = 'done'
    log(f'{len(jobs)} jobs, {sum(j.state == "done" for j in jobs.values())} already done, slots={SLOTS}')
    while True:
        # 1) reap finished / crashed processes
        for j in jobs.values():
            if j.state == 'running' and j.proc.poll() is not None:
                rc = j.proc.returncode
                if j.done():
                    j.state = 'done'
                    log(f'done   {j.name:32s} (rc={rc})')
                elif j.attempts >= MAX_ATTEMPTS:
                    j.state = 'failed'
                    log(f'FAILED {j.name:32s} after {j.attempts} attempts (rc={rc})')
                else:
                    j.state = 'pending'
                    log(f'crash  {j.name:32s} rc={rc} -> will resume')
        # 2) inline jobs
        for j in jobs.values():
            if isinstance(j, SelectJob) and j.state == 'pending' and all(jobs[d].state == 'done' for d in j.deps):
                j.run_inline(jobs)
                j.state = 'done'
        runnable = sorted([j for j in jobs.values() if j.state == 'pending' and not isinstance(j, SelectJob)
                           and all(jobs[d].state == 'done' for d in j.deps)], key=lambda j: -j.prio)
        # 3) GPU jobs
        busy = {j.slot for j in jobs.values() if j.state == 'running' and j.gpu}
        free = [g for g in range(len(SLOTS)) if g not in busy]
        for j in [r for r in runnable if r.gpu]:
            if not free:
                break
            launch(j, free.pop(0))
        # 4) CPU jobs
        ncpu = sum(1 for j in jobs.values() if j.state == 'running' and not j.gpu)
        for j in [r for r in runnable if not r.gpu]:
            if ncpu >= MAX_CPU_JOBS:
                break
            launch(j, None)
            ncpu += 1
        write_status(jobs)
        if all(j.state in ('done', 'failed') for j in jobs.values()):
            log('all jobs finished')
            break
        time.sleep(20)


if __name__ == '__main__':
    main()
