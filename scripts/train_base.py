"""Stage 1: fine-tune the released PhysHSI carry-box policy on the paper's task distribution.

  single GPU : CUDA_VISIBLE_DEVICES=0 python scripts/train_base.py --seed 1 --log_root logs/base_s1 --headless
  resume     : ... --init logs/base_s1/<run>/model_<N>.pt --start_iter <N+1>
  multi GPU  : torchrun --nproc_per_node 8 scripts/train_base.py --headless   (data parallel, see below)

Multi-GPU: each rank simulates its own envs on its own GPU (Isaac Gym sees one device per process); gradients of
actor-critic + AMP discriminator are averaged across ranks before every optimizer step and rank 0's KL-adaptive
learning rate is broadcast, so all replicas stay identical.  NOTE: on Blackwell GPUs Isaac Gym segfaults at random,
and with DDP one crashing rank stops all of them -- use independent single-GPU jobs there (scripts/scheduler.py).
"""
import os
import sys

LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
RANK = int(os.environ.get('RANK', 0))
WORLD = int(os.environ.get('WORLD_SIZE', 1))
if 'LOCAL_RANK' in os.environ:   # launched by torchrun: one GPU per process -> device cuda:0 everywhere
    os.environ['CUDA_VISIBLE_DEVICES'] = str(LOCAL_RANK)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splitadapter.utils import parse, make_env, task_registry, BASE_CKPT, ROOT  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import rsl_rl.runners.him_on_policy_runner as him_runner_mod  # noqa: E402


def make_synced_step(alg, opt, world):
    params = [p for g in opt.param_groups for p in g['params']]
    orig_step = opt.step

    def step(*a, **k):
        grads = [p.grad for p in params if p.grad is not None]
        if world > 1 and grads:
            flat = torch.cat([g.reshape(-1) for g in grads])
            dist.all_reduce(flat)
            flat /= world
            i = 0
            for g in grads:
                n = g.numel()
                g.copy_(flat[i:i + n].view_as(g))
                i += n
            lr = torch.tensor([float(alg.learning_rate)], device=flat.device)
            dist.broadcast(lr, 0)
            alg.learning_rate = float(lr.item())
            for g in opt.param_groups:
                g['lr'] = alg.learning_rate
        return orig_step(*a, **k)

    opt.step = step


def main():
    ours, gym_args = parse([('--iters', dict(type=int, default=None)),
                            ('--init', dict(type=str, default=BASE_CKPT)),
                            ('--envs', dict(type=int, default=None)),
                            ('--start_iter', dict(type=int, default=0)),
                            ('--save_interval', dict(type=int, default=10)),
                            ('--log_root', dict(type=str, default=None))])
    if WORLD > 1:
        dist.init_process_group('nccl', rank=RANK, world_size=WORLD)
        torch.cuda.set_device(0)
    if RANK != 0:
        sys.stdout = open(os.devnull, 'w')   # only rank 0 prints the training log
    env_cfg, train_cfg = task_registry.get_cfgs('carrybox_split_base')
    if ours.envs:
        env_cfg.env.num_envs = ours.envs
    env_cfg.seed = (gym_args.seed or 1) + 1000 * RANK
    train_cfg.seed = env_cfg.seed
    gym_args.seed = env_cfg.seed
    train_cfg.runner.resume = True
    train_cfg.runner.resume_path = ours.init
    iters = ours.iters or train_cfg.runner.max_iterations
    train_cfg.runner.save_interval = ours.save_interval
    env, _ = make_env(gym_args, env_cfg, name='carrybox_split_base')
    if RANK != 0:
        him_runner_mod.store_code_state = lambda *a, **k: []
    # non-zero ranks get a throw-away log dir (PhysHSI's runner builds save paths from it); their saves are no-ops
    log_root = (ours.log_root or os.path.join(ROOT, 'logs', 'base')) if RANK == 0 else f'/tmp/splitadapter_ddp_rank{RANK}'
    runner, _ = task_registry.make_alg_runner(env=env, name='carrybox_split_base', args=gym_args,
                                              train_cfg=train_cfg, log_root=log_root)
    if RANK != 0:
        runner.save = lambda *a, **k: None
        runner.logger_type = 'tensorboard'
    # PhysHSI's HIMPPO keeps the adaptive lr in `alg.learning_rate`, which runner.load() does not restore (only the
    # optimizer state); restore it so a resumed run continues with the adapted learning rate.
    runner.alg.learning_rate = float(runner.alg.optimizer.param_groups[0]['lr'])
    if RANK == 0:
        print(f'[resume] restored adaptive lr = {runner.alg.learning_rate:.2e}', flush=True)
    make_synced_step(runner.alg, runner.alg.optimizer, WORLD)
    if WORLD > 1:  # identical start (all ranks loaded the same checkpoint; make sure)
        for p in list(runner.alg.actor_critic.parameters()) + list(runner.alg.amp.parameters()):
            dist.broadcast(p.data, 0)
    runner.current_learning_iteration = ours.start_iter
    if RANK == 0:
        print(f'[DDP] world={WORLD}  envs/rank={env.num_envs}  total envs={env.num_envs * WORLD}  '
              f'iters {ours.start_iter}->{iters}  init={ours.init}', flush=True)
    runner.learn(num_learning_iterations=iters - ours.start_iter, init_at_random_ep_len=True)
    if WORLD > 1:
        # replicas must be identical: compare a parameter checksum across ranks
        with torch.no_grad():
            cs = torch.stack([sum(p.double().sum() for p in runner.alg.actor_critic.parameters()),
                              sum(p.double().sum() for p in runner.alg.amp.parameters())]).cuda()
        allcs = [torch.zeros_like(cs) for _ in range(WORLD)]
        dist.all_gather(allcs, cs)
        if RANK == 0:
            allcs = torch.stack(allcs)
            print(f'[DDP] param checksum spread across ranks: {(allcs.max(0).values - allcs.min(0).values).tolist()}',
                  flush=True)
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
