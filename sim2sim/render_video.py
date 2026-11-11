"""Render a MuJoCo sim-to-sim episode to a GIF / MP4 (off-screen), with the adapter's online estimates overlaid.

  MUJOCO_GL=egl python sim2sim/render_video.py --ckpt logs/adapter_split_s1/model_last.pt --base <base.pt> \
      --mass 4 --height 0.3 --out media/splitadapter_4kg_30cm.gif
  --find_success N : try up to N scenario seeds and render the first successful episode
"""
import os
import sys
import argparse

import numpy as np
import mujoco
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sim2sim.mujoco_carrybox import MujocoCarryBox  # noqa: E402
from sim2sim.eval_mujoco import load_policy  # noqa: E402


def run(env, ac, mass, h, seed, frames=None, renderer=None, cam=None, every=2, label=''):
    obs, hist = env.reset(mass, h, h, seed)
    done, t = False, 0
    while not done:
        with torch.no_grad():
            act, enc = ac.adapter(torch.from_numpy(obs)[None], torch.from_numpy(hist)[None])
        obs, hist, done, info = env.step(act[0].numpy())
        if frames is not None and t % every == 0:
            p = env.data.xpos[env.b_pelvis]
            cam.lookat[:] = [p[0], p[1], 0.6]
            renderer.update_scene(env.data, camera=cam)
            img = Image.fromarray(renderer.render())
            d = ImageDraw.Draw(img, 'RGBA')
            txt = f'{label}   box {mass:.0f} kg, pickup/place {h * 100:.0f} cm   t={t * env.dt:4.1f}s'
            if 'loaded_p' in enc:
                txt += f'\nloaded p={float(enc["loaded_p"][0]):.2f}   est. payload={float(enc["m_eff"][0]):.1f} kg'
            if env.ever_success:
                txt += '\nbox placed at goal'
            box = d.multiline_textbbox((12, 10), txt, font=FONT, spacing=4)
            d.rectangle((box[0] - 6, box[1] - 5, box[2] + 6, box[3] + 5), fill=(255, 255, 255, 190))
            d.multiline_text((12, 10), txt, fill=(20, 20, 20), font=FONT, spacing=4)
            frames.append(np.asarray(img))
        t += 1
        if env.ever_success and frames is not None and t * env.dt > env.success_time + 2.5:
            break   # keep ~2.5 s after the successful placement
    return env.ever_success


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='base')
    p.add_argument('--base', default=None)
    p.add_argument('--mass', type=float, default=4.0)
    p.add_argument('--height', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--find_success', type=int, default=0)
    p.add_argument('--label', default='SplitAdapter')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height_px', type=int, default=400)
    p.add_argument('--fps', type=int, default=25)
    p.add_argument('--out', default='media/episode.gif')
    a = p.parse_args()
    torch.set_num_threads(4)
    ac, _ = load_policy(a.ckpt, a.base)
    env = MujocoCarryBox()
    seed = a.seed
    if a.find_success:
        for s in range(a.seed, a.seed + a.find_success):
            if run(env, ac, a.mass, a.height, s):
                seed = s
                break
        print('rendering seed', seed)
    renderer = mujoco.Renderer(env.model, a.height_px, a.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 3.4, 140.0, -16.0
    frames = []
    ok = run(env, ac, a.mass, a.height, seed, frames, renderer, cam, every=max(1, round(1 / (env.dt * a.fps))),
             label=a.label)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    if a.out.endswith('.gif'):
        imageio.mimsave(a.out, frames, duration=1000 / a.fps, loop=0)
    else:
        imageio.mimsave(a.out, frames, fps=a.fps, quality=7, macro_block_size=8)
    print(f'saved {a.out}: {len(frames)} frames, success={ok}')


FONT = ImageFont.load_default()
for _f in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf']:
    if os.path.exists(_f):
        FONT = ImageFont.truetype(_f, 15)
        break

if __name__ == '__main__':
    main()
