"""Print selected TensorBoard scalars of a run: python tests/read_tb.py <logdir> [tag_substring ...]"""
import sys, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator(glob.glob(sys.argv[1] + '/events*')[0], size_guidance={'scalars': 0}); ea.Reload()
keys = [k for k in ea.Tags()['scalars'] if (len(sys.argv) < 3 or any(s in k for s in sys.argv[2:]))]
for k in keys:
    v = ea.Scalars(k)
    pts = v if len(v) <= 8 else [v[int(i * (len(v) - 1) / 7)] for i in range(8)]
    print(f"{k:28s}", ' '.join(f"{p.step}:{p.value:.4g}" for p in pts))
