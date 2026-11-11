# SplitAdapter reproduction - result tables

Cells: Full-task successes per condition (heights in cm). 10 trials per condition.

### MuJoCo sim-to-sim, paper protocol (training seed 1, 10 trials / condition)

| Method | 2kg 0 | 2kg 30 | 2kg 60 | 4kg 0 | 4kg 30 | 4kg 60 | 6kg 0 | 6kg 30 | 6kg 60 | Full-task | Lift-up |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PhysHSI (released ckpt) | 9 | 8 | 0 | 5 | 7 | 0 | 2 | 3 | 0 | 34/90 | 75/90 |
| Base (PhysHSI, fine-tuned) | 8 | 9 | 1 | 4 | 8 | 1 | 1 | 7 | 1 | 40/90 | 58/90 |
| AnyAdapter-style (it 500) | 9 | 9 | 5 | 5 | 9 | 5 | 0 | 9 | 2 | 53/90 | 62/90 |
| AnyAdapter-style (it 1500) | 1 | 4 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 8/90 | 50/90 |
| w/o split latent | 9 | 10 | 3 | 3 | 10 | 2 | 1 | 8 | 2 | 48/90 | 66/90 |
| w/o hierarchical FiLM | 9 | 10 | 4 | 7 | 9 | 2 | 2 | 10 | 1 | 54/90 | 67/90 |
| w/o GRL | 10 | 10 | 3 | 5 | 10 | 2 | 1 | 10 | 1 | 52/90 | 66/90 |
| SplitAdapter | 8 | 10 | 3 | 7 | 9 | 3 | 2 | 8 | 1 | 51/90 | 63/90 |

### MuJoCo sim-to-sim, mean ± std over all training seeds x 3 evaluation seeds (per-run counts out of 90)

| Method | 2kg 0 | 2kg 30 | 2kg 60 | 4kg 0 | 4kg 30 | 4kg 60 | 6kg 0 | 6kg 30 | 6kg 60 | Full-task | Lift-up |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PhysHSI (released ckpt) | 6.0 | 7.3 | 0.7 | 4.0 | 6.7 | 0.3 | 1.0 | 1.7 | 0.0 | 27.7±4.5/90 | 76.7/90 |
| Base (PhysHSI, fine-tuned) | 7.3 | 9.7 | 2.0 | 3.3 | 9.0 | 0.7 | 0.3 | 7.3 | 1.3 | 41.0±1.4/90 | 55.0/90 |
| AnyAdapter-style (it 500) | 5.9 | 9.6 | 3.1 | 3.4 | 9.0 | 1.7 | 0.4 | 7.0 | 1.0 | 41.1±5.8/90 | 59.6/90 |
| AnyAdapter-style (it 1500) | 0.3 | 2.9 | 0.2 | 0.1 | 1.2 | 0.1 | 0.1 | 0.4 | 0.0 | 5.4±4.6/90 | 31.9/90 |
| w/o split latent | 8.7 | 9.9 | 2.9 | 4.3 | 9.2 | 2.7 | 1.6 | 7.7 | 2.3 | 49.2±2.7/90 | 67.1/90 |
| w/o hierarchical FiLM | 8.9 | 9.9 | 2.7 | 5.7 | 9.1 | 1.9 | 1.9 | 7.3 | 1.9 | 49.2±4.1/90 | 66.2/90 |
| w/o GRL | 8.7 | 9.7 | 2.2 | 4.7 | 9.2 | 2.4 | 1.7 | 7.7 | 1.6 | 47.8±3.3/90 | 64.9/90 |
| SplitAdapter | 9.0 | 9.9 | 3.0 | 6.3 | 9.6 | 2.6 | 1.6 | 7.7 | 2.2 | 51.8±3.8/90 | 67.2/90 |

### Isaac Gym (training simulator, with domain randomisation), mean over training seeds

| Method | 2kg 0 | 2kg 30 | 2kg 60 | 4kg 0 | 4kg 30 | 4kg 60 | 6kg 0 | 6kg 30 | 6kg 60 | Full-task | Lift-up |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PhysHSI (released ckpt) | 9 | 3 | 0 | 5 | 3 | 0 | 1 | 1 | 0 | 22/90 | 73/90 |
| Base (PhysHSI, fine-tuned) | 8 | 3 | 0 | 4 | 5 | 2 | 1 | 2 | 0 | 25/90 | 55/90 |
| AnyAdapter-style (it 500) | 7.0 | 6.3 | 0.0 | 3.3 | 4.3 | 0.0 | 0.3 | 2.0 | 0.0 | 23.3±1.9/90 | 52.7/90 |
| AnyAdapter-style (it 1500) | 0.0 | 0.7 | 0.0 | 0.0 | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0±1.4/90 | 29.7/90 |
| w/o split latent | 7.7 | 3.0 | 0.3 | 6.0 | 4.3 | 1.7 | 2.3 | 2.0 | 0.3 | 27.7±3.3/90 | 59.7/90 |
| w/o hierarchical FiLM | 7.7 | 3.3 | 0.0 | 6.7 | 4.3 | 0.7 | 2.3 | 2.7 | 0.3 | 28.0±4.3/90 | 59.0/90 |
| w/o GRL | 7.0 | 3.7 | 0.3 | 6.7 | 4.0 | 1.0 | 1.7 | 4.3 | 0.0 | 28.7±2.4/90 | 58.0/90 |
| SplitAdapter | 7.3 | 3.7 | 0.3 | 7.3 | 6.7 | 1.7 | 1.3 | 4.7 | 0.0 | 33.0±2.8/90 | 60.7/90 |
| Base fine-tune candidates (3 seeds) | 5.7 | 4.7 | 0.3 | 4.3 | 4.7 | 0.7 | 1.7 | 3.0 | 0.0 | 25.0±5.1/90 | 49.3/90 |

### World-model losses and predictability gap (Isaac Gym rollouts, training distribution)

| Variant | obj WM L1 | robot WM L1 | R² z_obj→obj | R² z_dyn→obj | R² z_dyn→robot | R² z_obj→robot | gap obj | gap robot | mass probe R² |
|---|---|---|---|---|---|---|---|---|---|
| AnyAdapter-style (it 1500) | 0.1409 | 0.0250 | – | – | – | – | nan | nan | 0.211 |
| w/o split latent | 0.1411 | 0.0250 | – | – | – | – | nan | nan | 0.608 |
| w/o hierarchical FiLM | 0.1376 | 0.0220 | 0.353 | 0.023 | 0.003 | 0.018 | 0.330 | -0.015 | -0.615 |
| w/o GRL | 0.1380 | 0.0220 | 0.360 | 0.000 | 0.000 | 0.016 | 0.359 | -0.016 | 0.615 |
| SplitAdapter | 0.1345 | 0.0220 | 0.359 | 0.059 | 0.011 | 0.016 | 0.300 | -0.005 | 0.552 |
