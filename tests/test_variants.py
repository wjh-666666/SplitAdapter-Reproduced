"""All adapter variants: forward / aux losses / backward on random data; FiLM identity at init; GRL sign."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from splitadapter.modules.split_adapter import AdapterActorCritic, grl

class Cfg:  # mirrors SplitCarryBoxCfgPPO.adapter
    latent_obj = 16; latent_dyn = 16; latent_unified = 32; film_hidden = 128; grl_coef = 0.1
    grl_warmup_steps = 0; w_mass = 2.0; w_loaded = 1.0; w_obj_wm = 1.0
    w_dyn_wm = 1.0; w_grl = 1e-3; w_kl = 1e-3; w_l1_obj = 1e-3; w_l1_dyn = 5e-3; lambda_p = 1.0; lambda_R = 1.0

base = torch.load('third_party/PhysHSI/legged_gym/resources/ckpt/carrybox.pt', map_location='cpu')['model_state_dict']
B = 256
obs, hist = torch.randn(B, 738), torch.randn(B, 50, 123)
ref = None
for v in ['split', 'anyadapter', 'no_split', 'no_hier_film', 'no_grl']:
    cfg = Cfg(); cfg.variant = v
    ac = AdapterActorCritic(base, cfg, 126)
    mean, enc = ac.act_dist(obs, hist)
    with torch.no_grad():
        h = obs
        for i in (0, 2, 4):
            h = torch.nn.functional.elu(h @ base[f'actor.{i}.weight'].T + base[f'actor.{i}.bias'])
        base_out = h @ base['actor.6.weight'].T + base['actor.6.bias']
    assert torch.allclose(mean, base_out, atol=1e-5), v
    L, logs = ac.adapter.aux_losses(enc, obs[:, -123:], torch.randn(B, 29), torch.randn(B, 79), torch.randn(B, 6),
                                    torch.ones(B, dtype=torch.bool), torch.rand(B) * 5, torch.rand(B),
                                    (torch.rand(B) > 0.5).float(), torch.ones(B, dtype=torch.bool), True)
    (sum(L.values()) + mean.pow(2).mean()).backward()
    assert all(p.grad is None for p in ac.adapter.base.parameters()), 'base must stay frozen'
    nfilm = sum(p.numel() for p in ac.adapter.film_parameters()); naux = sum(p.numel() for p in ac.adapter.aux_parameters())
    print(f'{v:13s} losses={sorted(L)}  film={nfilm} aux={naux}  identity-at-init OK')
x = torch.ones(3, requires_grad=True); grl(x, 0.1).sum().backward(); assert torch.allclose(x.grad, -0.1 * torch.ones(3))
print('GRL gradient reversal OK')
