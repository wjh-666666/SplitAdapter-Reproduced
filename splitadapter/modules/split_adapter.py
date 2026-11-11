"""SplitAdapter modules.

  FrozenBasePolicy : the PhysHSI actor MLP [512, 256, 256] (+29-D head), weights frozen, with FiLM hooks
  SharedHistoryEncoder : Conv1d(123,64,k6,s2)+ELU -> Conv1d(64,32,k4,s2)+ELU -> 320-D (Table 5)
  SplitAdapterModel : object/load branch (task head: z_obj, mass, loaded), dynamics branch (z_dyn),
                      split world models, GRL cross adversaries, hierarchical FiLM generator
Variants (Table 1):
  'split'        : full SplitAdapter
  'no_grl'       : w/o GRL cross-adversarial separation
  'no_hier_film' : w/o hierarchical FiLM (both contexts modulate every layer, no routing)
  'no_split'     : w/o split latent (one unified latent + mass/loaded heads, single WM)
  'anyadapter'   : AnyAdapter-style WM-FiLM (one latent, one WM, FiLM everywhere, no mass/loaded heads)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- utils
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coef):
        ctx.coef = coef
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.coef * g, None


def grl(x, coef):
    return _GradReverse.apply(x, coef)


def mlp(inp, hidden, out, act=nn.ELU):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class EmpiricalNormalizer(nn.Module):
    """Running mean / std normaliser (per feature) for the adapter history."""

    def __init__(self, dim, eps=1e-2, until=int(1e9)):
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim))
        self.register_buffer('count', torch.zeros(()))

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1])
        if self.count >= self.until:
            return
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta ** 2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps)


# ----------------------------------------------------------------------------- frozen base
class FrozenBasePolicy(nn.Module):
    """PhysHSI actor: Linear(738,512)-ELU-Linear(512,256)-ELU-Linear(256,256)-ELU-Linear(256,29)."""

    def __init__(self, actor_state_dict):
        super().__init__()
        w = lambda i: actor_state_dict[f'actor.{i}.weight']
        self.l1 = nn.Linear(w(0).shape[1], w(0).shape[0])
        self.l2 = nn.Linear(w(2).shape[1], w(2).shape[0])
        self.l3 = nn.Linear(w(4).shape[1], w(4).shape[0])
        self.head = nn.Linear(w(6).shape[1], w(6).shape[0])
        for name, lin in zip([0, 2, 4, 6], [self.l1, self.l2, self.l3, self.head]):
            lin.weight.data.copy_(actor_state_dict[f'actor.{name}.weight'])
            lin.bias.data.copy_(actor_state_dict[f'actor.{name}.bias'])
        for p in self.parameters():
            p.requires_grad_(False)
        self.dims = [self.l1.out_features, self.l2.out_features, self.l3.out_features]

    def train(self, mode=True):  # always eval (no dropout/bn anyway)
        return super().train(False)

    def forward(self, obs, film=None):
        """film: dict with optional keys 'h1', 'h2a', 'h2b', 'h3' -> (gamma, beta)."""
        film = film or {}

        def mod(h, k):
            if k in film:
                g, b = film[k]
                h = g * h + b
            return h

        h1 = mod(F.elu(self.l1(obs)), 'h1')
        h2 = mod(mod(F.elu(self.l2(h1)), 'h2a'), 'h2b')
        h3 = mod(F.elu(self.l3(h2)), 'h3')
        return self.head(h3)


# ----------------------------------------------------------------------------- encoder
class SharedHistoryEncoder(nn.Module):
    def __init__(self, obs_dim=123, hist_len=50):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(obs_dim, 64, kernel_size=6, stride=2), nn.ELU(),
                                 nn.Conv1d(64, 32, kernel_size=4, stride=2), nn.ELU())
        with torch.no_grad():
            self.out_dim = self.net(torch.zeros(1, obs_dim, hist_len)).numel()

    def forward(self, hist):  # (B, H, D)
        return self.net(hist.transpose(1, 2)).flatten(1)


# ----------------------------------------------------------------------------- FiLM
class FiLMHead(nn.Module):
    """context -> (gamma, beta) for a set of layers; zero-initialised output => identity at start."""

    def __init__(self, ctx_dim, layer_dims: dict, hidden=128):
        super().__init__()
        self.layer_dims = layer_dims
        out = sum(2 * d for d in layer_dims.values())
        self.net = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, out))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, ctx):
        raw = self.net(ctx)
        res, i = {}, 0
        for k, d in self.layer_dims.items():
            dg, b = raw[:, i:i + d], raw[:, i + d:i + 2 * d]
            res[k] = (1.0 + dg, b)
            i += 2 * d
        return res


# ----------------------------------------------------------------------------- adapter
class SplitAdapterModel(nn.Module):
    ROBOT_DIM = 79
    OBJ_DIM = 6
    ACT_DIM = 29

    def __init__(self, base: FrozenBasePolicy, cfg, obs_dim=123, hist_len=50):
        super().__init__()
        self.base = base
        self.variant = cfg.variant
        self.cfg = cfg
        self.split = self.variant in ('split', 'no_grl', 'no_hier_film')
        self.use_mass = self.variant != 'anyadapter'
        self.use_grl = self.variant in ('split', 'no_hier_film')
        self.hier = self.variant in ('split', 'no_grl')
        d1, d2, d3 = base.dims
        self.normalizer = EmpiricalNormalizer(obs_dim)
        self.encoder = SharedHistoryEncoder(obs_dim, hist_len)
        E = self.encoder.out_dim
        A, O, R = self.ACT_DIM, obs_dim, self.ROBOT_DIM

        if self.split:
            zo, zd = cfg.latent_obj, cfg.latent_dyn
            # object/load branch ("Task Head"): Linear(320,64)+ELU, Linear(64,64)+ELU -> z_obj, mass, loaded
            self.obj_trunk = nn.Sequential(nn.Linear(E, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU())
            self.z_obj_head = nn.Linear(64, zo)
            self.mass_head = nn.Linear(64, 1)
            self.loaded_head = nn.Linear(64, 1)
            # dynamics branch: Linear(320,64)+ELU, Linear(64,16)
            self.dyn_branch = nn.Sequential(nn.Linear(E, 64), nn.ELU(), nn.Linear(64, zd))
            # split world models
            self.obj_wm = mlp(O + A + 2 + zo, [64, 64], self.OBJ_DIM)
            self.dyn_wm = mlp(O + A + zd + 1, [128, 128], R)
            # GRL cross adversaries
            self.adv_obj_from_dyn = mlp(O + A + zd + 1, [64, 64], self.OBJ_DIM)
            self.adv_robot_from_obj = mlp(O + A + zo + 1, [128, 128], R)
            if self.hier:
                # hierarchical FiLM: object/load context -> early layers, dynamics context -> late layers
                self.film_obj = FiLMHead(zo + 1, {'h1': d1, 'h2a': d2}, cfg.film_hidden)
                self.film_dyn = FiLMHead(zd, {'h2b': d2, 'h3': d3}, cfg.film_hidden)
            else:
                self.film_all = FiLMHead(zo + 1 + zd, {'h1': d1, 'h2a': d2, 'h3': d3}, cfg.film_hidden)
        else:
            zu = cfg.latent_unified
            self.uni_branch = nn.Sequential(nn.Linear(E, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU())
            self.z_head = nn.Linear(64, zu)
            extra = 0
            if self.use_mass:
                self.mass_head = nn.Linear(64, 1)
                self.loaded_head = nn.Linear(64, 1)
                extra = 1
            # single (unified) world model predicting robot + object transition
            self.uni_wm = mlp(O + A + zu + extra, [128, 128], R + self.OBJ_DIM)
            self.film_all = FiLMHead(zu + extra, {'h1': d1, 'h2a': d2, 'h3': d3}, cfg.film_hidden)

    # ------------------------------------------------------------------
    def encode(self, hist):
        e = self.encoder(self.normalizer(hist))
        out = {'e': e}
        if self.split:
            t = self.obj_trunk(e)
            out['z_obj'] = self.z_obj_head(t)
            out['z_dyn'] = self.dyn_branch(e)
            out['mass'] = F.softplus(self.mass_head(t)).squeeze(-1)
            out['loaded_logit'] = self.loaded_head(t).squeeze(-1)
        else:
            t = self.uni_branch(e)
            out['z'] = self.z_head(t)
            if self.use_mass:
                out['mass'] = F.softplus(self.mass_head(t)).squeeze(-1)
                out['loaded_logit'] = self.loaded_head(t).squeeze(-1)
        if self.use_mass:
            out['loaded_p'] = torch.sigmoid(out['loaded_logit'])
            out['m_eff'] = out['loaded_p'] * out['mass']
        return out

    def film_params(self, enc):
        # The history encoder is trained only by the auxiliary objectives (world models, mass / loaded supervision,
        # GRL); the RL loss updates the FiLM generators, so the latents are detached here.
        f = lambda x: x.detach()  # noqa: E731
        if self.split:
            ctx_obj = torch.cat([f(enc['z_obj']), enc['m_eff'].unsqueeze(-1).detach()], -1)
            ctx_dyn = f(enc['z_dyn'])
            if self.hier:
                film = self.film_obj(ctx_obj)
                film.update(self.film_dyn(ctx_dyn))
                return film
            return self.film_all(torch.cat([ctx_obj, ctx_dyn], -1))
        ctx = f(enc['z'])
        if self.use_mass:
            ctx = torch.cat([ctx, enc['m_eff'].unsqueeze(-1).detach()], -1)
        return self.film_all(ctx)

    def forward(self, obs, hist):
        enc = self.encode(hist)
        return self.base(obs, self.film_params(enc)), enc

    # ------------------------------------------------------------------ auxiliary objectives
    def aux_losses(self, enc, o_t, a_t, robot_next, obj_delta, wm_valid, mass, loaded_soft, loaded_hard,
                   lbl_valid, grl_active):
        c = self.cfg
        L, logs = {}, {}
        wm_mask = wm_valid.float()
        lbl_mask = lbl_valid.float()
        nwm = wm_mask.sum().clamp(min=1.0)
        robot_cur = o_t[:, :self.ROBOT_DIM]

        def l1(pred, tgt, mask, n):
            return ((pred - tgt).abs().mean(-1) * mask).sum() / n

        def obj_wm_loss(pred):
            lp = l1(pred[:, :3], obj_delta[:, :3], wm_mask, nwm)
            lr = l1(pred[:, 3:], obj_delta[:, 3:], wm_mask, nwm)
            return c.lambda_p * lp + c.lambda_R * lr

        # ---- supervised mass / loaded state
        if self.use_mass:
            w_m = loaded_hard * lbl_mask
            L['mass'] = c.w_mass * (F.smooth_l1_loss(enc['mass'], mass, reduction='none') * w_m).sum() / w_m.sum().clamp(min=1.0)
            L['loaded'] = c.w_loaded * (F.binary_cross_entropy_with_logits(enc['loaded_logit'], loaded_soft, reduction='none') * lbl_mask).sum() / lbl_mask.sum().clamp(min=1.0)
            m_est = enc['m_eff'].detach().unsqueeze(-1)
            with torch.no_grad():
                logs['mass_mae_loaded'] = ((enc['mass'] - mass).abs() * w_m).sum() / w_m.sum().clamp(min=1.0)
                logs['loaded_acc'] = (((enc['loaded_p'] > 0.5).float() == loaded_hard).float() * lbl_mask).sum() / lbl_mask.sum().clamp(min=1.0)

        if self.split:
            z_obj, z_dyn = enc['z_obj'], enc['z_dyn']
            mhat = enc['mass'].detach().unsqueeze(-1)
            lhat = enc['loaded_p'].detach().unsqueeze(-1)
            pred_obj = self.obj_wm(torch.cat([o_t, a_t, mhat, lhat, z_obj], -1))
            pred_rob = robot_cur + self.dyn_wm(torch.cat([o_t, a_t, z_dyn, m_est], -1))
            L['obj_wm'] = c.w_obj_wm * obj_wm_loss(pred_obj)
            L['dyn_wm'] = c.w_dyn_wm * l1(pred_rob, robot_next, wm_mask, nwm)
            logs['obj_wm_raw'] = L['obj_wm'].detach() / c.w_obj_wm
            logs['dyn_wm_raw'] = L['dyn_wm'].detach() / c.w_dyn_wm
            # cross adversaries; the adversaries always learn (to keep their probe meaningful);
            # the reversed gradient reaches the encoders only once warm-up is over
            coef = c.grl_coef if (self.use_grl and grl_active) else 0.0
            adv_obj = self.adv_obj_from_dyn(torch.cat([o_t, a_t, grl(z_dyn, coef), m_est], -1))
            adv_rob = robot_cur + self.adv_robot_from_obj(torch.cat([o_t, a_t, grl(z_obj, coef), m_est], -1))
            l_adv = obj_wm_loss(adv_obj) + l1(adv_rob, robot_next, wm_mask, nwm)
            L['grl'] = (c.w_grl if self.use_grl else 1e-3) * l_adv
            logs['adv_obj_raw'] = obj_wm_loss(adv_obj).detach()
            logs['adv_rob_raw'] = l1(adv_rob, robot_next, wm_mask, nwm).detach()
            if not self.use_grl:
                # train the probe adversaries only (no gradient into the latents)
                adv_obj_d = self.adv_obj_from_dyn(torch.cat([o_t, a_t, z_dyn.detach(), m_est], -1))
                adv_rob_d = robot_cur + self.adv_robot_from_obj(torch.cat([o_t, a_t, z_obj.detach(), m_est], -1))
                L['grl'] = obj_wm_loss(adv_obj_d) + l1(adv_rob_d, robot_next, wm_mask, nwm)
            # regularisers
            L['l1'] = c.w_l1_obj * z_obj.abs().sum(-1).mean() + c.w_l1_dyn * z_dyn.abs().sum(-1).mean()
            L['kl'] = c.w_kl * 0.5 * (z_obj.pow(2).sum(-1).mean() + z_dyn.pow(2).sum(-1).mean())
        else:
            z = enc['z']
            inp = [o_t, a_t, z] + ([m_est] if self.use_mass else [])
            out = self.uni_wm(torch.cat(inp, -1))
            pred_rob = robot_cur + out[:, :self.ROBOT_DIM]
            pred_obj = out[:, self.ROBOT_DIM:]
            L['obj_wm'] = c.w_obj_wm * obj_wm_loss(pred_obj)
            L['dyn_wm'] = c.w_dyn_wm * l1(pred_rob, robot_next, wm_mask, nwm)
            logs['obj_wm_raw'] = L['obj_wm'].detach() / c.w_obj_wm
            logs['dyn_wm_raw'] = L['dyn_wm'].detach() / c.w_dyn_wm
            L['l1'] = c.w_l1_obj * z.abs().sum(-1).mean()
            L['kl'] = c.w_kl * 0.5 * z.pow(2).sum(-1).mean()
        return L, logs

    # parameter groups
    def aux_parameters(self):
        film_ids = {id(p) for n, p in self.named_parameters() if n.startswith('film')}
        return [p for n, p in self.named_parameters()
                if p.requires_grad and id(p) not in film_ids and not n.startswith('base.')]

    def film_parameters(self):
        return [p for n, p in self.named_parameters() if n.startswith('film') and p.requires_grad]


class AdapterActorCritic(nn.Module):
    """Frozen base + adapter (actor) and a trainable critic initialised from the base critic."""
    is_recurrent = False

    def __init__(self, base_ckpt_state, adapter_cfg, critic_obs_dim, obs_dim=123, hist_len=50):
        super().__init__()
        actor_sd = {k: v for k, v in base_ckpt_state.items() if k.startswith('actor.')}
        base = FrozenBasePolicy(actor_sd)
        self.adapter = SplitAdapterModel(base, adapter_cfg, obs_dim, hist_len)
        crit = [base_ckpt_state[f'critic.{i}.weight'] for i in (0, 2, 4, 6)]
        self.critic = nn.Sequential(nn.Linear(crit[0].shape[1], crit[0].shape[0]), nn.ELU(),
                                    nn.Linear(crit[1].shape[1], crit[1].shape[0]), nn.ELU(),
                                    nn.Linear(crit[2].shape[1], crit[2].shape[0]), nn.ELU(),
                                    nn.Linear(crit[3].shape[1], crit[3].shape[0]))
        for j, i in enumerate((0, 2, 4, 6)):
            self.critic[2 * j].weight.data.copy_(base_ckpt_state[f'critic.{i}.weight'])
            self.critic[2 * j].bias.data.copy_(base_ckpt_state[f'critic.{i}.bias'])
        self.log_std = nn.Parameter(torch.log(base_ckpt_state['std'].clone().clamp(min=0.02)))
        self.distribution = None

    def act_dist(self, obs, hist):
        mean, enc = self.adapter(obs, hist)
        self.distribution = torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
        return mean, enc

    def act_inference(self, obs, hist):
        mean, _ = self.adapter(obs, hist)
        return mean

    def evaluate(self, critic_obs):
        return self.critic(critic_obs)
