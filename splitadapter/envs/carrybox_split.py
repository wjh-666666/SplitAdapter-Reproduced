"""PhysHSI G1 carry-box environment extended for SplitAdapter.

Additions on top of the PhysHSI environment (the frozen base policy's training env):
  * 50-step adaptation history of the 123-D one-step actor observation (Appendix A.1)
  * split world-model targets:  robot transition (79-D clean robot state at t+1) and
    object delta transition (box linear / angular velocity expressed in the pelvis frame at t)
  * supervised labels: box mass and loaded state (hard + soft)
  * paper's box randomisation (size / mass / height ranges) and object dynamics randomisation
  * deterministic evaluation scenarios (mass x pickup height x placement height) with
    Lift-up / Full-task success bookkeeping
"""
import numpy as np
import torch

from isaacgym import gymapi
from isaacgym.torch_utils import (quat_mul, quat_conjugate, quat_rotate_inverse, quat_apply,
                                  quat_from_angle_axis, torch_rand_float)

from legged_gym.envs.g1.carrybox import LeggedRobot as PhysHSICarryBox


def quat_to_axis_angle(q):
    """q: (N,4) xyzw -> (N,3) axis-angle, shortest rotation."""
    q = torch.where(q[:, 3:4] < 0, -q, q)
    xyz = q[:, :3]
    s = torch.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s, q[:, 3:4].clamp(min=-1, max=1))
    axis = xyz / s.clamp(min=1e-8)
    return axis * angle


class SplitCarryBox(PhysHSICarryBox):

    # ------------------------------------------------------------------ init
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.ad_cfg = cfg.adapter
        self.ev_cfg = cfg.evaluation
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        N, H, D = self.num_envs, self.ad_cfg.history_length, self.ad_cfg.one_step_obs
        self.adapt_hist = torch.zeros(N, H, D, device=self.device)
        self.hist_count = torch.zeros(N, dtype=torch.long, device=self.device)  # valid frames in adapt_hist
        # world-model bookkeeping
        self._wm_prev_box_pos = self.box_states[:, 0:3].clone()
        self._wm_prev_box_quat = self.box_states[:, 3:7].clone()
        self._wm_prev_frame_quat = self.rigid_body_states[:, self.upper_body_index, 3:7].clone()
        self.wm_robot_next = torch.zeros(N, self.ad_cfg.robot_state_dim, device=self.device)
        self.wm_obj_delta = torch.zeros(N, self.ad_cfg.obj_delta_dim, device=self.device)
        self.wm_valid = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._first_step = torch.ones(N, dtype=torch.bool, device=self.device)
        # labels describing the *current* observation
        self.lbl_loaded_soft = torch.zeros(N, device=self.device)
        self.lbl_loaded_hard = torch.zeros(N, device=self.device)
        self.lbl_valid = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._box_bottom_above_support = torch.zeros(N, device=self.device)
        # evaluation bookkeeping
        self.ev_lift_counter = torch.zeros(N, dtype=torch.long, device=self.device)
        self.ev_lifted = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.ev_success = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.ev_finished = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.ev_fell = torch.zeros(N, dtype=torch.bool, device=self.device)
        self.ev_success_time = torch.full((N,), -1.0, device=self.device)

    # ------------------------------------------------------------------ box assets
    def _load_box_asset(self):
        N = self.num_envs
        bc = self.box_cfg
        gen = torch.Generator(device='cpu')
        gen.manual_seed(int(self.ev_cfg.seed) + 12345)
        if self.ev_cfg.enabled:
            size = torch.tensor(self.ev_cfg.box_size, dtype=torch.float).reshape(1, 3).repeat(N, 1)
            mass = torch.as_tensor(np.asarray(self.ev_cfg.masses, dtype=np.float32))
            assert mass.shape[0] == N
        else:
            lo = torch.tensor([bc.size_range_x[0], bc.size_range_y[0], bc.size_range_z[0]])
            hi = torch.tensor([bc.size_range_x[1], bc.size_range_y[1], bc.size_range_z[1]])
            size = lo + (hi - lo) * torch.rand(N, 3, generator=gen)
            mass = bc.mass_range[0] + (bc.mass_range[1] - bc.mass_range[0]) * torch.rand(N, generator=gen)
        self._box_size = size.to(self.device)
        self._box_scale = self._box_size / torch.tensor(bc.base_size, device=self.device).reshape(1, 3)
        volume = self._box_size.prod(dim=-1)
        self._box_density = mass.to(self.device) / volume
        self.box_mass_target = mass.to(self.device)

        corner_xy = torch.tensor([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]],
                                 dtype=torch.float, device=self.device)
        self.tag_pos_local = torch.zeros(N, 4, 3, dtype=torch.float, device=self.device)
        self.tag_pos_local[:, :, :2] = corner_xy.unsqueeze(0).expand(N, -1, -1)
        self.tag_pos_local[:, :, 2] = (self._box_size[:, 2] / 2.0).unsqueeze(-1).expand(-1, 4)

        self.box_assets = []
        for i in range(N):
            opt = gymapi.AssetOptions()
            opt.density = float(self._box_density[i])
            opt.angular_damping = 0.01
            opt.linear_damping = 0.01
            opt.max_angular_velocity = 100.0
            opt.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            self.box_assets.append(self.gym.create_box(self.sim, float(self._box_size[i, 0]),
                                                       float(self._box_size[i, 1]),
                                                       float(self._box_size[i, 2]), opt))

    # ------------------------------------------------------------------ resets
    def _reset_default(self, env_ids):
        super()._reset_default(env_ids)
        if self.ev_cfg.enabled:
            self.root_states[env_ids, 7:13] = 0.0
            self.dof_pos[env_ids] = self.default_dof_pos
            self.dof_vel[env_ids] = 0.0

    def _reset_actors(self, env_ids):
        if self.ev_cfg.enabled:
            self._reset_default(env_ids)
        else:
            super()._reset_actors(env_ids)

    def _reset_boxes(self, env_ids):
        if not self.ev_cfg.enabled:
            super()._reset_boxes(env_ids)
            # resample pickup heights in the paper range for randomly placed boxes
            rand_ids = self._random_box_env_ids()
            if rand_ids is not None and len(rand_ids) > 0:
                h = torch_rand_float(*self.box_cfg.pickup_height_range, (len(rand_ids), 1), device=self.device).squeeze(1)
                self._place_box_on_platform(rand_ids, self.box_states[rand_ids, 0:2], h, None)
            return
        self._eval_reset_boxes(env_ids)

    def _random_box_env_ids(self):
        ids = []
        if len(self._reset_default_env_ids) > 0:
            ids.append(self._reset_default_env_ids)
        if self._reset_ref_env_ids.get('loco') is not None:
            ids.append(self._reset_ref_env_ids['loco'])
        return torch.cat(ids) if len(ids) > 0 else None

    def _place_box_on_platform(self, ids, xy, h, yaw):
        """Box bottom at height h (h=0: on the ground) resting on the (thin, fixed) start platform."""
        half_z = self._box_size[ids, 2] / 2
        self.box_states[ids, 0:2] = xy
        self.box_states[ids, 2] = h + half_z + 0.01
        if yaw is not None:
            axis = self.z_axis_unit.expand(len(ids), -1)
            self.box_states[ids, 3:7] = quat_from_angle_axis(yaw, axis)
        self.box_states[ids, 7:13] = 0.0
        self.platform_pos[ids, 0:2] = xy
        self.platform_pos[ids, 2] = h - self._platform_height / 2

    def _eval_reset_boxes(self, env_ids):
        n = len(env_ids)
        g = self._eval_gen
        root_xy = self.root_states[env_ids, 0:2]
        fwd = quat_apply(self.root_states[env_ids, 3:7], self.forward_vec[env_ids])[:, 0:2]
        fwd = fwd / fwd.norm(dim=-1, keepdim=True)
        lat = torch.stack([-fwd[:, 1], fwd[:, 0]], dim=-1)
        lat_off = (torch.rand(n, 1, generator=g, device=self.device) * 2 - 1) * self.ev_cfg.lateral_noise
        xy = root_xy + self.ev_cfg.box_distance * fwd + lat_off * lat
        yaw = torch.rand(n, generator=g, device=self.device) * 2 * np.pi
        self._place_box_on_platform(env_ids, xy, self.ev_pick_h[env_ids], yaw)
        # perception-related buffers (same as PhysHSI)
        self.thresh_tag[env_ids] = torch_rand_float(self.box_cfg.thresh_tag[0], self.box_cfg.thresh_tag[1], (n, 1), device=self.device).squeeze(1)
        self.far_pos_offset[env_ids] = torch_rand_float(-self.box_cfg.far_pos_offset, self.box_cfg.far_pos_offset, (n, 3), device=self.device)
        self.far_pos_offset[env_ids, 2] *= 2.0
        self.has_seen_tag[env_ids] = False

    def _set_goal(self, ids, goal_xy, h):
        goal_z = h + self._box_size[ids, 2] / 2 + self._platform_height
        self.goal_pos[ids, 0:2] = goal_xy
        self.goal_pos[ids, 2] = goal_z
        self.tar_platform_pos[ids, 0:2] = goal_xy
        self.tar_platform_pos[ids, 2] = goal_z - self._box_size[ids, 2] / 2 - self._platform_height / 2

    def _reset_task(self, env_ids):
        if not self.ev_cfg.enabled:
            super()._reset_task(env_ids)
            # resample placement heights in the paper range (except putDown RSI which comes from mocap)
            ids = [self._reset_default_env_ids] if len(self._reset_default_env_ids) > 0 else []
            for sk in ['loco', 'pickUp', 'carryWith']:
                if self._reset_ref_env_ids.get(sk) is not None:
                    ids.append(self._reset_ref_env_ids[sk])
            if len(ids) > 0:
                ids = torch.cat(ids)
                h = torch_rand_float(*self.box_cfg.place_height_range, (len(ids), 1), device=self.device).squeeze(1)
                self._set_goal(ids, self.goal_pos[ids, 0:2].clone(), h)
            return
        n = len(env_ids)
        g = self._eval_gen
        dir_to_robot = self.root_states[env_ids, 0:2] - self.box_states[env_ids, 0:2]
        base = torch.atan2(dir_to_robot[:, 1], dir_to_robot[:, 0])
        lo, hi = self.ev_cfg.goal_angle_offset_deg
        off = (lo + (hi - lo) * torch.rand(n, generator=g, device=self.device)) * np.pi / 180.0
        sign = torch.where(torch.rand(n, generator=g, device=self.device) < 0.5, -1.0, 1.0)
        ang = base + sign * off
        dlo, dhi = self.ev_cfg.goal_distance_range
        dist = dlo + (dhi - dlo) * torch.rand(n, generator=g, device=self.device)
        goal_xy = self.box_states[env_ids, 0:2] + dist.unsqueeze(-1) * torch.stack([torch.cos(ang), torch.sin(ang)], -1)
        self._set_goal(env_ids, goal_xy, self.ev_place_h[env_ids])

    def _init_buffers(self):
        super()._init_buffers()
        if self.ev_cfg.enabled:
            self.ev_pick_h = torch.as_tensor(np.asarray(self.ev_cfg.pickup_heights, dtype=np.float32), device=self.device)
            self.ev_place_h = torch.as_tensor(np.asarray(self.ev_cfg.place_heights, dtype=np.float32), device=self.device)
            self._eval_gen = torch.Generator(device=self.device)
            self._eval_gen.manual_seed(int(self.ev_cfg.seed))

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.ev_cfg.enabled and hasattr(self, 'ev_finished'):
            # an episode ended for these envs (fall or timeout); freeze their evaluation result
            fell = ~self.time_out_buf[env_ids]
            newly = (~self.ev_finished[env_ids]) & (self.episode_length_buf[env_ids] > 1)
            env_ids_fin = env_ids[newly]
            self.ev_fell[env_ids_fin] |= fell[newly] & ~self.ev_success[env_ids_fin]
            self.ev_finished[env_ids_fin] = True
        super().reset_idx(env_ids)
        if hasattr(self, 'adapt_hist'):
            self._first_step[env_ids] = True
            self.ev_lift_counter[env_ids] = 0

    def check_termination(self):
        super().check_termination()
        if self.ev_cfg.enabled:
            # same as PhysHSI except the head-height threshold (evaluation only)
            rb = self.rigid_body_states
            r = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 10., dim=1)
            r |= self.time_out_buf
            r |= rb[:, self.head_index, 2] < self.ev_cfg.head_height_term
            r |= self.root_states[:, 2] < 0.2
            r |= torch.logical_or(torch.abs(self.roll) > 0.5, torch.abs(self.pitch) > 1.1)
            r |= torch.norm(rb[:, 2, 7:9], dim=-1) > 3.0
            r |= torch.any(rb[:, self.hip_yaw_indices, 2] < 0.15, dim=1)
            if self.box_cfg.box_termination:
                r |= (self.projected_gravity_box[:, 2] > -0.05)
            self.reset_buf = r

    # ------------------------------------------------------------------ per-step hooks
    def _clean_robot_state(self):
        return torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                          self.projected_gravity,
                          (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                          self.dof_vel * self.obs_scales.dof_vel,
                          self.end_effector_pos), dim=-1)

    def _support_top(self):
        box_xy = self.box_states[:, 0:2]
        top = torch.zeros(self.num_envs, device=self.device)
        for plat in (self.platform_pos, self.tar_platform_pos):
            on = torch.norm(box_xy - plat[:, 0:2], dim=-1) < 0.3
            plat_top = plat[:, 2] + self._platform_height / 2
            top = torch.where(on, torch.maximum(top, plat_top), top)
        return top

    def compute_amp_observations(self):
        """Called after physics/reward and *before* resets -> transition targets for (o_t, a_t)."""
        if hasattr(self, 'adapt_hist'):
            # ---- robot transition target (clean, no observation noise)
            self.wm_robot_next[:] = self._clean_robot_state()
            # ---- object delta transition in the pelvis frame at time t (scaled by 1/dt)
            dp = self.box_states[:, 0:3] - self._wm_prev_box_pos
            dp_local = quat_rotate_inverse(self._wm_prev_frame_quat, dp) / self.dt
            dq = quat_mul(self.box_states[:, 3:7], quat_conjugate(self._wm_prev_box_quat))
            dq_local = quat_mul(quat_conjugate(self._wm_prev_frame_quat), quat_mul(dq, self._wm_prev_frame_quat))
            dr_local = quat_to_axis_angle(dq_local) / self.dt
            self.wm_obj_delta[:] = torch.cat((dp_local, dr_local), dim=-1).clamp(-20.0, 20.0)
            self.wm_valid[:] = ~self._first_step
            # ---- loaded labels (state at t+1)
            half_z = self._box_size[:, 2] / 2
            above = self.box_states[:, 2] - half_z - self._support_top()
            self._box_bottom_above_support[:] = above
            hand_contact = (torch.norm(self.contact_forces[:, self.hand_colli_indices], dim=-1) > 1.0).any(dim=1)
            near = self.robot2object_dist < 0.8
            hard = (above > self.ad_cfg.loaded_height_thresh) & hand_contact & near
            soft = torch.clamp(above / self.ad_cfg.loaded_soft_scale, 0.0, 1.0) * (hand_contact & near).float()
            self.lbl_loaded_hard[:] = hard.float()
            self.lbl_loaded_soft[:] = soft
            # ---- evaluation bookkeeping
            if self.ev_cfg.enabled:
                active = ~self.ev_finished
                self.ev_lift_counter = torch.where(hard, self.ev_lift_counter + 1, torch.zeros_like(self.ev_lift_counter))
                self.ev_lifted |= active & (self.ev_lift_counter >= self.ad_cfg.lift_hold_steps)
                new_succ = active & self.success_buf & ~self.ev_success
                self.ev_success_time[new_succ] = self.episode_length_buf[new_succ].float() * self.dt
                self.ev_success |= new_succ
        return super().compute_amp_observations()

    def compute_observations(self):
        super().compute_observations()
        if not hasattr(self, 'adapt_hist'):
            return
        newest = self.obs_buf[:, -self.ad_cfg.one_step_obs:]
        reset = self.reset_buf.bool()
        self.adapt_hist[reset] = 0.0
        self.hist_count[reset] = 0
        self.adapt_hist = torch.cat((self.adapt_hist[:, 1:], newest.unsqueeze(1)), dim=1)
        self.hist_count = torch.clamp(self.hist_count + 1, max=self.ad_cfg.history_length)
        # labels for the current observation are only valid if no reset happened this step
        self.lbl_valid[:] = ~reset
        # previous-state bookkeeping for the next transition
        self._wm_prev_box_pos[:] = self.box_states[:, 0:3]
        self._wm_prev_box_quat[:] = self.box_states[:, 3:7]
        self._wm_prev_frame_quat[:] = self.rigid_body_states[:, self.upper_body_index, 3:7]
        self._first_step[:] = reset

    # ------------------------------------------------------------------ helpers for the runner
    def get_adapter_inputs(self):
        return self.adapt_hist

    def get_labels(self):
        """Labels for the current observation (mass, loaded soft, loaded hard, valid)."""
        return (self.box_mass_target.clone(), self.lbl_loaded_soft.clone(),
                self.lbl_loaded_hard.clone(), self.lbl_valid.clone())

    def get_wm_targets(self):
        return self.wm_robot_next.clone(), self.wm_obj_delta.clone(), self.wm_valid.clone()
