"""MuJoCo sim-to-sim carry-box environment (single env, numpy), mirroring the PhysHSI/Isaac Gym env.

Observation pipeline replicated from PhysHSI `carrybox.py`:
  proprio (108) = [pelvis ang vel*0.25, projected gravity, q - q_default, dq*0.05, 5 end-effectors (pelvis frame), action]
  task (15)     = [box pos (pelvis frame), box rot 6D (tan/norm), box size, goal pos (pelvis frame)]
  with PhysHSI observation noise and the AprilTag/camera visibility model for box perception.
Actor input = 6-step history (738); adapter input = 50-step history of the 123-D frame.
"""
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G1_DIR = os.path.join(ROOT, 'third_party', 'PhysHSI', 'legged_gym', 'resources', 'robots', 'g1')

JOINTS = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint',
          'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint',
          'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
          'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
          'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint',
          'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint',
          'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
DEFAULT_Q = {'left_hip_pitch_joint': -0.1, 'left_knee_joint': 0.3, 'left_ankle_pitch_joint': -0.2,
             'right_hip_pitch_joint': -0.1, 'right_knee_joint': 0.3, 'right_ankle_pitch_joint': -0.2,
             'left_shoulder_roll_joint': 0.1, 'left_elbow_joint': 1.2, 'right_shoulder_roll_joint': -0.1,
             'right_elbow_joint': 1.2}
KP = {'hip_yaw': 150, 'hip_roll': 150, 'hip_pitch': 150, 'knee': 300, 'ankle': 40, 'waist_yaw': 300, 'waist_roll': 300,
      'waist_pitch': 300, 'shoulder': 200, 'elbow': 100, 'wrist': 20}
KD = {'hip_yaw': 2, 'hip_roll': 2, 'hip_pitch': 2, 'knee': 4, 'ankle': 1, 'waist_yaw': 4, 'waist_roll': 4,
      'waist_pitch': 4, 'shoulder': 3, 'elbow': 1, 'wrist': 0.5}


# ----------------------------------------------------------------------------- quaternion helpers (xyzw)
def qmul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                     w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def qrot(q, v):
    qv = q[:3]
    t = 2.0 * np.cross(qv, v)
    return v + q[3] * t + np.cross(qv, t)


def qrot_inv(q, v):
    return qrot(qconj(q), v)


def wxyz2xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


def xyzw2wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def q_from_axis_angle(axis, ang):
    s = np.sin(ang / 2)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(ang / 2)])


def tan_norm(q):
    return np.concatenate([qrot(q, np.array([1.0, 0, 0])), qrot(q, np.array([0, 0, 1.0]))])


def roll_pitch(q):
    x, y, z, w = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return roll, pitch


# ----------------------------------------------------------------------------- model
def build_xml(box_size, timestep=0.002, robot_friction=0.8, box_friction=0.8, floor_friction=1.0):
    raw = os.path.join(ROOT, 'sim2sim', 'assets', 'g1_29dof_raw.xml')
    if not os.path.exists(raw):
        src = open(os.path.join(G1_DIR, 'urdf', 'g1_29dof.urdf')).read()
        src = re.sub(r'<mujoco>.*?</mujoco>', '<mujoco><compiler meshdir="../meshes" balanceinertia="true" '
                     'discardvisual="false" fusestatic="false" strippath="false"/></mujoco>', src, flags=re.S)
        tmp = os.path.join(G1_DIR, 'urdf', '_g1_mj.urdf')
        open(tmp, 'w').write(src)
        m = mujoco.MjModel.from_xml_path(tmp)
        os.makedirs(os.path.dirname(raw), exist_ok=True)
        mujoco.mj_saveLastXML(raw, m)
    tree = ET.parse(raw)
    root = tree.getroot()
    comp = root.find('compiler')
    comp.set('meshdir', os.path.join(G1_DIR, 'meshes'))
    for k in ['strippath']:
        comp.attrib.pop(k, None)
    for asset_mesh in root.find('asset').findall('mesh'):
        asset_mesh.set('file', os.path.basename(asset_mesh.get('file')))
    opt = ET.SubElement(root, 'option', timestep=str(timestep), integrator='implicitfast', cone='pyramidal')
    ET.SubElement(opt, 'flag', warmstart='enable')
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'joint', armature='0.01', damping='0.0', frictionloss='0.0')
    ET.SubElement(default, 'geom', friction=f'{robot_friction} 0.005 0.0001')
    wb = root.find('worldbody')
    pelvis = wb.find('body')
    pelvis.set('pos', '0 0 0.8')
    pelvis.insert(0, ET.Element('freejoint', name='root'))
    ET.SubElement(wb, 'light', pos='0 0 5', dir='0 0 -1', directional='true')
    # visual-only assets (sky + checker floor); they do not change the physics
    asset = root.find('asset')
    ET.SubElement(asset, 'texture', name='sky', type='skybox', builtin='gradient', rgb1='0.92 0.94 0.97',
                  rgb2='0.62 0.70 0.80', width='512', height='512')
    ET.SubElement(asset, 'texture', name='grid', type='2d', builtin='checker', rgb1='0.86 0.86 0.84',
                  rgb2='0.76 0.76 0.74', width='512', height='512')
    ET.SubElement(asset, 'material', name='grid', texture='grid', texrepeat='8 8', reflectance='0.05')
    ET.SubElement(wb, 'geom', name='floor', type='plane', size='20 20 1', friction=f'{floor_friction} 0.005 0.0001',
                  material='grid')
    hx, hy, hz = [s / 2 for s in box_size]
    box = ET.SubElement(wb, 'body', name='box', pos='2 0 0.2')
    ET.SubElement(box, 'freejoint', name='box_root')
    ET.SubElement(box, 'geom', name='box_geom', type='box', size=f'{hx} {hy} {hz}', mass='2.0',
                  friction=f'{box_friction} 0.005 0.0001', rgba='0.8 0.5 0.2 1')
    for name, rgba in [('platform', '0.5 0.235 0.6 1'), ('tar_platform', '0.0 0.0 0.8 1')]:
        b = ET.SubElement(wb, 'body', name=name, mocap='true', pos='0 0 -5')
        ET.SubElement(b, 'geom', type='box', size='0.2 0.2 0.01', rgba=rgba, friction=f'{floor_friction} 0.005 0.0001')
    return ET.tostring(root, encoding='unicode')


class MujocoCarryBox:
    def __init__(self, box_size=(0.30, 0.43, 0.30), timestep=0.002, decimation=10, add_noise=True,
                 episode_length_s=20.0, hist_len=50, actor_hist=6, motor_strength=1.0, head_term=0.45, **fric):
        self.box_size = np.array(box_size, dtype=np.float64)
        self.model = mujoco.MjModel.from_xml_string(build_xml(box_size, timestep, **fric))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.decimation = decimation
        self.dt = timestep * decimation
        self.max_steps = int(np.ceil(episode_length_s / self.dt))
        self.add_noise = add_noise
        self.hist_len, self.actor_hist = hist_len, actor_hist
        self.motor_strength = motor_strength
        self.head_term = head_term
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.b_pelvis, self.b_box = bid('pelvis'), bid('box')
        self.b_hands = [bid('left_palm_link'), bid('right_palm_link')]
        self.b_feet = [bid('left_ankle_pitch_link'), bid('right_ankle_pitch_link')]
        self.b_head, self.b_cam = bid('mid360_link'), bid('d455_link')
        self.b_hip_yaw = [bid('left_hip_yaw_link'), bid('right_hip_yaw_link')]
        self.b_rubber = {bid('left_rubber_hand'), bid('right_rubber_hand')}
        self.mocap_plat = m.body_mocapid[bid('platform')]
        self.mocap_tar = m.body_mocapid[bid('tar_platform')]
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in JOINTS]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.vadr = np.array([m.jnt_dofadr[j] for j in jid])
        self.box_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'box_root')]
        self.box_vadr = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'box_root')]
        self.q_default = np.array([DEFAULT_Q.get(j, 0.0) for j in JOINTS])
        self.kp = np.array([next(v for k, v in KP.items() if k in j) for j in JOINTS], dtype=np.float64)
        self.kd = np.array([next(v for k, v in KD.items() if k in j) for j in JOINTS], dtype=np.float64)
        self.tau_lim = np.array([m.jnt_actfrcrange[j][1] if m.jnt_actfrclimited[j] else 0 for j in jid])
        if np.any(self.tau_lim == 0):
            # URDF effort limits are stored by the URDF importer as actuator-free joint limits; read them from the URDF
            urdf = open(os.path.join(G1_DIR, 'urdf', 'g1_29dof.urdf')).read()
            for i, j in enumerate(JOINTS):
                mm = re.search(r'<joint name="%s".*?effort="([0-9.]+)"' % j, urdf, flags=re.S)
                self.tau_lim[i] = float(mm.group(1))
        ns = np.zeros(108)
        ns[0:3] = 0.3 * 0.25
        ns[3:6] = 0.05
        ns[6:35] = 0.02
        ns[35:64] = 2.0 * 0.05
        ns[64:79] = 0.05
        self.noise_vec = ns

    # ------------------------------------------------------------------
    def set_box_mass(self, mass):
        m = self.model
        sx, sy, sz = self.box_size
        m.body_mass[self.b_box] = mass
        m.body_inertia[self.b_box] = mass / 12.0 * np.array([sy ** 2 + sz ** 2, sx ** 2 + sz ** 2, sx ** 2 + sy ** 2])
        self.box_mass = mass

    def reset(self, mass, pick_h, place_h, seed):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.set_box_mass(mass)
        d, m = self.data, self.model
        mujoco.mj_resetData(m, d)
        yaw0 = 0.0
        d.qpos[0:3] = [0, 0, 0.8]
        d.qpos[3:7] = [np.cos(yaw0 / 2), 0, 0, np.sin(yaw0 / 2)]
        d.qpos[self.qadr] = self.q_default
        fwd = np.array([np.cos(yaw0), np.sin(yaw0)])
        lat = np.array([-fwd[1], fwd[0]])
        box_xy = d.qpos[0:2] + 2.0 * fwd + rng.uniform(-0.2, 0.2) * lat
        byaw = rng.uniform(0, 2 * np.pi)
        hz = self.box_size[2] / 2
        d.qpos[self.box_qadr:self.box_qadr + 3] = [box_xy[0], box_xy[1], pick_h + hz + 0.01]
        d.qpos[self.box_qadr + 3:self.box_qadr + 7] = [np.cos(byaw / 2), 0, 0, np.sin(byaw / 2)]
        d.mocap_pos[self.mocap_plat] = [box_xy[0], box_xy[1], pick_h - 0.01]
        dir_to_robot = d.qpos[0:2] - box_xy
        base = np.arctan2(dir_to_robot[1], dir_to_robot[0])
        ang = base + rng.choice([-1, 1]) * np.deg2rad(rng.uniform(10, 80))
        dist = rng.uniform(2.5, 4.0)
        goal_xy = box_xy + dist * np.array([np.cos(ang), np.sin(ang)])
        self.goal = np.array([goal_xy[0], goal_xy[1], place_h + hz + 0.02])
        d.mocap_pos[self.mocap_tar] = [goal_xy[0], goal_xy[1], place_h + 0.01]
        self.plat_top = pick_h
        self.tar_top = place_h + 0.02
        self.plat_xy, self.tar_xy = box_xy.copy(), goal_xy.copy()
        mujoco.mj_forward(m, d)
        # perception randomisation (PhysHSI)
        self.thresh_tag = rng.uniform(0.7, 2.0)
        self.far_off = rng.uniform(-0.2, 0.2, 3) * np.array([1, 1, 2.0])
        self.hfov, self.vfov = np.deg2rad(rng.uniform(85, 90)), np.deg2rad(rng.uniform(55, 60))
        self.facing = rng.uniform(np.cos(np.deg2rad(70)), np.cos(np.deg2rad(50)))
        self.has_seen = False
        self.actions = np.zeros(29)
        self.obs_buf = np.zeros(self.actor_hist * 123)
        self.hist = np.zeros((self.hist_len, 123))
        self.t = 0
        self.success = False
        self.ever_success = False
        self.success_time = -1.0
        self.lift_counter = 0
        self.lifted = False
        self.fell = False
        self.loaded_soft = 0.0
        self.trace = []
        # PhysHSI: reset is followed by one zero-action step
        return self.step(np.zeros(29))[0:2]

    # ------------------------------------------------------------------
    def _state(self):
        d = self.data
        pq = wxyz2xyzw(d.xquat[self.b_pelvis])
        ppos = d.xpos[self.b_pelvis].copy()
        ang_world = d.cvel[self.b_pelvis][0:3]  # world-frame angular velocity (cvel = [ang, lin])
        ang_local = qrot_inv(pq, ang_world)
        grav = qrot_inv(pq, np.array([0, 0, -1.0]))
        q = d.qpos[self.qadr]
        dq = d.qvel[self.vadr]
        ee = []
        for b in self.b_hands + self.b_feet + [self.b_head]:
            ee.append(qrot_inv(pq, d.xpos[b] - ppos))
        box_pos = d.qpos[self.box_qadr:self.box_qadr + 3].copy()
        box_q = wxyz2xyzw(d.qpos[self.box_qadr + 3:self.box_qadr + 7])
        return dict(pq=pq, ppos=ppos, ang=ang_local, grav=grav, q=q, dq=dq, ee=np.concatenate(ee),
                    box_pos=box_pos, box_q=box_q)

    def _hand_contact(self):
        d, m = self.data, self.model
        f6 = np.zeros(6)
        force = {b: np.zeros(3) for b in self.b_rubber}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 in self.b_rubber or b2 in self.b_rubber:
                mujoco.mj_contactForce(m, d, i, f6)
                fw = c.frame.reshape(3, 3).T @ f6[:3]
                if b1 in self.b_rubber:
                    force[b1] -= fw
                if b2 in self.b_rubber:
                    force[b2] += fw
        return any(np.linalg.norm(f) > 1.0 for f in force.values())

    def _can_see_tag(self, s, dist):
        d = self.data
        cam_p = d.xpos[self.b_cam]
        cam_q = wxyz2xyzw(d.xquat[self.b_cam])
        corners = np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]])
        tags = np.array([qrot(s['box_q'], np.array([c[0], c[1], self.box_size[2] / 2])) + s['box_pos'] for c in corners])
        rel = np.array([qrot_inv(cam_q, t - cam_p) for t in tags])
        normal = qrot(s['box_q'], np.array([0, 0, 1.0]))
        vd = cam_p - tags.mean(0)
        vd = vd / max(np.linalg.norm(vd), 1e-9)
        facing = normal @ vd > self.facing
        front = rel[:, 2] > 0.1
        h = np.arctan2(rel[:, 0], rel[:, 2])
        v = np.arctan2(rel[:, 1], rel[:, 2])
        in_view = np.all(front & (np.abs(h) < self.hfov / 2) & (np.abs(v) < self.vfov / 2))
        good = np.linalg.norm(rel.mean(0)) < 2.5
        see = bool(facing and in_view and good)
        if see:
            self.has_seen = True
        if dist >= self.thresh_tag:
            self.has_seen = False
        return see

    def _task_obs(self, s, see, dist):
        rng = self.rng
        box_pos = s['box_pos'] - s['ppos']
        goal_pos = self.goal - s['ppos']
        box_q = s['box_q'].copy()
        if self.add_noise:
            coarse = (dist >= self.thresh_tag) or ((dist < self.thresh_tag) and (not see) and (not self.has_seen))
            mask = (not see) and self.has_seen and dist < 0.65
            if coarse:
                box_pos = box_pos + self.far_off
                box_q = np.array([0, 0, 0, 1.0])
            box_pos = box_pos + rng.uniform(-0.05, 0.05, 3)
            bl = qrot_inv(s['pq'], box_pos)
            vec = rng.uniform(0, 1, 3)
            axis = vec / np.linalg.norm(vec)
            box_q = qmul(box_q, q_from_axis_angle(axis, rng.uniform(-np.deg2rad(5), np.deg2rad(5))))
            ql = qmul(qconj(s['pq']), box_q)
            if mask:
                bl = np.zeros(3)
                ql = np.array([0, 0, 0, 1.0])
            goal_pos = goal_pos + rng.uniform(-0.05, 0.05, 3)
        else:
            bl = qrot_inv(s['pq'], box_pos)
            ql = qmul(qconj(s['pq']), box_q)
        task = np.concatenate([bl, tan_norm(ql), self.box_size, qrot_inv(s['pq'], goal_pos)])
        if self.success:
            task[:] = -1.0
        return task

    def step(self, action):
        d, m = self.data, self.model
        a = np.clip(np.asarray(action, dtype=np.float64), -100, 100)
        self.actions = a
        target = self.q_default + 0.25 * a
        for _ in range(self.decimation):
            q, dq = d.qpos[self.qadr], d.qvel[self.vadr]
            tau = (self.kp * (target - q) - self.kd * dq) * self.motor_strength
            d.qfrc_applied[self.vadr] = np.clip(tau, -self.tau_lim, self.tau_lim)
            mujoco.mj_step(m, d)
        self.t += 1
        s = self._state()
        dist = np.linalg.norm(s['box_pos'][:2] - s['ppos'][:2])
        see = self._can_see_tag(s, dist)
        # termination (PhysHSI check_termination)
        roll, pitch = roll_pitch(s['pq'])
        box_grav = qrot_inv(s['box_q'], np.array([0, 0, -1.0]))
        box_v = d.qvel[self.box_vadr:self.box_vadr + 2]
        fell = (d.xpos[self.b_head][2] < self.head_term or s['ppos'][2] < 0.2 or abs(roll) > 0.5 or abs(pitch) > 1.1 or
                np.linalg.norm(box_v) > 3.0 or min(d.xpos[b][2] for b in self.b_hip_yaw) < 0.15 or box_grav[2] > -0.05)
        timeout = self.t > self.max_steps
        # success (test-mode criterion)
        self.success = bool(np.linalg.norm(box_grav[:2]) < 0.2 and np.linalg.norm(s['box_pos'] - self.goal) < 0.1)
        if self.success and not self.ever_success:
            self.ever_success = True
            self.success_time = self.t * self.dt
        # loaded label / Lift-up
        support = 0.0
        for xy, top in ((self.plat_xy, self.plat_top), (self.tar_xy, self.tar_top)):
            if np.linalg.norm(s['box_pos'][:2] - xy) < 0.3:
                support = max(support, top)
        above = s['box_pos'][2] - self.box_size[2] / 2 - support
        hc = self._hand_contact()
        near = dist < 0.8
        hard = above > 0.03 and hc and near
        self.loaded_soft = float(np.clip(above / 0.05, 0, 1) * (hc and near))
        self.loaded_hard = float(hard)
        self.lift_counter = self.lift_counter + 1 if hard else 0
        if self.lift_counter >= 25:
            self.lifted = True
        # observations
        prop = np.concatenate([s['ang'] * 0.25, s['grav'], s['q'] - self.q_default, s['dq'] * 0.05, s['ee'], a])
        if self.add_noise:
            prop = prop + (2 * self.rng.random(108) - 1) * self.noise_vec
        frame = np.concatenate([prop, self._task_obs(s, see, dist)])
        self.obs_buf = np.concatenate([self.obs_buf[123:], frame])
        self.hist = np.concatenate([self.hist[1:], frame[None]], 0)
        done = fell or timeout
        if fell and not self.ever_success:
            self.fell = True
        info = dict(box_pos=s['box_pos'], robot_pos=s['ppos'], goal=self.goal, loaded=self.loaded_soft)
        return self.obs_buf.astype(np.float32), self.hist.astype(np.float32), done, info
