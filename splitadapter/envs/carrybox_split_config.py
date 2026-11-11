"""Config for the SplitAdapter carry-box environment.

Inherits the PhysHSI G1 carry-box config (the frozen base policy was trained with it) and
applies the task / domain-randomisation settings listed in the SplitAdapter paper (Appendix A.2).
"""
import numpy as np
from legged_gym.envs.g1.carrybox_resume_config import G1Cfg, G1CfgPPO  # stage-2 PhysHSI config (released ckpt)


class SplitCarryBoxCfg(G1Cfg):
    class env(G1Cfg.env):
        num_envs = 4096
        episode_length_s = 20

    class asset(G1Cfg.asset):
        class box(G1Cfg.asset.box):
            # --- Paper A.2: box size x,y in [0.25,0.45], z in [0.25,0.35]; mass in [0.5, 5.0] kg
            size_range_x = [0.25, 0.45]
            size_range_y = [0.25, 0.45]
            size_range_z = [0.25, 0.35]
            mass_range = [0.5, 5.0]
            # pickup / placement height above the ground (box bottom), paper: [0, 0.6] m
            pickup_height_range = [0.0, 0.6]
            place_height_range = [0.0, 0.6]
            # Object dynamics randomisation (Table 3)
            random_props = True
            friction_range = [0.5, 1.0]
            restitution_range = [0.05, 0.2]
            platform_friction_range = [0.5, 2.0]

    class domain_rand(G1Cfg.domain_rand):
        # Table 3 of the paper.  The torso payload of PhysHSI is not listed -> disabled
        randomize_payload_mass = False
        actuation_offset_range = [-0.05, 0.05]
        motor_strength_range = [0.9, 1.1]
        com_displacement_range = [-0.1, 0.1]
        link_mass_range = [0.8, 1.2]
        kp_range = [0.9, 1.1]
        kd_range = [0.9, 1.1]
        disturbance = True
        disturbance_interval = 8
        disturbance_range = [-50, 50]

    class sim(G1Cfg.sim):
        class physx(G1Cfg.sim.physx):
            # larger GPU contact buffers than the PhysHSI defaults
            max_gpu_contact_pairs = 2 ** 25
            default_buffer_size_multiplier = 10

    class adapter:
        history_length = 50          # Table 4
        one_step_obs = 123           # 108 proprio + 15 task
        robot_state_dim = 79         # robot transition target (ang vel, gravity, q, dq, end effectors)
        obj_delta_dim = 6            # box linear (3) + angular (3) velocity in the pelvis frame at t
        loaded_height_thresh = 0.03  # box bottom above its support [m] for the hard loaded label
        loaded_soft_scale = 0.05     # soft label = clip(height / scale, 0, 1) * hand_contact
        lift_hold_steps = 25         # 0.5 s of continuous loaded state -> Lift-up success

    class evaluation:
        enabled = False
        # per-env scenario arrays are injected by the evaluation script
        box_size = [0.30, 0.43, 0.30]     # validation box geometry used in the paper
        box_distance = 2.0                # box ~2 m in front of the robot
        lateral_noise = 0.2
        goal_distance_range = [2.5, 4.0]
        goal_angle_offset_deg = [10.0, 80.0]  # measured from the box->robot direction (PhysHSI range)
        masses = None
        pickup_heights = None
        place_heights = None
        seed = 0
        head_height_term = 0.45   # training uses 0.6 (a heuristic the deep floor-pickup squat nearly touches)


class SplitCarryBoxCfgPPO(G1CfgPPO):
    seed = 1

    class policy(G1CfgPPO.policy):
        pass

    class algorithm(G1CfgPPO.algorithm):
        num_learning_epochs = 5
        num_mini_batches = 8
        clip_param = 0.2
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.003
        learning_rate = 1e-5          # RL learning rate (FiLM generators, std, critic) - adaptive
        aux_learning_rate = 1e-3      # Table 4 "auxiliary learning rate"
        entropy_coef = 0.01
        value_loss_coef = 1.0
        max_grad_norm = 1.0
        schedule = 'adaptive'

    class adapter:
        variant = 'split'            # split | anyadapter | no_split | no_hier_film | no_grl
        latent_obj = 16
        latent_dyn = 16
        latent_unified = 32
        film_hidden = 128
        grl_coef = 0.1
        grl_warmup_steps = 10000
        # loss weights (Table 6)
        w_mass = 2.0
        w_loaded = 1.0
        w_obj_wm = 1.0
        w_dyn_wm = 1.0
        w_grl = 1e-3
        w_kl = 1e-3
        w_l1_obj = 1e-3
        w_l1_dyn = 5e-3
        lambda_p = 1.0
        lambda_R = 1.0

    class runner(G1CfgPPO.runner):
        num_steps_per_env = 100
        max_iterations = 1500
        save_interval = 10           # frequent checkpoints: Isaac Gym may segfault, jobs auto-resume
        experiment_name = 'splitadapter'
        run_name = 'split'
        logger = 'tensorboard'


class BaseFinetuneCfgPPO(G1CfgPPO):
    """Stage 1: fine-tune the released PhysHSI carry-box policy with its own AMP+PPO trainer on the paper's
    task distribution (heights [0, 0.6] m, mass [0.5, 5] kg, box sizes, Table 3 DR)."""

    class runner(G1CfgPPO.runner):
        max_iterations = 1500
        save_interval = 10
        run_name = 'base_finetune'
        experiment_name = 'splitadapter_base'
        resume = True
        resume_path = None
