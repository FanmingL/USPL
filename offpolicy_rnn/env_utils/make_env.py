import isaacgym
from legged_gym.envs import *

import torch
from envs_train.make_pomdp_env import make_pomdp_env
from envs_train.pomdp_config import env_config
import gym
from typing import Dict
import numpy as np
import contextlib
import random
from argparse import Namespace
try:
    import envs_train.dmc
    has_dm_control = True
except:
    has_dm_control = False
from legged_gym.utils import task_registry
from isaacgym import gymapi

@contextlib.contextmanager
def fixed_seed(seed):
    """上下文管理器，用于同时固定random和numpy.random的种子"""
    state_np = np.random.get_state()
    state_random = random.getstate()
    torch_random = torch.get_rng_state()
    if torch.cuda.is_available():
        # TODO: if not preproducable, check torch.get_rng_state
        torch_cuda_random = torch.cuda.get_rng_state()
        torch_cuda_random_all = torch.cuda.get_rng_state_all()
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    try:
        yield
    finally:
        np.random.set_state(state_np)
        random.setstate(state_random)
        torch.set_rng_state(torch_random)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(torch_cuda_random)
            torch.cuda.set_rng_state_all(torch_cuda_random_all)

def print_config(cfg, indent=0):
    # Iterate through all attributes of the class or instance
    if indent >= 16:
        return
    for attr in dir(cfg):
        # Ignore built-in attributes and methods
        if not attr.startswith('__'):
            value = getattr(cfg, attr)
            # Check if the attribute is a nested class or a configuration object
            if isinstance(value, type) or isinstance(value, object):
                # Check if it's a nested class or a simple data structure
                if not callable(value) and not isinstance(value, (int, float, str, bool, list, dict)):
                    print(' ' * indent + f"{attr}:")
                    # Recursively print attributes of the nested class/object
                    print_config(value, indent + 4)
                else:
                    print(' ' * indent + f"{attr}: {value}")
# from envpool.python.api import py_env, ParallelEnvSpec
#
# @py_env
# class MyBulletEnvPool:
#     def __init__(self, config):
#         seed = config.get('seed', None)
#         self.env = MyPyBulletEnv(seed=seed)
#
#     def reset(self):
#         return self.env.reset()
#
#     def step(self, action):
#         return self.env.step(action)
#
#     @staticmethod
#     def gen_spec(config):
#         env = MyPyBulletEnv()
#         return ParallelEnvSpec(
#             obs_space=env.observation_space,
#             act_space=env.action_space,
#             max_episode_steps=1000
#         )

def make_env(env_name: str, seed: int, env=None, **kwargs) -> Dict:
    if env_name in env_config:
        with fixed_seed(seed):
            result = make_pomdp_env(env_name, seed)
        result['seed'] = seed
        return result
    elif env_name == 'ParkourGo-v0' or env_name == 'ParkourAlien-v0':
        task_name = 'a1' if env_name == 'ParkourGo-v0' else 'aliengo'
        device_type = kwargs['device_type'] if 'device_type' in kwargs else 'cuda'
        device_id = kwargs['device_id'] if 'device_id' in kwargs else 0
        device_name = f'{device_type}:{device_id}' if device_id is not None else device_type
        if 'save_dir' in kwargs and kwargs['save_dir'] is not None:
            kwargs['headless'] = False
        # TODO
        args = Namespace(
            checkpoint=-1,
            cols=None,
            compute_device_id=device_id,
            daggerid=None,
            debug=False,
            delay=False,
            device=device_name,
            draw=False,
            experiment_name=None,
            exptid='PARKOUR',
            flex=False,
            graphics_device_id=device_id,
            headless=kwargs['headless'] if 'headless' in kwargs else True,   # TODO important
            hitid=None,
            horovod=False,
            load_run=None,
            mask_obs=False,
            max_iterations=None,
            no_wandb=False,
            nodelay=False,
            num_envs=None,
            num_threads=0,
            physics_engine=gymapi.SIM_PHYSX,
            physx=False,
            pipeline='gpu',
            proj_name='parkour',
            resume=False,
            resumeid=None,
            rl_device=device_name,
            rows=None,
            run_name=None,
            save=False,
            seed=None,
            sim_device=device_name,
            sim_device_id=device_id,
            sim_device_type=device_type,
            slices=0,
            subscenes=0,
            task=task_name,
            task_both=False,
            teacher=None,
            use_camera=False,
            use_gpu=True,
            use_gpu_pipeline=True,
            use_jit=False,
            use_latent=False,
            web='save_dir' in kwargs and kwargs['save_dir'] is not None,
        )
        if env is None:
            if 'env_cfg' in kwargs:
                env_cfg = kwargs['env_cfg']
                train_cfg = kwargs['train_cfg']
            else:
                env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
            # override some parameters for testing
            env_cfg.env.num_envs = 6144 # luofm comment: fix the parallel env num to 64
            env_cfg.env.num_envs = 1024 # luofm comment: fix the parallel env num to 64
            if 'num_envs' in kwargs:
                env_cfg.env.num_envs = kwargs['num_envs']
            if 'tracking_goal_vel_factor' in kwargs:
                env_cfg.rewards.scales.tracking_goal_vel = kwargs['tracking_goal_vel_factor']
            if 'only_positive_rewards' in kwargs:
                env_cfg.rewards.only_positive_rewards = kwargs['only_positive_rewards']
            if 'mix_positive_reward' in kwargs:
                env_cfg.rewards.mix_positive_reward = kwargs['mix_positive_reward']
            if 'no_privileged_info' in kwargs:
                env_cfg.env.no_privileged_info = kwargs['no_privileged_info']
            if 'curriculum' in kwargs:
                env_cfg.terrain.curriculum = kwargs['curriculum']
            if 'depth' in kwargs:
                env_cfg.depth.use_camera = kwargs['depth']
            if 'no_ext_privileged_info' in kwargs:
                env_cfg.env.no_ext_privileged_info = kwargs['no_ext_privileged_info']
            if 'extreme_little_info' in kwargs:
                env_cfg.env.extreme_little_info = kwargs['extreme_little_info']
            if 'action_delay' in kwargs:
                env_cfg.domain_rand.action_delay = kwargs['action_delay']
            if 'include_absolute_position' in kwargs:
                env_cfg.env.include_absolute_position = kwargs['include_absolute_position']
            if 'include_scan_dot' in kwargs:
                env_cfg.env.include_scan_dot = kwargs['include_scan_dot']
            if 'include_yaw' in kwargs:
                env_cfg.env.include_yaw = kwargs['include_yaw']
            if 'higher_mode' in kwargs:
                env_cfg.env.higher_mode = kwargs['higher_mode']
                if 'higher_pitch' in kwargs and kwargs['higher_pitch']:
                    env_cfg.env.include_pitch = True
            if 'ground_truth_mode' in kwargs:
                env_cfg.env.ground_truth_mode = kwargs['ground_truth_mode']
            if env_cfg.env.include_scan_dot:
                env_cfg.terrain.terrain_dict["parkour_hurdle"] = 1.0
                env_cfg.terrain.terrain_dict['parkour_flat'] = 0.0
                env_cfg.terrain.terrain_dict['parkour_step'] = 0.0
                env_cfg.terrain.terrain_dict["parkour_gap"] = 0.0
                env_cfg.terrain.terrain_dict["demo"] = 0.0

                env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
            if 'plat_task' in kwargs and kwargs['plat_task']:
                env_cfg.terrain.terrain_dict["parkour_hurdle"] = 0.0
                env_cfg.terrain.terrain_dict['parkour_flat'] = 1.0
                env_cfg.terrain.terrain_dict['parkour_step'] = 0.0
                env_cfg.terrain.terrain_dict["parkour_gap"] = 0.0
                env_cfg.terrain.terrain_dict["demo"] = 0.0
                env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
                if not env_cfg.env.include_scan_dot:
                    env_cfg.terrain.terrain_dict["parkour_hurdle"] = 0.0
                    env_cfg.terrain.terrain_dict['parkour_flat'] = 0.0
                    env_cfg.terrain.terrain_dict['parkour_step'] = 0.0
                    env_cfg.terrain.terrain_dict["parkour_gap"] = 1.0
                    env_cfg.terrain.terrain_dict["demo"] = 0.0
                    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
            if 'max_difficulty' in kwargs and kwargs['max_difficulty']:
                env_cfg.terrain.max_difficulty = True
            camera_setting_but_no_camera = kwargs['camera_setting_but_no_camera'] if 'camera_setting_but_no_camera' in kwargs else False
            no_simplify = kwargs['no_simplify'] if 'no_simplify' in kwargs else False
            # luofm comment simply here
            env_cfg.terrain.simplify_grid = True  # IMPORTANT
            env_cfg.terrain.height = [0.02, 0.02]
            if 'env_num_rows' in kwargs:
                env_cfg.terrain.num_rows = kwargs['env_num_rows']
            if 'env_num_cols' in kwargs:
                env_cfg.terrain.num_cols = kwargs['env_num_cols']
            if env_cfg.depth.use_camera or camera_setting_but_no_camera:
                if camera_setting_but_no_camera:
                    env_num = env_cfg.env.num_envs
                    env_cfg.depth.camera_num_envs = env_num
                    kwargs['camera_num_envs'] = env_num
                # modifed 10.25
                env_cfg.depth.camera_terrain_num_cols = 20 # env_cfg.terrain.num_cols
                env_cfg.depth.camera_terrain_num_rows = env_cfg.terrain.num_rows
                env_cfg.env.num_envs = env_cfg.depth.camera_num_envs
                if 'camera_num_envs' in kwargs:
                    env_cfg.env.num_envs = kwargs['camera_num_envs']
                    env_cfg.depth.camera_num_envs = kwargs['camera_num_envs']
                    # print(f'env num envs: {env_cfg.env.num_envs}')
                env_cfg.terrain.num_rows = env_cfg.depth.camera_terrain_num_rows
                env_cfg.terrain.num_cols = env_cfg.depth.camera_terrain_num_cols
                if 'env_num_rows' in kwargs:
                    env_cfg.terrain.num_rows = kwargs['env_num_rows']
                if 'env_num_cols' in kwargs:
                    env_cfg.terrain.num_cols = kwargs['env_num_cols']

                if not camera_setting_but_no_camera and not no_simplify: # modified 10.25
                    # env_cfg.terrain.max_error = env_cfg.terrain.max_error_camera
                    # env_cfg.terrain.horizontal_scale = env_cfg.terrain.horizontal_scale_camera
                    # env_cfg.terrain.simplify_grid = True      # IMPORTANT
                    env_cfg.terrain.max_error_camera = env_cfg.terrain.max_error
                    env_cfg.terrain.horizontal_scale_camera = env_cfg.terrain.horizontal_scale
                    env_cfg.terrain.simplify_grid = True  # IMPORTANT

                else:
                    env_cfg.terrain.max_error_camera = env_cfg.terrain.max_error
                    env_cfg.terrain.horizontal_scale_camera = env_cfg.terrain.horizontal_scale
                    # env_cfg.terrain.simplify_grid = True      # IMPORTANT

                # env_cfg.terrain.terrain_dict["parkour"] = 0.2
                # env_cfg.terrain.terrain_dict["parkour_hurdle"] = 0.2
                # env_cfg.terrain.terrain_dict["parkour_flat"] = 0.15
                # env_cfg.terrain.terrain_dict["parkour_gap"] = 0.2
                # env_cfg.terrain.terrain_dict["parkour_step"] = 0.2
                # env_cfg.terrain.terrain_dict["demo"] = 0.05
                env_cfg.terrain.terrain_dict["parkour_hurdle"] = 0.0
                env_cfg.terrain.terrain_dict["parkour_gap"] = 0.0
                env_cfg.terrain.terrain_dict['parkour_flat'] = 1.0
                env_cfg.terrain.terrain_dict['parkour_step'] = 0.0
                if env_cfg.env.include_scan_dot:
                    env_cfg.terrain.terrain_dict["parkour_hurdle"] = 1.0
                    env_cfg.terrain.terrain_dict['parkour_flat'] = 0.0
                    env_cfg.terrain.terrain_dict['parkour_step'] = 0.0
                    env_cfg.terrain.terrain_dict["parkour_gap"] = 0.0
                    env_cfg.terrain.terrain_dict["demo"] = 0.0
                    env_cfg.terrain.middle_choose_length = 2.55
                    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())

                if 'plat_task' in kwargs and kwargs['plat_task']:
                    env_cfg.terrain.terrain_dict["parkour_hurdle"] = 0.0
                    env_cfg.terrain.terrain_dict['parkour_flat'] = 1.0
                    env_cfg.terrain.terrain_dict['parkour_step'] = 0.0
                    env_cfg.terrain.terrain_dict["parkour_gap"] = 0.0
                    env_cfg.terrain.terrain_dict["demo"] = 0.0

                    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
                    if not env_cfg.env.include_scan_dot:
                        env_cfg.terrain.terrain_dict["parkour_hurdle"] = 0.0
                        env_cfg.terrain.terrain_dict['parkour_flat'] = 0.0
                        env_cfg.terrain.terrain_dict['parkour_step'] = 0.0
                        env_cfg.terrain.terrain_dict["parkour_gap"] = 0.0
                        env_cfg.terrain.terrain_dict["demo"] = 1.0
                        env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())

                env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
                # commented 10.25
                # if not camera_setting_but_no_camera and not no_simplify:
                #     env_cfg.terrain.y_range = [-0.1, 0.1]
                env_cfg.terrain.height = [0.02, 0.02]
            # if 'headless' in kwargs and not kwargs['headless']:
            #     env_cfg.terrain.num_rows = 5
            #     env_cfg.terrain.num_cols = 5
            #     env_cfg.env.num_envs = 16
            #
            #     env_cfg.terrain.height = [0.02, 0.02]
            #     env_cfg.terrain.terrain_dict = {"smooth slope": 0.,
            #                                     "rough slope up": 0.0,
            #                                     "rough slope down": 0.0,
            #                                     "rough stairs up": 0.,
            #                                     "rough stairs down": 0.,
            #                                     "discrete": 0.,
            #                                     "stepping stones": 0.0,
            #                                     "gaps": 0.,
            #                                     "smooth flat": 0,
            #                                     "pit": 0.0,
            #                                     "wall": 0.0,
            #                                     "platform": 0.,
            #                                     "large stairs up": 0.,
            #                                     "large stairs down": 0.,
            #                                     "parkour": 0.2,
            #                                     "parkour_hurdle": 0.2,
            #                                     "parkour_flat": 0.,
            #                                     "parkour_step": 0.2,
            #                                     "parkour_gap": 0.2,
            #                                     "demo": 0.2}
            #
            #     env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
            #     env_cfg.terrain.curriculum = False
            #     env_cfg.terrain.max_difficulty = True  # luofm modified,  by default True
            #
            # env_cfg.depth.angle = [0, 1]
            # env_cfg.noise.add_noise = True
            #     env_cfg.domain_rand.randomize_friction = True
            #     env_cfg.domain_rand.push_robots = False
            #     env_cfg.domain_rand.push_interval_s = 6
            #     env_cfg.domain_rand.randomize_base_mass = False
            #     env_cfg.domain_rand.randomize_base_com = False

            # prepare environment
            env_cfg.seed = seed
            # print_config(env_cfg)
            # print_config(args)
            with fixed_seed(seed):
                env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        else:
            env_cfg, train_cfg = None, None
        act_dim = 12
        if 'higher_mode' in kwargs and kwargs['higher_mode']:
            if 'higher_pitch' in kwargs and kwargs['higher_pitch']:
                act_dim = 2
            else:
                act_dim = 1
        result = {
            'train_env': env,
            'eval_env': env,
            'train_tasks': [],
            'eval_tasks': [None] * 20,
            'max_rollouts_per_task': 1,
            'max_trajectory_len': env.max_episode_length + 1, # when timestep > env.max_episode_length, triger done
            'obs_dim': env.get_processed_observation().shape[-1],
            'priv_obs_dim': env.get_processed_privileged_observation().shape[-1],
            'act_dim': act_dim,
            'act_continuous': True,
            'seed': seed,
            'multiagent': False,
            'additional_configs': (env_cfg, train_cfg),
        }
        return result
    else:
        with fixed_seed(seed):
            if env_name.startswith('dmc'):
                env = gym.make(env_name, seed=seed)
                max_episode_steps = env.unwrapped._max_episode_steps
            else:
                env = gym.make(env_name)
                max_episode_steps = env._max_episode_steps
        env.seed(seed)
        env.action_space.seed(seed+1)
        env.observation_space.seed(seed+2)
        result = {
            'train_env': env,
            'eval_env': env,
            'train_tasks': [],
            'eval_tasks': [None] * 20,
            'max_rollouts_per_task': 1,
            'max_trajectory_len': max_episode_steps,
            'obs_dim': env.observation_space.shape[0],
            'act_dim': env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else env.action_space.n,
            'act_continuous': isinstance(env.action_space, gym.spaces.Box),
            'seed': seed,
            'multiagent': False
        }

        return result