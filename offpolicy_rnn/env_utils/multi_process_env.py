import threading
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym.*")
warnings.filterwarnings("ignore", ".*The DISPLAY environment variable is missing") # dmc
warnings.filterwarnings("ignore", category=FutureWarning) # dmc
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

import isaacgym

from .make_env import make_env
from multiprocessing import Process, Queue
import os
import torch
import time
import faulthandler
from legged_gym.utils import webviewer_local_save
from low_control_pkg.LowerPolicy import LowerPolicy
from low_control_pkg.LowerPolicyPitch import LowerPitchPolicy
import numpy as np

def child_main(queue, queue_result, env_cfg, train_cfg, env_name: str, seed: int, save_dir: str, kwargs):
    parent_pid = os.getppid()
    web = False
    kwargs['save_dir'] = save_dir
    higher_mode = kwargs['higher_mode'] if 'higher_mode' in kwargs else False
    higher_pitch = kwargs['higher_pitch'] if 'higher_pitch' in kwargs else False

    if env_cfg is not None:
        env_dict = make_env(env_name, seed, env_cfg=env_cfg, train_cfg=train_cfg, **kwargs)
    else:
        env_dict = make_env(env_name, seed, **kwargs)

    env = env_dict['train_env']
    env_cfg, _ = env_dict['additional_configs']
    if higher_mode:
        if higher_pitch:
            lower_policy = LowerPitchPolicy()
            lower_policy.to_device(env.device)
        else:
            lower_policy = LowerPolicy()
            lower_policy.to_device(env.device)

    if save_dir is not None:
        web = True
        web_viewer = webviewer_local_save.WebViewer(output_dir=save_dir, maximum_camera_num=20)
        faulthandler.enable()
    if web:
        web_viewer.setup(env)
    def check_parent():
        try:
            # 通过发送信号0来检查父进程是否还在
            os.kill(parent_pid, 0)
        except OSError:
            return False
        return True

    def get_dict():
        result = {
            'train_env': None,
            'eval_env': None,
            'train_tasks': env_dict['train_tasks'],
            'eval_tasks': env_dict['eval_tasks'],
            'max_rollouts_per_task': env_dict['max_rollouts_per_task'],
            'max_trajectory_len': env_dict['max_trajectory_len'],  # when timestep > env.max_episode_length, triger done
            'obs_dim': env_dict['obs_dim'],
            'act_dim': env_dict['act_dim'],
            'act_continuous': env_dict['act_continuous'],
            'seed': env_dict['seed'],
            'multiagent': env_dict['multiagent'],
            'priv_obs_dim': env_dict['priv_obs_dim'],

        }
        if 'additional_configs' in env_dict:
            result['additional_configs'] = env_dict['additional_configs']
        return result

    def episode_length_buf():
        return env.episode_length_buf
    step_time_sum = [0]
    step_cnt = [0]
    obs_l = [None]
    priv_obs_l = [None]
    def step(act):
        act = act.to(env.device)
        start_time = time.time()
        if higher_mode:
            if higher_pitch:
                # action_lower = lower_policy.forward(env, act[..., 0:1].clamp(-1, 1) * np.pi / 3 + env.yaw.clone().unsqueeze(1),
                #                                     (act[..., 1:2].clamp(-1, 1) + 1) / 2 * 0.7 + 0.1,
                #                                     act[..., 2:3].clamp(-1, 1) * np.deg2rad(30.))
                action_lower = lower_policy.forward(env,
                                                    act[..., 0:1].clamp(-1, 1) * 3.14 / 2 + env.yaw.clone().unsqueeze(1),
                                                    # (act[..., 1:2].clamp(-1, 1) + 1) / 2 * 0.9 + 0.1,
                                                    act[..., :1] * 0.0 + 0.5,
                                                    torch.clamp(act[..., 1:2].clamp(-1, 1) * np.deg2rad(45.), -np.deg2rad(30.), np.deg2rad(30)))
            else:
                action_lower = lower_policy.forward(env, act.clamp(-1, 1) * 3.14 + env.yaw.clone().unsqueeze(1), torch.ones((env.num_envs, 1), device=env.device) * 0.7)

            next_state, privalege_state, reward, done, info = env.step(action_lower)
            lower_policy.reset(done)
        else:
            next_state, privalege_state, reward, done, info = env.step(act)
        step_cnt[0] += 1
        info.pop('depth', None)
        info.pop('time_outs', None)
        info.pop('delta_yaw_ok', None)

        step_time_sum[0] += time.time() - start_time
        if web:
            web_viewer.render_all(terminal_flag=done,
                                  fetch_results=True,
                                   step_graphics=True,
                                   render_all_camera_sensors=True,
                                   wait_for_page_load=True)
        obs_l[0] = next_state
        priv_obs_l[0] = privalege_state
        return [obs_l[0].clone(), priv_obs_l[0].clone(), reward.clone(), done.clone(), info]

    def step_with_ebd_diff(act, ebd_diff, target_std, target_flag, ep_return, target_ebd, privileged_ebd, noised_embedding):
        act = act.to(env.device)
        start_time = time.time()
        if higher_mode:
            if higher_pitch:
                action_lower = lower_policy.forward(env,
                                                    act[..., 0:1].clamp(-1, 1) * 3.14 / 2 + env.yaw.clone().unsqueeze(
                                                        1),
                                                    # (act[..., 1:2].clamp(-1, 1) + 1) / 2 * 0.9 + 0.1,
                                                    act[..., :1] * 0.0 + 0.5,
                                                    torch.clamp(act[..., 1:2].clamp(-1, 1) * np.deg2rad(45.), -np.deg2rad(30.), np.deg2rad(30)))
            else:
                action_lower = lower_policy.forward(env, act.clamp(-1, 1) * 3.14 + env.yaw.clone().unsqueeze(1), torch.ones((env.num_envs, 1), device=env.device) * 0.7)
            next_state, privalege_state, reward, done, info = env.step(action_lower)
            lower_policy.reset(done)
        else:
            next_state, privalege_state, reward, done, info = env.step(act)
        step_cnt[0] += 1
        info.pop('depth', None)
        info.pop('time_outs', None)
        info.pop('delta_yaw_ok', None)

        step_time_sum[0] += time.time() - start_time
        if web:
            web_viewer.render_all(terminal_flag=torch.ones_like(done) if step_cnt[0] % 1000 == 0 else torch.zeros_like(done),
                                  target_flag=target_flag,
                                  fetch_results=True,
                                  step_graphics=True,
                                  render_all_camera_sensors=True,
                                  wait_for_page_load=True,
                                  target_std=target_std,
                                  ebd_diff=ebd_diff,
                                  ep_return=ep_return + reward,
                                  target_ebd=target_ebd,
                                  privileged_ebd=privileged_ebd,
                                  noised_embedding=noised_embedding,
                                  save_attachments={
                                      # 'obs': obs_l[0],
                                      # 'priv_obs': priv_obs_l[0],
                                      # 'next_obs': next_state,
                                      # 'priv_next_obs': privalege_state,
                                      # 'act': act,
                                      # 'rew': reward,
                                      # 'done': done,
                                      # 'target_std': target_std,
                                      # 'ebd_diff': ebd_diff,
                                      # 'ep_return': ep_return + reward,
                                      # 'target_ebd': target_ebd,
                                      # 'privileged_ebd': privileged_ebd,
                                      # 'noised_embedding': noised_embedding
                                  }
                                  )
        obs_l[0] = next_state
        priv_obs_l[0] = privalege_state
        return [next_state.clone(), privalege_state.clone(), reward.clone(), done.clone(), info]

    def max_episode_length_s():
        return env.max_episode_length_s

    def device():
        return env.device.type if isinstance(env.device, torch.device) else env.device

    def get_processed_observation():
        obs_l[0] = env.get_processed_observation()
        return obs_l[0]

    def get_processed_privileged_observation():
        priv_obs_l[0] = env.get_processed_privileged_observation()
        return priv_obs_l[0]

    def reset_idx(env_idx):
        env.reset_idx(env_idx.to(env.device))

    def get_processed_observation_no_camera():
        return env.get_processed_observation_no_camera()

    def reset():
        print(f'[{os.getpid()}], step time cost: {step_time_sum[0]}')
        step_time_sum[0] = 0
        step_cnt[0] = 0
        return env.reset()

    def to_device(data, target_device):
        if isinstance(data, dict):
            for k in data:
                data[k] = to_device(data[k], target_device)
            return data
        elif isinstance(data, list):
            for i in range(len(data)):
                data[i] = to_device(data[i], target_device)
            return data
        elif isinstance(data, tuple):
            return tuple(to_device(data[i], target_device) for i in range(len(data)))
        elif isinstance(data, torch.Tensor):
            if data.device == target_device:
                return data
            return data.to(target_device)
        else:
            return data

    while True:
        if queue.empty():
            if not check_parent():
                break
            time.sleep(0.00001)
        else:
            args = queue.get()
            args = to_device([item for item in args], env.device)

            func_name = args[0]
            if func_name == 'step':
                result = step(args[1])
            elif func_name == 'step_with_ebd_diff':
                target_ebd = None if len(args) < 7 else args[6]
                privileged_ebd = None if len(args) < 8 else args[7]
                noised_embedding = None if len(args) < 9 else args[8]
                result = step_with_ebd_diff(args[1], args[2], args[3], args[4], args[5], target_ebd, privileged_ebd, noised_embedding)
            elif func_name == 'episode_length_buf':
                result = episode_length_buf()
            elif func_name == 'max_episode_length_s':
                result = max_episode_length_s()
            elif func_name == 'reset_idx':
                result = reset_idx(args[1])
            elif func_name == 'device':
                result = device()
            elif func_name == 'get_processed_observation':
                result = get_processed_observation()
            elif func_name == 'get_dict':
                result = get_dict()
            elif func_name == 'reset':
                result = reset()
            elif func_name == 'close':
                if web:
                    web_viewer.close()
                queue_result.put(None)
                break
            elif func_name == 'num_envs':
                result = env.num_envs
            elif func_name == 'get_processed_observation_no_camera':
                result = get_processed_observation_no_camera()
            elif func_name == 'terrain_levels':
                result = env.terrain_levels.clone().to(torch.device('cpu'))
            elif func_name == 'set_terrain_levels':
                try:
                    env.terrain_levels[:] = args[1][:].to(env.device)
                except Exception as e:
                    import traceback
                    print(type(args[1]))
                    traceback.print_exc()
                result = None
            elif func_name == 'get_processed_privileged_observation':
                result = get_processed_privileged_observation()
            else:
                raise NotImplementedError
            # print(f'child put result: {result}')
            # queue_result.put(to_device(result, torch.device('cpu')))
            queue_result.put(result)

class MultiProcessEnv:
    def __init__(self, env_name, seed, device, kwargs):
        self.queue = Queue()
        self.queue_result = Queue()
        self.cpu_device = torch.device('cpu')
        self.sum_step_time = 0
        self.close_command = False
        env_cfg, train_cfg = None, None
        self.env_device = torch.device('cuda:0')
        if 'save_dir' in kwargs:
            self.save_dir = kwargs['save_dir']
            kwargs.pop('save_dir')
        else:
            self.save_dir = None
        self.target_device = self.env_device if device is None else device
        if 'device_type' in kwargs:
            if kwargs['device_type'] == 'cuda':
                current_devices = os.getenv('CUDA_VISIBLE_DEVICES').split(',')
                if len(current_devices) > 1:
                    os.environ['CUDA_VISIBLE_DEVICES'] = current_devices[kwargs['device_index']] # If set multiple gpu, there will be an error in camera
                self.env_device = torch.device(f'{kwargs["device_type"]}:{kwargs["device_index"]}')
                print(f'environment device: {self.env_device}')
            kwargs.pop('device_type')
            kwargs.pop('device_index')
        self.sub_process = Process(
            target=child_main,
            args=(
                self.queue,
                self.queue_result,
                env_cfg,
                train_cfg,
                env_name,
                seed,
                self.save_dir,
                kwargs
            ),

        )
        self.simulator_alive = True
        self.sub_process.start()
        self.guard_thread = threading.Thread(target=self.monitor_process)
        self.guard_thread.start()

        self.env_dict = self.get_dict(timeout=250.0)
        self.num_envs = self.get_num_envs()

    def monitor_process(self,):
        while self.sub_process.is_alive():
            time.sleep(0.2)  # 每秒检查一次

        if not self.close_command:
            self.simulator_alive = False
            # raise RuntimeError(f'Simulator failed!')

    def get_from_queue(self, timeout=4.0):

        result = self.queue_result.get(timeout=timeout)
        while not self.queue_result.empty():
            result = self.queue_result.get(timeout=timeout)
        # print(f'get from queue: {result}, {type(result)}')
        return result

    def to_device(self, data, target_device, clone=False):
        if isinstance(data, dict):
            for k in data:
                data[k] = self.to_device(data[k], target_device)
            return data
        elif isinstance(data, list):
            for i in range(len(data)):
                data[i] = self.to_device(data[i], target_device)
            return data
        elif isinstance(data, tuple):
            return tuple(self.to_device(data[i], target_device) for i in range(len(data)))
        elif isinstance(data, torch.Tensor):
            if clone:
                if data.device == target_device:
                    return data.clone()
                return data.to(target_device).clone()
            else:
                if data.device == target_device:
                    return data
                return data.to(target_device)
        else:
            return data

    def call(self, args, timeout=10.0):
        args = self.to_device(args, self.env_device, clone=True)
        self.queue.put(args)
        result = self.get_from_queue(timeout)
        result = self.to_device(result, self.target_device, clone=True)
        return result

    def get_dict(self, timeout):
        args = ('get_dict', )
        return self.call(args, timeout=timeout)

    def get_num_envs(self):
        args = ('num_envs',)
        return self.call(args)

    def step(self, act):
        args = ('step', act)
        start_time = time.time()
        res = self.call(args)
        self.sum_step_time += time.time() - start_time
        return res

    def step_with_ebd_diff(self, act, ebd_diff, target_std, target_flag, ep_return, target_ebd=None, privileged_ebd=None, noised_ebd=None):
        args = ('step_with_ebd_diff', act, ebd_diff, target_std, target_flag, ep_return, target_ebd, privileged_ebd, noised_ebd)
        start_time = time.time()
        res = self.call(args)
        self.sum_step_time += time.time() - start_time
        return res

    @property
    def episode_length_buf(self):
        args = ('episode_length_buf',)
        return self.call(args)

    @property
    def max_episode_length_s(self):
        args = ('max_episode_length_s',)
        return self.call(args)

    @property
    def terrain_levels(self):
        args = ('terrain_levels',)
        for _ in range(5):
            terrain_levels = self.call(args)
            if isinstance(terrain_levels, list):
                print(f'terrain_levels: {len(terrain_levels)} {terrain_levels}')
            else:
                break
        return terrain_levels.clone()

    def set_terrain_levels(self, levels):
        args = ('set_terrain_levels', levels.clone())
        return self.call(args)

    def reset_idx(self, idx):
        args = ('reset_idx', idx)
        return self.call(args)

    def reset(self):
        print(f'parent [{os.getpid()}], step time cost: {self.sum_step_time}')
        self.sum_step_time = 0
        args = ('reset',)
        return self.call(args)

    @property
    def device(self):
        return self.env_device

    def get_processed_observation(self):
        args = ('get_processed_observation', )
        return self.call(args)

    def get_processed_privileged_observation(self):
        args = ('get_processed_privileged_observation',)
        return self.call(args)

    def get_processed_observation_no_camera(self):
        args = ('get_processed_observation_no_camera', )
        return self.call(args)

    def close(self):
        args = ('close',)
        self.close_command = True
        self.call(args)
        self.queue.close()  # 关闭队列
        self.queue_result.close()
        return None

    def force_close(self):
        try:
            self.sub_process.terminate()
            self.queue.close()  # 关闭队列
            self.queue_result.close()
        except Exception as e:
            pass


def make_multi_process_env(env_name, seed, device=None, **kwargs):
    multi_process_env = MultiProcessEnv(env_name, seed, device, kwargs)
    env_dict = multi_process_env.env_dict
    env_dict['train_env'] = multi_process_env
    env_dict['eval_env'] = multi_process_env
    return env_dict


