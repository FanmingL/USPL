import copy
import threading

import math
import time

from ..buffers.replay_memory import Transition
from ..parameter.Parameter import Parameter
from ..policy_value_models.contextual_ppo_policy_mlp_encoder import ContextualPPOPolicy
from ..policy_value_models.contextual_ppo_value_mlp_encoder import ContextualPPOValue
from ..utility.timer import Timer
from ..env_utils.make_env import make_env
from ..utility.ValueScheduler import CosineScheduler
from typing import List, Union, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
import random
from legged_gym.utils import webviewer
import torch
import smart_logger
from smart_logger import Logger
import os
import gym
from ..policy_value_models.make_models import make_policy_model, make_value_model
from ..env_utils.multi_process_env import make_multi_process_env
from ..buffers.transition_buffer.nested_replay_memory import NestedMemoryArray as NestedTransitionMemoryArray
import pickle
from queue import Queue
import redis
import errno
from ..utility.too_many_open_files import close_open_pipes


class PPO:
    def __init__(self, parameter, env=None):
        # 1. smart_logger init and parameter init
        self.parameter = parameter
        self.WEB = False

        self.timer = Timer()
        self.logger = Logger(log_name=self.parameter.short_name)
        self.parameter.set_config_path(os.path.join(self.logger.output_dir, 'config'))
        if not parameter.debug:
            self.parameter.save_config()
            self.logger(self.parameter)
        # 2. make env
        self.env_name = self.parameter.env_name
        # self.device = torch.device(f'cuda:{torch.cuda.device_count()-1}') if torch.cuda.is_available() else torch.device('cpu')
        self.device = torch.device(f'cuda:{torch.cuda.device_count() - 1}') if torch.cuda.is_available() else torch.device('cpu')
        self.simulator_device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # self.device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # self.simulator_device = torch.device(f'cuda:{torch.cuda.device_count() - 1}') if torch.cuda.is_available() else torch.device('cpu')
        self.sample_device = self.simulator_device if self.parameter.cuda_inference else torch.device('cpu')

        self.remote_env_additional_configs = {
            'only_positive_rewards': False,
            'tracking_goal_vel_factor': 200.0,
            'curriculum': False,
            'depth': self.parameter.image_input if hasattr(self.parameter, 'image_input') else False,
            'num_envs': 900, # RNN policy requires more GPU memory, num_envs should be set to 600
            'no_privileged_info': self.parameter.no_privileged_info if hasattr(self.parameter, 'no_privileged_info') else False,
            'no_ext_privileged_info': self.parameter.no_ext_privileged_info if hasattr(self.parameter, 'no_ext_privileged_info') else False,
            'headless': not self.WEB,
            # 'camera_num_envs': 130,
            'camera_num_envs': 180,
            'device_type': self.simulator_device.type,
            'device_index': self.simulator_device.index,
            'camera_setting_but_no_camera': False,
            'no_simplify': False,
            'action_delay': self.parameter.action_delay if hasattr(self.parameter, 'action_delay') else False,
            'extreme_little_info': self.parameter.extreme_little_info if hasattr(self.parameter, 'extreme_little_info') else False,
            'include_absolute_position': self.parameter.include_absolute_position if hasattr(self.parameter, 'include_absolute_position') else False,
            'include_scan_dot': self.parameter.square_task if hasattr(self.parameter, 'square_task') else False,
            'plat_task': self.parameter.plat_task if hasattr(self.parameter, 'plat_task') else False,
            'include_yaw': self.parameter.include_yaw if hasattr(self.parameter, 'include_yaw') else False,
            'save_dir': self.logger.output_dir,
            'higher_mode': self.parameter.higher_mode if hasattr(self.parameter, 'higher_mode') else False,
            'higher_pitch': self.parameter.higher_pitch if hasattr(self.parameter, 'higher_pitch') else False,
            'ground_truth_mode': self.parameter.ground_truth_mode if hasattr(self.parameter, 'ground_truth_mode') else False,
        }
        self.role = 'master' if self.parameter.seed == 1 or not self.parameter.image_input else 'client'
        if not self.role == 'master':
            self.remote_env_additional_configs['save_dir'] = None
        else:
            self.remote_env_additional_configs['save_dir'] = None # self.logger.output_dir
            pass
        self.traj_reserve_prob = 1.0
        self.offpolicy_reserve_prob = 0.8
        if env is not None:
            self.env_info = make_env(self.env_name, self.parameter.seed + 1, env, headless=not self.WEB)
            if parameter.higher_mode:
                self.env_info['act_dim'] = 2 if hasattr(parameter, 'higher_pitch') and parameter.higher_pitch else 1
            else:
                self.env_info['act_dim'] = 12
        else:
            self.env_info = make_multi_process_env(self.env_name, self.parameter.seed + 1, self.sample_device, **self.remote_env_additional_configs)
        self.env_info['save_dir'] = self.remote_env_additional_configs['save_dir']
        self.env: gym.Env = self.env_info['train_env']
        self.max_episode_steps = int(self.env_info['max_trajectory_len'])
        self.discrete_env = not self.env_info['act_continuous']
        self.obs_dim = self.env_info['obs_dim']
        self.priv_obs_dim = self.env_info['priv_obs_dim']
        self.act_dim = self.env_info['act_dim']
        # 3. set random seed
        self._seed(self.parameter.seed)
        # 4. make policy and value
        self.policy_args = self._make_policy_args(self.parameter)
        self.value_args = self._make_value_args(self.parameter)
        # if torch.has_mps:
        #     self.device = torch.device('mps')
        self.optim_class = torch.optim.AdamW
        self.base_algorithm = self.parameter.base_algorithm if hasattr(self.parameter, 'base_algorithm') else 'sac'
        self.policy: ContextualPPOPolicy = make_policy_model(self.policy_args, self.base_algorithm, self.discrete_env)
        self.values: List[ContextualPPOValue] = [make_value_model(self.value_args, self.base_algorithm, self.discrete_env) for _ in range(self.parameter.value_net_num)]
        self.target_values: List[ContextualPPOValue] = [make_value_model(self.value_args, self.base_algorithm, self.discrete_env) for _ in range(self.parameter.value_net_num)]
        for value in self.values + self.target_values:
            value.to(self.device)
        self.policy.to(self.sample_device)
        self._value_update(tau=0.0)     # hard update
        if self.discrete_env:
            self.logger(f'[Notice] Action space {self.parameter.env_name} is discrete.')
            self.parameter.no_alpha_auto_tune = True
        if hasattr(self.parameter, 'no_alpha_auto_tune') and self.parameter.no_alpha_auto_tune:
            self.log_sac_alpha = torch.Tensor([math.log(self.parameter.sac_alpha)]).to(self.device).to(torch.get_default_dtype()).requires_grad_(True)
        else:
            self.log_sac_alpha = torch.Tensor([-1.0]).to(self.device).to(torch.get_default_dtype()).requires_grad_(True)
        self.target_entropy = -self.act_dim * self.parameter.target_entropy_ratio if not self.discrete_env else self.parameter.target_entropy_ratio
        # 5. make policy and value optimizer
        self.optimizer_policy = self.optim_class(self.policy.parameters(True),
                                                 lr=self.parameter.policy_lr,
                                                 weight_decay=self.parameter.policy_l2_norm)
        value_parameters = [param for value in self.values for param in value.parameters(True)]
        value_embedding_parameters = [param for value in self.values for param in value.embedding_network.parameters(True)]

        self.value_parameters = value_parameters
        self.value_embedding_parameters = value_embedding_parameters
        self.optimizer_value = self.optim_class(value_parameters, lr=self.parameter.value_lr, weight_decay=self.parameter.value_l2_norm)
        self.optimizer_alpha = self.optim_class([self.log_sac_alpha], lr=self.parameter.alpha_lr)
        for value in self.values:
            value.train()
        for target_value in self.target_values:
            target_value.eval()
        self.policy.train()
        if hasattr(self.parameter, 'directly_train_target') and self.parameter.directly_train_target:
            self.policy.target_mode = True
            self.policy.no_logstd_output = True

        # 6. make replay buffer
        # self.replay_buffer = MemoryArray(self.parameter.max_buffer_traj_num, smart_logger.get_customized_value('MAX_TRAJ_STEP'))

        self.parallel_agent_num = self.env.num_envs
        self.logger(f'parallel_agent_num: {self.parallel_agent_num}')
        # 7. init variables
        self.state_np = self.env.get_processed_observation().clone() # torch.zeros((self.parallel_agent_num, self.obs_dim), device=self.sample_device)
        self.priv_state_np = self.env.get_processed_privileged_observation().clone() # torch.zeros((self.parallel_agent_num, self.obs_dim), device=self.sample_device)
        self.last_action_np = torch.zeros((self.parallel_agent_num, self.act_dim), device=self.sample_device)
        self.last_state_np = torch.zeros((self.parallel_agent_num, self.obs_dim), device=self.sample_device)
        self.last_priv_state_np = torch.zeros((self.parallel_agent_num, self.priv_obs_dim), device=self.sample_device)
        self.reward_np = torch.zeros((self.parallel_agent_num, 1), device=self.sample_device)

        self.sample_hidden = self._init_sample_hidden()
        self.target_encoder_hidden = self.policy.target_encoder.make_init_state(self.parallel_agent_num, self.sample_device)
        self.collect_status = True
        self.env_reset()

        # 8. statistic variables
        self.sample_num = 0
        self.grad_num = 0
        self.start_time = time.time()
        self.env_max_episode_length_s = self.env.max_episode_length_s
        # 9. sample process

        # self.process_context = multiprocessing.get_context("spawn")
        # self.process_pool = ProcessPoolExecutor(max_workers=self.parameter.test_nprocess, mp_context=self.process_context)
        # self.instable_env = False

        # 10. learning scheduler [optional], if requiring setting the schedulers reload the init_lr_scheduler function
        self.actor_lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.critic_lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.target_entropy_scheduler: Optional[CosineScheduler] = None

        # 11. check whether nest-stack is allowed
        self.allow_nest_stack = self.allow_nest_stack_trajs()
        self.current_env_step = 0
        self.dog_cnt = 0

        if hasattr(self.parameter, 'resume_log_name') and self.parameter.resume_log_name is not None:
            self.load(os.path.join(smart_logger.get_base_path(), 'logfile', self.parameter.resume_log_name, 'model'), True, True)
        if env is None:
            self.env_queue = Queue(maxsize=1)
            self.env_preparing_thread = threading.Thread(target=self._env_prepare_th)
            self.env_preparing_thread.start()
            time.sleep(0.5)

        # self.role = 'client'
        self.master_ip = '10.244.6.193'
        # self.master_ip = '10.244.2.154'
        self.total_client_num = 3 if self.parameter.image_input else 0
        if env is not None:
            self.total_client_num = 0
        # if self.role == 'master':
        #     self.pool = redis.ConnectionPool(host='0.0.0.0', port=12232, decode_responses=True)
        self.replay_buffer = NestedTransitionMemoryArray(int(self.max_episode_steps * self.env.num_envs * (1 + self.total_client_num) * 1.1),
                                                         self.env_info['max_trajectory_len'],
                                                         additional_history_len=self._get_skip_len())
        self.offpolicy_buf_size = 4.5e6
        # self.offpolicy_buf_size = 3.5e6
        self.offpolicy_replay_buffer = NestedTransitionMemoryArray(int(self.offpolicy_buf_size),
                                                         self.env_info['max_trajectory_len'],
                                                         additional_history_len=self._get_skip_len())
        if env is None:
            self.image_dictionary_offpolicy = None
            self.image_offpolicy_cnt = 1
            self.image_offpolicy_maximum_cnt = int(self.offpolicy_buf_size / self.env_info['additional_configs'][0].depth.update_interval * 2.0)
            self.logger(f'image_offpolicy_maximum_cnt: {self.image_offpolicy_maximum_cnt}')
        if self.total_client_num > 0:
            self.redis = redis.StrictRedis(
                host='127.0.0.1' if self.role == 'master' else self.master_ip,
                port=16379,
                db=0,
                password=None,
                decode_responses=False,
                health_check_interval=30
            )
            if self.role == 'master':
                self.redis.flushall()
        self.machine_id = 0 if self.role == 'master' else self.parameter.seed - 1
        self.iter_train = 0
        self.threshold_iter = 100 if self.parameter.resume_log_name is None else 0
        # self.threshold_iter = 150 if self.parameter.resume_log_name is None else 0
        self.target_encoder_learn_std = self.parameter.target_encoder_learn_std if hasattr(self.parameter, 'target_encoder_learn_std') else False


    def _env_prepare_th(self):
        # while True:
        #     time.sleep(0.1)
        env_cnt = 10
        while True:
            if self.env_queue.empty():
                try:
                    remote_env_additional_configs = copy.deepcopy(self.remote_env_additional_configs)
                    if env_cnt > 5 and env_cnt % 5 == 0 and self.role == 'master':
                        remote_env_additional_configs['save_dir'] = self.logger.output_dir
                    else:
                        remote_env_additional_configs['save_dir'] = None
                    env_info = make_multi_process_env(self.env_name, random.randint(0, 100000000), self.sample_device,
                                                           **remote_env_additional_configs)
                    env_info['save_dir'] = remote_env_additional_configs['save_dir']
                    self.env_queue.put(env_info)
                    self.remote_env_additional_configs = remote_env_additional_configs
                    env_cnt += 1
                except OSError as e:  # 明确捕获操作系统相关异常
                    if e.errno == errno.EMFILE:
                        self.logger('Too many open files, closing open pipes...')
                        close_open_pipes(self.logger)
                    else:
                        self.logger(f'OSError occurred: {e}')
                except Exception as e:
                    self.logger(f'making new environment failed!! {e}')
                    import traceback
                    traceback.print_exc()
                    time.sleep(1)
            else:
                time.sleep(0.2)

    def reset_compression(self, agent_idx=0):
        self.image_dictionary = None
        self.last_image = None
        self.agent_idx = agent_idx
        self.idx = 1
        self.ptr = 1
        self.offset = 0

    def compress_state(self, state, step, device):
        if self.parameter.image_input:
            # idx = self.idx
            image_dim = self.policy.image_dim
            env_cfg, _ = self.env_info['additional_configs']
            update_flag = step % env_cfg.depth.update_interval == 0
            if self.image_dictionary is None:
                size = self.parallel_agent_num * ((self.max_episode_steps // env_cfg.depth.update_interval) + 4)
                self.offset = self.agent_idx * size
                self.image_dictionary = torch.zeros((size, image_dim), device=device)
                self.logger(f'creating image dictionary with shape: {self.image_dictionary.shape}')
            original_shape = list(state.shape)
            state_2dim = state.reshape((-1, state.shape[-1]))
            image_num = state_2dim.shape[0]
            if update_flag:
                image = state_2dim[:, -image_dim:]
                self.image_dictionary[self.idx:self.idx+image_num, :] = image
                self.ptr = self.idx
                self.idx += image_num

            state_2dim = torch.cat((state_2dim[:, :-image_dim], torch.arange(0, image_num, device=device).reshape((-1, 1)).to(state.dtype) + self.ptr + self.offset), dim=-1)
            state = state_2dim.reshape(original_shape[:-1] + [state_2dim.shape[-1]])
        return state

    def uncompress_state(self, state):
        if self.parameter.image_input:
            original_shape = list(state.shape)
            state_2dim = state.reshape((-1, state.shape[-1]))
            image_idx = state_2dim[:, -1].to(torch.long).to(self.image_dictionary.device)
            image = self.image_dictionary[image_idx - self.offset]
            state_2dim = torch.cat((state_2dim[:, :-1], image.to(state.device)), dim=-1)
            state = state_2dim.reshape(original_shape[:-1] + [state_2dim.shape[-1]])
        return state

    def uncompress_state_offpolicy(self, state):
        if self.parameter.image_input:
            original_shape = list(state.shape)
            state_2dim = state.reshape((-1, state.shape[-1]))
            image_idx = state_2dim[:, -1].to(torch.long).to(self.image_dictionary_offpolicy.device)
            image = self.image_dictionary_offpolicy[image_idx]
            state_2dim = torch.cat((state_2dim[:, :-1], image.to(state.device)), dim=-1)
            state = state_2dim.reshape(original_shape[:-1] + [state_2dim.shape[-1]])
        return state

    def _get_skip_len(self):
        skip_len = 0
        for rnn_base in [self.values[0].uni_network, self.values[0].embedding_network,
                         self.policy.uni_network, self.policy.embedding_network, self.policy.target_encoder.encoder]:
            for i in range(len(rnn_base.layer_type)):
                if 'smamba' in rnn_base.layer_type[i]:
                    skip_len = max(rnn_base.layer_list[i].d_conv, skip_len)
                elif 'mamba' in rnn_base.layer_type[i]:
                    skip_len = max(rnn_base.layer_list[i].mixer.d_conv, skip_len)
                elif 'conv1d' in rnn_base.layer_type[i]:
                    skip_len = max(rnn_base.layer_list[i].d_conv, skip_len)
        return skip_len + 1

    def reinitialize_env(self):
        try:
            terrain_levels = self.env.terrain_levels
            self.env.close()
        except Exception as e:
            terrain_levels = None
            self.logger(f'close env failed!!')
            self.env.force_close()
        while True:
            try:
                self.env_info = self.env_queue.get()
                self.logger(f'Obtained New Env!')
                if 'save_dir' in self.env_info and self.env_info['save_dir'] is not None:
                    self.logger(f'VISUALIZING MODE..')
                time.sleep(1.0)
                self.env = self.env_info['train_env']
                if terrain_levels is not None:
                    self.env.set_terrain_levels(terrain_levels)
                self.env.reset()
                self.current_env_step = 0
                self.state_np = self.env.get_processed_observation().clone()
                self.priv_state_np = self.env.get_processed_privileged_observation().clone()
                break
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue
        self.dog_cnt = 1

    def allow_nest_stack_trajs(self):
        for rnn_base in [self.values[0].uni_network, self.values[0].embedding_network,
                         self.policy.uni_network, self.policy.embedding_network]:
            for i in range(len(rnn_base.layer_type)):
                if 'transformer' in rnn_base.layer_type[i]:
                    return False
                if 'gru' in rnn_base.layer_type[i]:
                    return False
        return True

    def init_lr_scheduler(self):
        pass

    def _init_sample_hidden(self):
        if not self.parameter.randomize_first_hidden:
            sample_hidden = self.policy.make_init_state(self.parallel_agent_num, self.sample_device)
        else:
            sample_hidden = self.policy.make_rnd_init_state(self.parallel_agent_num, self.sample_device)
        return sample_hidden

    def env_reset(self, reset_buf=None):
        if reset_buf is None:
            self.last_action_np = torch.zeros((self.parallel_agent_num, self.act_dim), device=self.sample_device)
            self.last_state_np = torch.zeros_like(self.state_np)
            self.last_priv_state_np = torch.zeros_like(self.priv_state_np)
            self.reward_np = torch.zeros((self.parallel_agent_num, 1), device=self.sample_device)
            self.sample_hidden = self._init_sample_hidden()
            self.target_encoder_hidden = self.policy.target_encoder.make_init_state(self.parallel_agent_num,
                                                                                    self.sample_device)
        else:
            self.last_action_np[reset_buf] = 0
            self.last_state_np[reset_buf] = 0
            self.last_priv_state_np[reset_buf] = 0
            self.compressed_last_state_np[reset_buf] = 0
            self.reward_np[reset_buf] = 0
            self.sample_hidden.hidden_state_mask_reset_(reset_buf)
            self.target_encoder_hidden.hidden_state_mask_reset_(reset_buf)

    def env_step(self, next_obs, next_obs_compressed, next_priv_obs, act, reward, reset_buf):

        self.last_state_np = self.state_np.clone()
        self.last_priv_state_np = self.priv_state_np.clone()
        self.compressed_last_state_np = self.compressed_state_np.clone()
        self.state_np = next_obs.clone()
        self.priv_state_np = next_priv_obs.clone()
        self.compressed_state_np = next_obs_compressed.clone()
        self.last_action_np = act.clone()
        self.reward_np = reward.clone()
        self.env_reset(reset_buf)

    def _seed(self, seed: int):
        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.cuda.manual_seed_all(seed + 4)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _value_update(self, tau):
        """
            if tau = 0, self <--- src_net
            if tau = 1, self <--- self
        """
        for value, target_value in zip(self.values, self.target_values):
            value: ContextualPPOValue
            target_value: ContextualPPOValue
            target_value.copy_weight_from(value, tau)

    def _make_policy_args(self, parameter: Parameter) -> Dict[str, Union[int, float, List[int]]]:
        policy_args = {
            'state_dim': self.obs_dim,
            'action_dim': self.act_dim,
            'priv_state_dim': self.priv_obs_dim,
            'embedding_size': parameter.policy_embedding_dim,
            'embedding_hidden': parameter.policy_embedding_hidden_size,
            'embedding_activations': parameter.policy_embedding_activations,
            'embedding_layer_type': parameter.policy_embedding_layer_type,
            'uni_model_hidden': parameter.policy_hidden_size,
            'uni_model_activations': parameter.policy_activations,
            'uni_model_layer_type': parameter.policy_layer_type,
            'fix_rnn_length': parameter.rnn_fix_length,
            'reward_input': parameter.reward_input,
            'last_action_input': not parameter.no_last_action_input,
            'last_state_input': hasattr(parameter, 'last_state_input') and parameter.last_state_input,
            'uni_model_input_mapping_dim': parameter.policy_uni_model_input_mapping_dim,
            'separate_encoder': hasattr(parameter, 'state_action_encoder') and parameter.state_action_encoder,
            'use_camera': parameter.image_input if hasattr(parameter, 'image_input') else False,
        }
        if hasattr(parameter, 'base_algorithm') and parameter.base_algorithm in ['td3', 'ppo']:
            policy_args['sample_std'] = parameter.sample_std
            policy_args['std_learnable'] = parameter.std_learnable if hasattr(parameter, 'std_learnable') else False
        if hasattr(parameter, 'policy_embedding_output_activation'):
            policy_args['embedding_output_activation'] = parameter.policy_embedding_output_activation
        if hasattr(parameter, 'target_encoder_rnn_type'):
            policy_args['target_rnn_type'] = parameter.target_encoder_rnn_type
        if hasattr(parameter, 'target_encoder_learn_std'):
            policy_args['target_network_learn_std'] = parameter.target_encoder_learn_std
        if hasattr(parameter, 'target_logstd_input'):
            policy_args['target_logstd_input'] = parameter.target_logstd_input
        if hasattr(parameter, 'square_task'):
            policy_args['use_scan_dot'] = parameter.square_task
        if hasattr(parameter, 'plat_task'):
            policy_args['continuous_vector'] = parameter.plat_task and parameter.square_task
        if hasattr(parameter, 'mean_target_input'):
            policy_args['mean_target_input'] = parameter.mean_target_input
        if hasattr(parameter, 'directly_train_target'):
            policy_args['baseline_mode'] = parameter.directly_train_target
        policy_args['valid_priv_vector_dim'] = 1 if not self.parameter.square_task and not self.parameter.plat_task else 2
        return policy_args

    def _make_value_args(self, parameter: Parameter) -> Dict[str, Union[int, float, List[int]]]:
        value_args = {
            'state_dim': self.priv_obs_dim,
            'action_dim': self.act_dim,
            'embedding_size': parameter.value_embedding_dim,
            'embedding_hidden': parameter.value_embedding_hidden_size,
            'embedding_activations': parameter.value_embedding_activations,
            'embedding_layer_type': parameter.value_embedding_layer_type,
            'uni_model_hidden': parameter.value_hidden_size,
            'uni_model_activations': parameter.value_activations,
            'uni_model_layer_type': parameter.value_layer_type,
            'fix_rnn_length': parameter.rnn_fix_length,
            'reward_input': parameter.reward_input,
            'last_action_input': not parameter.no_last_action_input,
            'last_state_input': hasattr(parameter, 'last_state_input') and parameter.last_state_input,
            'uni_model_input_mapping_dim': parameter.value_uni_model_input_mapping_dim,
            'separate_encoder': hasattr(parameter, 'state_action_encoder') and parameter.state_action_encoder,
            'use_camera': False,
        }
        if hasattr(parameter, 'square_task'):
            value_args['use_scan_dot'] = parameter.square_task
        if hasattr(parameter, 'mean_target_input'):
            value_args['mean_target_input'] = parameter.mean_target_input
        if hasattr(parameter, 'target_logstd_input'):
            value_args['target_logstd_input'] = parameter.target_logstd_input
        return value_args

    def train_one_batch(self) -> Dict:
        return {}

    def process_one_epoch_data(self, one_epoch_data: torch.Tensor):
        parallel_agent_num = one_epoch_data.shape[0]
        name2range = self.replay_buffer.name2range
        mask_idx = name2range['mask'][0]
        start_idx = name2range['start'][0]
        timeout_idx = name2range['timeout'][0]
        done_idx = name2range['done'][0]

        # 如果当前步timeout是1，且done是0，说明下一步timeout也是1，done是1，这个时候需要把下一步mask->0，这一步的done->1，
        one_epoch_data = one_epoch_data.reshape((-1, one_epoch_data.shape[-1]))
        timeout_timestep = \
        torch.where(torch.bitwise_and(one_epoch_data[:, timeout_idx] == 2, one_epoch_data[:, done_idx] == 1))[0]
        timeout_timestep2 = \
        torch.where(torch.bitwise_and(one_epoch_data[:, timeout_idx] == 1, one_epoch_data[:, done_idx] == 1))[0]
        one_epoch_data[timeout_timestep, mask_idx] = 0
        one_epoch_data[timeout_timestep, done_idx] = 0
        one_epoch_data[timeout_timestep, start_idx] = 1
        one_epoch_data[timeout_timestep, timeout_idx] = 1
        one_epoch_data[timeout_timestep2, timeout_idx] = 0
        one_epoch_data[timeout_timestep - 1, done_idx] = 1

        one_epoch_data[one_epoch_data[:, timeout_idx] == 2, timeout_idx] = 1
        one_epoch_data = one_epoch_data.reshape((parallel_agent_num, -1, one_epoch_data.shape[-1]))

        # 最后一步不为done，则需要丢弃，mask设为零

        one_epoch_data[one_epoch_data[:, -1, done_idx] == 0, -1, mask_idx] = 0

        # 所有的最后一步都设done=True，这样每条轨迹的start和done的数量必然相等，但会存在最后一步既是start又是done的情况，后续要把这种情况舍去（把done=1, mask=0的轨迹舍去即可）
        one_epoch_data[:, -1, done_idx] = 1
        # 把所有轨迹都拼成一条长轨迹
        one_epoch_data = one_epoch_data.reshape((-1, one_epoch_data.shape[-1]))
        done_time_step = torch.where(one_epoch_data[:, done_idx] > 0)[0]
        start_time_step = torch.where(one_epoch_data[:, start_idx] > 0)[0]
        if not done_time_step.shape[0] == start_time_step.shape[0]:
            self.logger.log(f'done_time_step: {[(i, float(item)) for i, item in enumerate(done_time_step)]}')
            self.logger.log(f'start_time_step: {[(i, float(item)) for i, item in enumerate(start_time_step)]}')
        valid_done_mask = one_epoch_data[done_time_step, mask_idx] == 1
        done_time_step = done_time_step[valid_done_mask]
        start_time_step = start_time_step[valid_done_mask]

        traj_start_time_step = start_time_step
        traj_end_time_step = done_time_step + 1
        traj_length = traj_end_time_step - traj_start_time_step

        state_image_idx = name2range['state'][-1]
        last_state_image_idx = name2range['last_state'][-1]
        next_state_image_idx = name2range['next_state'][-1]
        target_flag_idx = name2range['target_flag'][0]
        one_epoch_data = one_epoch_data.detach().cpu().numpy()
        return one_epoch_data, traj_length, traj_start_time_step, traj_end_time_step

    def add_to_buffer(self, one_epoch_data: torch.Tensor, data_type='onpolicy'):
        name2range = self.replay_buffer.name2range
        mask_idx = name2range['mask'][0]
        start_idx = name2range['start'][0]
        timeout_idx = name2range['timeout'][0]
        done_idx = name2range['done'][0]

        state_image_idx = name2range['state'][-1]
        last_state_image_idx = name2range['last_state'][-1]
        next_state_image_idx = name2range['next_state'][-1]
        target_flag_idx = name2range['target_flag'][0]
        target_data = one_epoch_data[:, 0, target_flag_idx]

        if data_type == 'onpolicy':
            one_epoch_data = one_epoch_data[target_data==0, :]
        elif data_type == 'offpolicy':
            one_epoch_data = one_epoch_data[target_data==1, :]
        else:
            pass
        print(data_type, one_epoch_data.shape)

        one_epoch_data, traj_length, traj_start_time_step, traj_end_time_step = self.process_one_epoch_data(one_epoch_data)

        for i in range(traj_length.shape[0]):
            start, end = int(traj_start_time_step[i]), int(traj_end_time_step[i])
            length = int(traj_length[i])
            one_traj_data = one_epoch_data[start:end, :]
            if data_type == 'onpolicy':
                self.replay_buffer.trajectory_start.append(self.replay_buffer.ptr)
                self.replay_buffer.memory_buffer[self.replay_buffer.ptr:self.replay_buffer.ptr+length] = one_traj_data.copy()
                self.replay_buffer.ptr += length
                self.replay_buffer.trajectory_length.append(length)
                self.replay_buffer.transition_count += length
            if random.random() < self.offpolicy_reserve_prob: # offpolicy replay buffer
                if one_traj_data[0, target_flag_idx] and one_traj_data[-1, target_flag_idx] and data_type == 'offpolicy':
                    if self.parameter.image_input:
                        codes = np.concatenate((
                            one_traj_data[:, state_image_idx],
                            one_traj_data[:, last_state_image_idx],
                            one_traj_data[:, next_state_image_idx],
                        ), axis=0)
                        codes_unique = np.unique(codes).astype(np.int64)
                        code_book = self.image_dictionary[codes_unique]
                        codes_len = len(codes_unique)
                        if self.image_dictionary_offpolicy is None:
                            self.image_dictionary_offpolicy = torch.zeros(
                                (self.image_offpolicy_maximum_cnt + 1, self.policy.image_dim),
                                device=torch.device('cpu'))
                            self.logger(
                                f'creating OFFPOLICY image dictionary with shape: {self.image_dictionary_offpolicy.shape}')
                        if self.image_offpolicy_cnt + codes_len >= self.image_offpolicy_maximum_cnt:
                            self.image_offpolicy_cnt = 1
                        self.image_dictionary_offpolicy[self.image_offpolicy_cnt:self.image_offpolicy_cnt + codes_len, :] = code_book.clone().to(self.image_dictionary_offpolicy.device)
                        codes_indices_tmp = np.zeros((self.image_dictionary.shape[0],))
                        codes_indices_tmp[codes_unique] = np.arange(codes_len) + self.image_offpolicy_cnt
                        codes_new = codes_indices_tmp[codes.astype(np.int64)]
                        self.image_offpolicy_cnt += codes_len
                        one_traj_data[:, state_image_idx] = codes_new[:one_traj_data.shape[0]]
                        one_traj_data[:, last_state_image_idx] = codes_new[one_traj_data.shape[0]:(2 * one_traj_data.shape[0])]
                        one_traj_data[:, next_state_image_idx] = codes_new[2*one_traj_data.shape[0]:]
                    if self.offpolicy_replay_buffer.memory_buffer is None:
                        transition_start = self.replay_buffer.array_to_transition(one_traj_data[:1, :])
                        self.offpolicy_replay_buffer._init_memory_buffer(transition_start)
                    self.offpolicy_replay_buffer.traj_push(one_traj_data)

        # self.trajectory_start.append(self.ptr)
        # for ind, transition in enumerate(memory):
        #     self.memory_buffer[self.ptr] = 0
        #     self._insert_transition(transition)
        #     self.ptr += 1
        # self.trajectory_length.append(traj_len)
        # self.transition_count += len(memory)
        pass

    def get_schedule_weight(self):
        schedule_range = 600 if (self.parameter.square_task
                                 and not self.parameter.plat_task) or (self.parameter.plat_task
                                                                      and not self.parameter.square_task) else 300
        if not self.parameter.square_task and self.parameter.plat_task:
            if self.parameter.image_input:
                schedule_range = 300
            else:
                schedule_range = 600
        iter_train = self.iter_train - self.threshold_iter
        weight = (1 - np.cos(np.pi * iter_train / schedule_range)) / 2
        if iter_train >= schedule_range:
            weight = 1
        if iter_train <= 0:
            weight = 0
        return weight

    def get_target_sampling_schedule_weight(self):
        schedule_range = 100
        iter_train = self.iter_train - self.threshold_iter
        weight = (1 - np.cos(np.pi * iter_train / schedule_range)) / 2
        if iter_train >= schedule_range:
            weight = 1
        if iter_train <= 0:
            weight = 0
        return weight

    def get_lr_schedule_weight(self):
        schedule_range = 3000
        iter_train = self.iter_train - self.threshold_iter
        weight = (1 - np.cos(np.pi * iter_train / schedule_range)) / 2
        if iter_train >= schedule_range:
            weight = 1
        if iter_train <= 0:
            weight = 0
        return weight

    def train(self):
        initial_lr = [param_group['lr'] for param_group in self.optimizer_policy.param_groups]
        initial_lr_value = [param_group['lr'] for param_group in self.optimizer_value.param_groups]
        average_traj_length = 1000.0
        total_feature_dim = None
        for iter_train in range(self.parameter.total_iteration):
            # if iter_train == 300:
            #     self.policy.reset_target_encoder()
            self.iter_train = iter_train
            self.logger(f'random test: random: {random.random()}, numpy: {np.random.random()}, torch: {torch.randn((1,))}, torch cuda: {torch.randn((1,), device=self.device).item()}')
            self.policy.to(torch.device('cpu'))
            if self.role == 'master':
                policy_state_dict = copy.deepcopy(self.policy.state_dict())
                policy_state_dict = pickle.dumps(policy_state_dict)
                for i in range(self.total_client_num):
                    self.logger(f'try to put policy state dict to client: {i + 1} len: {len(policy_state_dict)}')
                    self.redis.set(f'model_{i+1}', policy_state_dict)
                agent_idx = 0
            else:
                self.logger(f'waiting model queue...')
                agent_idx = self.machine_id
                while not self.redis.exists(f'model_{agent_idx}'):
                    time.sleep(0.1)
                policy_state_dict = self.redis.get(f'model_{agent_idx}')
                self.redis.delete(f'model_{agent_idx}')
                policy_state_dict_str = policy_state_dict
                policy_state_dict = pickle.loads(policy_state_dict_str)
                self.logger(f'client: {agent_idx} obtained policy_state_dict, len: {len(policy_state_dict_str)}')
                self.policy.load_state_dict(policy_state_dict)
            if not self.parameter.std_learnable:
                std_weight = 1 - self.get_lr_schedule_weight()
                maximum_sample_std = self.parameter.sample_std
                minimum_sample_std = self.parameter.sample_std / 5.0
                current_sample_std = (maximum_sample_std - minimum_sample_std) * std_weight + minimum_sample_std
                self.policy.log_std = np.log(current_sample_std)
            for i, param_group in enumerate(self.optimizer_policy.param_groups):
                coeff = 20.0
                param_group['lr'] = ((coeff - 1) * (1 - self.get_lr_schedule_weight()) + 1) * initial_lr[i] / coeff
            for i, param_group in enumerate(self.optimizer_value.param_groups):
                coeff = 10.0
                param_group['lr'] = ((coeff - 1) * (1 - self.get_lr_schedule_weight()) + 1) * initial_lr_value[i] / coeff

            self.policy.train()
            self.policy.to(self.sample_device)
            # if iter_train > 0:
            self.env_reset()
            self.reset_compression(agent_idx)
            simulator_healthy = True

            try:
                self.env.reset()
                self.state_np = self.env.get_processed_observation()
                self.priv_state_np = self.env.get_processed_privileged_observation()
            except Exception as e:
                self.logger(f'simulator failed, skipping...')
                simulator_healthy = False
            self.compressed_state_np = self.compress_state(self.state_np, 0, self.sample_device)
            self.compressed_last_state_np = torch.zeros_like(self.compressed_state_np)
            batch_ep_ret = torch.zeros((self.parallel_agent_num, ), device=self.sample_device)
            batch_last_logstd = torch.zeros((self.parallel_agent_num, ), device=self.sample_device)
            batch_last_smoothed_std = None
            batch_ep_ret_original = torch.zeros((self.parallel_agent_num, ), device=self.sample_device)
            batch_ep_ret_ebd_diff = 0
            batch_ep_len = torch.zeros((self.parallel_agent_num,), device=self.sample_device)
            batch_ep_len_no_reset = torch.zeros((self.parallel_agent_num,), device=self.sample_device)
            batch_start_cnt = torch.zeros((self.parallel_agent_num,), device=self.sample_device)
            traj_buffer = None if total_feature_dim is None else torch.zeros((self.parallel_agent_num, self.max_episode_steps,
                                               total_feature_dim), device=self.sample_device)

            minimal_target_logstd = 1e9
            maximal_target_logstd = -1e9
            subseq_embedding_noise = None
            reward_factor = 1.0
            if self.parameter.directly_train_target:
                reward_factor = 0.0
            test_epoch = iter_train > 10 and iter_train % 40 == 0
            # start training
            target_flag_static = torch.rand((self.parallel_agent_num,1,1), device=self.sample_device) < (0.25 if not self.parameter.directly_train_target else 0.0) # TODO 0310
            if test_epoch:
                target_flag_static[:] = False
            target_flag_middle_change_ = torch.rand((self.parallel_agent_num,1,1), device=self.sample_device) < -1.0

            target_flag_middle_change = target_flag_middle_change_.clone()
            middle_change_prob = 1 / (average_traj_length / 2)
            target_indices = torch.where(target_flag_static)[0]
            target_indices_shuffle = target_indices[torch.randperm(target_indices.shape[0])]
            rnd_signs = None
            rnd_values = None
            reset_num_sum = 0
            for step in tqdm(range(self.max_episode_steps), dynamic_ncols=True):
                if not simulator_healthy:
                    break
                target_flag_middle_change = target_flag_middle_change & (torch.rand(target_flag_middle_change.shape, device=self.sample_device) >= middle_change_prob)
                target_flag = target_flag_static | target_flag_middle_change
                # sample from env and save to replay buffer
                self.timer.register_point(tag='sample_in_env', level=1)
                if not self.parameter.directly_train_target:
                    with torch.no_grad():
                        # TODO!!! 3-dim data forward bug!!
                        target_embedding, target_logstd, self.target_encoder_hidden = self.policy.get_target_embedding(self.uncompress_state(self.compressed_state_np.unsqueeze(-2)), self.last_action_np.unsqueeze(-2), self.target_encoder_hidden)
                        if batch_last_smoothed_std is None:
                            batch_last_smoothed_std = torch.zeros_like(target_logstd) + 0.5
                        batch_last_smoothed_std = batch_last_smoothed_std * 0.95 + 0.05 * target_logstd.exp()
                        target_logstd = torch.log(batch_last_smoothed_std)
                        # TODO shuffle
                        priv_state_np_shuffled = self.priv_state_np.unsqueeze(-2).clone()
                        if self.parameter.square_task:
                            priv_state_np_shuffled[target_indices_shuffle, :, -134:-132] = priv_state_np_shuffled[target_indices, :, -134:-132].clone()
                        else:
                            priv_state_np_shuffled[target_indices_shuffle, :, -2:] = priv_state_np_shuffled[target_indices, :, -2:].clone()

                        privileged_embedding, _ = self.policy.get_privileged_embedding(priv_state_np_shuffled, None)
                        minimal_target_logstd = min(minimal_target_logstd, target_logstd.min().item())
                        maximal_target_logstd = max(maximal_target_logstd, target_logstd.max().item())
                        if self.get_lr_schedule_weight() > 0 and not self.parameter.no_noise_perturbation and not self.parameter.encoder_bc_policy_rl:
                            noise_size = 4.0 * self.get_schedule_weight()

                            if self.parameter.plat_task and self.parameter.square_task:
                                if rnd_values is None or step % 10 == 0:
                                    rnd_values = torch.rand((self.parallel_agent_num, 1, privileged_embedding.shape[-1]),
                                                   device=self.sample_device) * 2 - 1
                                embedding_noise = ((privileged_embedding + noise_size * rnd_values * target_logstd.exp()) * target_flag).squeeze(
                                    dim=-2) + (rnd_values * target_logstd.exp() * noise_size * (1 - target_flag.float())).squeeze(dim=-2)
                            else:
                                if self.parameter.square_task:
                                    if rnd_signs is None or step % 10 == 0:
                                        rnd_signs = torch.sign(
                                            0.125 - torch.rand((self.parallel_agent_num, 1, privileged_embedding.shape[-1]),
                                                              device=self.sample_device) * target_logstd.exp().clamp_max(0.25 * self.get_schedule_weight()))

                                else:
                                    if rnd_signs is None or step % 10 == 0:
                                        rnd_signs = torch.sign(
                                            0.09 - torch.rand((self.parallel_agent_num, 1, privileged_embedding.shape[-1]),
                                                              device=self.sample_device) * target_logstd.exp().clamp_max(
                                                0.18 * self.get_schedule_weight()))
                                desire_embedding = privileged_embedding * rnd_signs
                                embedding_noise = target_flag * desire_embedding + (1 - target_flag.float()) * (
                                            desire_embedding - privileged_embedding)
                                embedding_noise = embedding_noise.squeeze(-2)
                        else:
                            embedding_noise = (privileged_embedding * target_flag).squeeze(dim=-2)
                            # embedding_noise = (target_embedding * target_flag).squeeze(dim=-2)
                        if test_epoch:
                            embedding_noise = (target_embedding - privileged_embedding).squeeze(dim=-2)
                        elif self.parameter.encoder_bc_policy_rl:
                            embedding_noise = ((target_embedding - privileged_embedding) * (1 - target_flag.float()) + target_flag * target_embedding).squeeze(dim=-2)
                        if subseq_embedding_noise is None:
                            subseq_embedding_noise = torch.zeros((self.parallel_agent_num, target_embedding.shape[-1] * 2), device=self.sample_device)
                        if embedding_noise is not None:
                            subseq_embedding_noise = torch.cat((subseq_embedding_noise[..., -target_embedding.shape[-1]:], embedding_noise), dim=-1)
                        MIN_LOGSTD, MAX_LOGSTD = -10, 1
                        target_logstd_norm = (torch.clamp(target_logstd, min=MIN_LOGSTD, max=MAX_LOGSTD) - MIN_LOGSTD) / (MAX_LOGSTD - MIN_LOGSTD) * 2 - 1
                        target_logstd_norm = target_logstd_norm * reward_factor
                        if self.parameter.mean_target_input:
                            target_logstd_norm = torch.cat((target_logstd_norm, target_embedding), dim=-1)
                        act_mean, embedding_output, act_sample, logp_old, self.sample_hidden, _ = self.policy.forward(
                            # state=self.uncompress_state(self.compressed_state_np.unsqueeze(-2)),
                            # lst_state=self.uncompress_state(self.compressed_last_state_np.unsqueeze(-2)),
                            state=self.priv_state_np.unsqueeze(-2),
                            lst_state=self.last_priv_state_np.unsqueeze(-2),
                            lst_action=self.last_action_np.unsqueeze(-2),
                            rnn_memory=self.sample_hidden,
                            reward=self.reward_np.unsqueeze(-2),
                            detach_embedding=True,
                            embedding_noise=subseq_embedding_noise.unsqueeze(-2),
                            target_logstd=target_logstd_norm,
                            target_flag=target_flag.float(),
                        )
                        if embedding_noise is None:
                            embedding_noise = 0 * embedding_output.squeeze(1)
                        # embedding_diff = (embedding_output - target_embedding).pow(2).max(dim=-1, keepdim=True).values.sqrt()
                        embedding_diff_2 = (embedding_output - target_embedding).squeeze(dim=1)
                        embedding_output_norm = embedding_output.pow(2).mean().sqrt()
                        target_embedding_norm = target_embedding.pow(2).mean().sqrt()

                        embedding_diff = (embedding_output - target_embedding).pow(2).mean(dim=-1, keepdim=True).sqrt()
                        self.logger.add_tabular_data(embedding_diff=embedding_diff.mean().item(),
                                                     embedding_output_norm=embedding_output_norm.item(),
                                                     target_embedding_norm=target_embedding_norm.item())
                        if torch.any(target_flag):
                            self.logger.add_tabular_data(embedding_diff_tgt=((embedding_diff * target_flag).sum() / target_flag.sum()).item())
                        self.logger.add_tabular_data(target_agent_num=target_flag.sum().item())
                        self.logger.add_tabular_data(target_agent_static_num=target_flag_static.sum().item())
                        self.logger.add_tabular_data(target_agent_middle_num=target_flag_middle_change.sum().item())

                        if target_logstd is not None:
                            self.logger.add_tabular_data(target_std=target_logstd.exp().mean().item())
                        act_mean, act_sample, logp_old = map(lambda x: x[:, 0], [
                            act_mean, act_sample, logp_old
                        ])
                        act = act_sample
                        if test_epoch:
                            act = act_mean
                else:
                    with torch.no_grad():
                        act_mean, embedding_output, act_sample, logp_old, self.sample_hidden, _ = self.policy.forward(
                            state=self.uncompress_state(self.compressed_state_np.unsqueeze(-2)),
                            lst_state=self.uncompress_state(self.compressed_last_state_np.unsqueeze(-2)),
                            # state=self.priv_state_np.unsqueeze(-2),
                            # lst_state=self.last_priv_state_np.unsqueeze(-2),
                            lst_action=self.last_action_np.unsqueeze(-2),
                            rnn_memory=self.sample_hidden,
                            reward=self.reward_np.unsqueeze(-2),
                            detach_embedding=True,
                            # embedding_noise=subseq_embedding_noise.unsqueeze(-2),
                            # target_logstd=target_logstd_norm,
                        )
                        target_logstd_norm = torch.zeros_like(embedding_output)
                        privileged_embedding = torch.zeros_like(embedding_output)
                        subseq_embedding_noise = torch.zeros_like(embedding_output).repeat_interleave(2, dim=-1)
                        target_embedding = torch.zeros_like(embedding_output)
                        target_logstd = torch.zeros_like(embedding_output)
                        embedding_noise = torch.zeros_like(embedding_output).squeeze(-2)
                        batch_last_smoothed_std = torch.zeros_like(target_logstd) + 0.5
                        embedding_diff = torch.ones_like(embedding_output).mean(dim=-1, keepdim=True)
                        act_mean, act_sample, logp_old = map(lambda x: x[:, 0], [
                            act_mean, act_sample, logp_old
                        ])
                        act = act_sample

                # self.timer.register_end(level=1)
                self.timer.register_point(tag='env_step', level=1)
                try:
                    if self.role == 'master':
                        noised_embedding = (1 - target_flag.float()) * (embedding_noise.unsqueeze(-2) + privileged_embedding) + target_flag * embedding_noise.unsqueeze(-2)
                        next_state, next_priv_state, reward, done, info = self.env.step_with_ebd_diff(act,
                                                                                                      embedding_diff.squeeze(-1),
                                                                                                      target_logstd.exp().mean(dim=-1),
                                                                                                      target_flag,
                                                                                                      batch_ep_ret,
                                                                                                      target_embedding,
                                                                                                      privileged_embedding,
                                                                                                      noised_embedding)
                    else:
                        next_state, next_priv_state, reward, done, info = self.env.step(act)
                    # next_state, next_priv_state, reward, done, info = self.env.step_with_ebd_diff(act,
                    #                                                                               embedding_diff.squeeze(-1),
                    #                                                                               target_logstd.exp().mean(dim=-1),
                    #                                                                               target_flag)

                except Exception as e:
                    print(
                        f'embedding_diff: {embedding_diff.shape}, target_logstd: {target_logstd.shape}, '
                        f'target_flag: {target_flag.shape}, batch_ep_ret: {batch_ep_ret.shape}, '
                        f'target_embedding: {target_embedding.shape}, '
                        f'privileged_embedding: {privileged_embedding.shape}')
                    self.logger(f'environment Error!!!')
                    simulator_healthy = False
                    import traceback
                    traceback.print_exc()
                    break
                batch_ep_ret_original += reward
                with torch.no_grad():
                    # embedding_diff_reward = torch.clamp_min(0.02 - embedding_diff.squeeze(1).squeeze(1), 0.0) / 0.02
                    # if self.target_encoder_learn_std:
                    if not self.parameter.directly_train_target and not self.parameter.no_reward_modification:
                        current_logstd = target_logstd.mean(dim=-1, keepdim=False).squeeze(1)
                        logstd_delta = batch_last_logstd - current_logstd
                        maximal_bound = 0.2
                        minimal_bound = 0.01
                        embedding_diff_reward = torch.clamp(-((target_logstd.exp().mean(dim=-1, keepdim=False).squeeze(1)).log() - np.log(maximal_bound)), 0, -(np.log(minimal_bound) - np.log(maximal_bound)))
                        reward_factor_2 = (embedding_diff_reward + 0.05)
                        reward_factor_2 = (1 - reward_factor_2) * (1 - self.get_schedule_weight()) + reward_factor_2
                        reward = reward * reward_factor_2 #  * self.get_lr_schedule_weight()
                        batch_last_logstd = current_logstd
                    else:
                        embedding_diff_reward = 0 * reward
                        pass
                        # reward = reward + embedding_diff_reward * 0.0
                    # reward = torch.min(reward, embedding_diff_reward)
                batch_ep_ret += reward
                batch_ep_ret_ebd_diff += embedding_diff_reward
                episode_length_buf = info['episode_length_buf']
                batch_ep_ret = batch_ep_ret * (episode_length_buf != 1)
                batch_ep_ret_ebd_diff = batch_ep_ret_ebd_diff * (episode_length_buf != 1)
                batch_ep_ret_original = batch_ep_ret_original * (episode_length_buf != 1)
                batch_ep_len = batch_ep_len + 1
                batch_ep_len_no_reset = batch_ep_len_no_reset + 1

                reset_buf = done
                next_state_compress = self.compress_state(next_state, step+1, self.sample_device)
                self.timer.register_point(tag='mem_push', level=1)
                mask = torch.ones_like(reward).reshape((self.parallel_agent_num, -1))
                # if self.iter_train <= self.threshold_iter + 5:
                #     mask[target_flag.view((-1,))] = 0
                # timeout_data = (batch_ep_len >= self.max_episode_steps-1).float().reshape((self.parallel_agent_num, -1))
                timeout_data = (batch_ep_len - self.max_episode_steps + 2).float().reshape((self.parallel_agent_num, -1))
                timeout_data = torch.clamp_min(timeout_data, 0)
                # timeout_data_2 = (batch_ep_len >= self.max_episode_steps-1).float().reshape((self.parallel_agent_num, -1))
                done_data = done.float().reshape((self.parallel_agent_num, -1))
                start_flag = (batch_ep_len == 1).float().reshape((self.parallel_agent_num, -1))

                # inference the next logstd
                # with torch.no_grad():
                #     hidden_next = copy.deepcopy(self.target_encoder_hidden)
                #     _, target_logstd_next, _ = self.policy.get_target_embedding(
                #         self.uncompress_state(next_state_compress.unsqueeze(-2)), act.unsqueeze(-2),
                #         hidden_next)
                #     next_target_logstd_norm = (torch.clamp(target_logstd_next, min=MIN_LOGSTD, max=MAX_LOGSTD) - MIN_LOGSTD) / (
                #                 MAX_LOGSTD - MIN_LOGSTD) * 2 - 1
                #     next_target_logstd_norm = next_target_logstd_norm * reward_factor
                # --------
                # additional_flag = (batch_ep_len_no_reset <= batch_ep_len).float().reshape((self.parallel_agent_num, -1))
                # mask = mask * additional_flag
                # timeout_data = timeout_data * additional_flag
                # done_data = done_data * additional_flag
                # start_flag = (batch_ep_len == 1).float().reshape((self.parallel_agent_num, -1))
                # batch_start_cnt = batch_start_cnt + start_flag
                # start_flag = (batch_start_cnt <= 2).float().reshape((self.parallel_agent_num, -1)) * start_flag
                # --------
                # if torch.any(timeout_data):
                #     check_result = torch.sum(done_data * timeout_data) == torch.sum(timeout_data)
                #     self.logger(f'checking timeout valid: {check_result}')
                # if torch.any(timeout_data_2):
                #     check_result_2 = torch.sum(done_data * timeout_data_2) == torch.sum(timeout_data_2)
                #     self.logger(f'checking timeout valid type2: {check_result_2}')

                one_step_data = Transition(
                    state = self.compressed_state_np,
                    priv_state = self.priv_state_np,
                    last_state = self.compressed_last_state_np,
                    last_priv_state = self.last_priv_state_np,
                    last_action = self.last_action_np,
                    expert_action=None,
                    action = act,
                    next_state = next_state_compress,
                    next_priv_state=next_priv_state,
                    reward = reward.reshape((self.parallel_agent_num, -1)),
                    logp=logp_old.reshape((self.parallel_agent_num, -1)),
                    mask=mask,
                    done=done_data,
                    timeout=timeout_data, # 1 or 2
                    start=start_flag,
                    reward_input=self.reward_np,
                    target_embedding=target_embedding.squeeze(1),
                    embedding_noise=subseq_embedding_noise,
                    target_logstd=target_logstd_norm.squeeze(1),
                    target_flag=target_flag.view((-1, 1)),
                    next_target_logstd=target_logstd_norm.squeeze(1),
                )
                one_step_data_tensor = torch.cat([item.reshape((self.parallel_agent_num, -1))
                                                  for item in one_step_data if item is not None], dim=-1)
                # print(one_step_data_tensor.shape, [item.shape for item in one_step_data if item is not None])
                # exit(0)

                if traj_buffer is None:
                    traj_buffer = torch.zeros((self.parallel_agent_num, self.max_episode_steps,
                                               one_step_data_tensor.shape[-1]), device=self.sample_device)
                    total_feature_dim = one_step_data_tensor.shape[-1]
                    self.logger(f'constructed trajectory buffer: {traj_buffer.shape}, device: {traj_buffer.device}')
                if self.replay_buffer.memory_buffer is None:
                    self.replay_buffer._init_memory_buffer(one_step_data)
                with torch.no_grad():
                    traj_buffer[:, step, :] = one_step_data_tensor.detach().clone()

                self.timer.register_end(level=1)

                if torch.any(reset_buf):
                    reset_num = int(reset_buf.sum().item())
                    reset_num_sum += reset_num
                    reset_buf_target = reset_buf * target_flag.reshape((-1,))
                    reset_buf_target_num = int(reset_buf_target.sum().item())

                    # self.logger(f'reset_num: {reset_num}')
                    if self.dog_cnt < 0:
                        if not test_epoch:
                            self.logger.add_tabular_data(tb_prefix='eval',
                                                         EpRet=[((reset_buf * batch_ep_ret).sum() / reset_buf.sum()).item()] * reset_num,
                                                         EpRetOriginal=[((reset_buf * batch_ep_ret_original).sum() / reset_buf.sum()).item()] * reset_num,
                                                         EpLen=[((reset_buf * batch_ep_len).sum() / reset_buf.sum()).item()] * reset_num,
                                                         EpEbdRet=[((reset_buf * batch_ep_ret_ebd_diff).sum() / reset_buf.sum()).item()] * reset_num,
                                                         )

                            if torch.any(reset_buf_target):
                                self.logger.add_tabular_data(tb_prefix='eval', EpRetTgt=[(
                                            (reset_buf_target * batch_ep_ret).sum() / reset_buf_target.sum()).item()] * reset_buf_target_num,
                                                             EpRetOriginalTgt=[((reset_buf_target * batch_ep_ret_original).sum() / reset_buf_target.sum()).item()] * reset_buf_target_num,
                                                             EpLenTgt=[((reset_buf_target * batch_ep_len).sum() / reset_buf_target.sum()).item()] * reset_buf_target_num,
                                                             EpEbdRetTgt=[((reset_buf_target * batch_ep_ret_ebd_diff).sum() / reset_buf_target.sum()).item()] * reset_buf_target_num)
                                average_traj_length = 0.97 * average_traj_length + 0.03 * ((reset_buf_target * batch_ep_len).sum() / reset_buf_target.sum()).item()

                            for k, v in info['episode'].items():
                                if k.startswith('rew'):
                                    self.logger.add_tabular_data(tb_prefix='eval',
                                                                 **{k: [float(v) * self.env_max_episode_length_s] * reset_num})
                                if k == 'goal_idx':
                                    # self.logger(f'goal_idx: {float(v)}, sum: {reset_num * float(v)}')
                                    self.logger.add_tabular_data(tb_prefix='eval', goal_idx=[float(v)] * reset_num)
                                if k == 'terrain_level':
                                    self.logger.add_tabular_data(tb_prefix='eval', terrain_level=[float(v)] * reset_num)
                            if (reset_num - reset_buf_target_num) > 0:
                                num = reset_num - reset_buf_target_num
                                reset_buf_non_target = reset_buf.float() - reset_buf_target.float()
                                self.logger.add_tabular_data(tb_prefix='eval',
                                                             EpRetNonTgt=[((reset_buf_non_target * batch_ep_ret).sum() / reset_buf_non_target.sum()).item()] * num,
                                                             EpRetOriginalNonTgt=[((reset_buf_non_target * batch_ep_ret_original).sum() / reset_buf_non_target.sum()).item()] * num,
                                                             EpLenNonTgt=[((reset_buf_non_target * batch_ep_len).sum() / reset_buf_non_target.sum()).item()] * num,
                                                             EpEbdRetNonTgt=[((reset_buf_non_target * batch_ep_ret_ebd_diff).sum() / reset_buf_non_target.sum()).item()] * num,
                                                             )
                        else:
                            self.logger.add_tabular_data(tb_prefix='eval',
                                                        EpRetTest=[((reset_buf * batch_ep_ret).sum() / reset_buf.sum()).item()] * reset_num,
                                                        EpRetOriginalTest=[((reset_buf * batch_ep_ret_original).sum() / reset_buf.sum()).item()] * reset_num,
                                                        EpLenTest=[((reset_buf * batch_ep_len).sum() / reset_buf.sum()).item()] * reset_num,
                                                        EpEbdRetTest=[((reset_buf * batch_ep_ret_ebd_diff).sum() / reset_buf.sum()).item()] * reset_num,
                                                        )
                            for k, v in info['episode'].items():
                                if k.startswith('rew'):
                                    self.logger.add_tabular_data(tb_prefix='eval',
                                                                 **{k+"Test": [float(v) * self.env_max_episode_length_s] * reset_num})
                                if k == 'goal_idx':
                                    self.logger.add_tabular_data(tb_prefix='eval', goal_idx_test=[float(v)] * reset_num)
                                if k == 'terrain_level':
                                    self.logger.add_tabular_data(tb_prefix='eval', terrain_level_test=[float(v)] * reset_num)
                    batch_ep_ret = batch_ep_ret * (1 - reset_buf.float())
                    batch_ep_ret_ebd_diff = batch_ep_ret_ebd_diff * (1 - reset_buf.float())
                    batch_ep_ret_original = batch_ep_ret_original * (1 - reset_buf.float())
                    batch_ep_len[reset_buf] = 0
                    batch_last_logstd[reset_buf] = 0
                    batch_last_smoothed_std[reset_buf] = 0.5
                    subseq_embedding_noise[reset_buf] = 0
                    target_flag_middle_change[reset_buf] = target_flag_middle_change_[reset_buf].clone()
                    # test start
                    # self.replay_buffer.sample_trajs(2)

                self.current_env_step += 1
                self.env_step(next_state, next_state_compress, next_priv_state, act, reward, reset_buf)
                self.sample_num += self.parallel_agent_num * (1 + self.total_client_num)
            if not simulator_healthy:
                self.logger(f'simulator failed, skipping....')
            self.policy.to(self.device)
            if self.role == 'master':
                redis_start_time = time.time()
                sample_results = [None for _ in range(self.total_client_num)]
                def obtain_from_redis(_clien_idx):
                    sample_result = {}
                    while not self.redis.exists(f'traj_buffer_{_clien_idx}') or not self.redis.exists(
                            f'image_dictionary_chunk_num_{_clien_idx}'):
                        time.sleep(0.1)
                    sample_result['traj_buffer'] = pickle.loads(self.redis.get(f'traj_buffer_{_clien_idx}')).to(self.device)
                    image_dict_ser = b''
                    for j in range(pickle.loads(self.redis.get(f'image_dictionary_chunk_num_{_clien_idx}'))):
                        image_dict_ser += self.redis.get(f'image_dictionary_{_clien_idx}_{j}')
                        self.redis.delete(f'image_dictionary_{_clien_idx}_{j}')
                    if self.parameter.image_input:
                        sample_result['image_dictionary'] = pickle.loads(image_dict_ser).to(self.device)
                    sample_results[_clien_idx-1] = sample_result
                    self.redis.delete(f'traj_buffer_{_clien_idx}')
                    self.redis.delete(f'image_dictionary_chunk_num_{_clien_idx}')
                if self.total_client_num >= 1:
                    with ThreadPoolExecutor(max_workers=self.total_client_num) as executor:
                        _ = list(executor.map(obtain_from_redis, range(1, self.total_client_num + 1)))
                if len(sample_results) > 0:
                    traj_buffer = torch.cat([traj_buffer.to(self.device)] + [item['traj_buffer'] for item in sample_results], dim=0)
                else:
                    traj_buffer = traj_buffer.to(self.device)
                if self.parameter.image_input:
                    self.image_dictionary = torch.cat([self.image_dictionary.to(self.device)] + [item['image_dictionary'] for item in sample_results], dim=0)
                self.logger(f'obtained data from REDIS! cost: {time.time() - redis_start_time:.2f}s')
            else:
                self.redis.set(f'traj_buffer_{self.machine_id}', pickle.dumps(traj_buffer.to(torch.device('cpu'))))
                if self.parameter.image_input:
                    dict_ser = pickle.dumps(self.image_dictionary.to(torch.device('cpu')))
                    chunk_size = 400000000
                    chunk_num = int(np.ceil(len(dict_ser) / chunk_size))
                    for _ in range(3):
                        try:
                            for j in range(chunk_num):
                                self.redis.set(f'image_dictionary_{self.machine_id}_{j}', dict_ser[j*chunk_size:(j+1)*chunk_size])
                            self.redis.set(f'image_dictionary_chunk_num_{self.machine_id}', pickle.dumps(chunk_num))
                            break
                        except Exception as e:
                            self.logger(f'length {len(dict_ser)} failed')
                else:
                    self.redis.set(f'image_dictionary_chunk_num_{self.machine_id}', pickle.dumps(0))

            # PPO training
            if self.dog_cnt < 0 and simulator_healthy and self.role == 'master' and self.env_info['save_dir'] is None and not test_epoch:
                try:
                    self.add_to_buffer(traj_buffer, 'onpolicy')
                    if not self.parameter.directly_train_target:
                        self.add_to_buffer(traj_buffer, 'offpolicy')
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    simulator_healthy = False
                del traj_buffer
                traj_buffer = None
                torch.cuda.empty_cache()
                self.logger(f'empty cache!!')
                if simulator_healthy:
                    training_log = self.train_one_batch()
                # (luofm comment): periodically reset the target encoder (it's hard to always track a changing encoder)
                # if self.iter_train % 250 == 0:
                #     self.policy.reset_target_encoder()
            else:
                training_log = {}
            del traj_buffer
            torch.cuda.empty_cache()
            traj_buffer = None
            if self.current_env_step > 5000 or not simulator_healthy or self.env_info['save_dir'] is not None:
                # TODO reinitialize env
                self.reinitialize_env()
                # batch_ep_ret[:] = 0
                # batch_ep_len[:] = 0
                # batch_ep_ret_ebd_diff[:] = 0
            self.logger.add_tabular_data(tb_prefix='train', **training_log)
            self.logger.add_tabular_data(tb_prefix='train', reset_num_sum=reset_num_sum)
            self.grad_num += 1
            # self.logger.add_tabular_data(tb_prefix='performance', **test_result)
            self.logger.log_tabular('iteration', iter_train, tb_prefix='timestep')
            self.logger.log_tabular('timestep', self.sample_num, tb_prefix='timestep')
            self.logger.log_tabular('grad_num', self.grad_num, tb_prefix='timestep')
            self.logger.log_tabular('time', time.time() - self.start_time, tb_prefix='timestep')
            self.logger.log_tabular('memory_size', self.replay_buffer.size, tb_prefix='buffer')
            self.logger.log_tabular('memory_trajectory_num', len(self.replay_buffer), tb_prefix='buffer')
            self.logger.log_tabular('minimal_target_std', np.exp(minimal_target_logstd), tb_prefix='sample')
            self.logger.log_tabular('maximal_target_std', np.exp(maximal_target_logstd), tb_prefix='sample')
            self.logger.log_tabular('offpolicy_memory_size', self.offpolicy_replay_buffer.size, tb_prefix='buffer')
            self.logger.log_tabular('offpolicy_memory_trajectory_num', len(self.offpolicy_replay_buffer), tb_prefix='buffer')
            if self.parameter.image_input:
                self.logger.log_tabular('image_offpolicy_cnt', self.image_offpolicy_cnt, tb_prefix='buffer')
            self.logger.add_tabular_data(tb_prefix='timer', **self.timer.summary(summation=True))


            self.logger.dump_tabular()
            self.logger(f'logdir: {self.logger.output_dir}')
            if iter_train % 10 == 0:
                self.save()
            if iter_train % 50 == 0 and self.parameter.backing_log:
                self.logger.sync_log_to_remote(replace=iter_train == 0, trial_num=5)
            if self.actor_lr_scheduler is not None:
                self.actor_lr_scheduler.step()
            if self.critic_lr_scheduler is not None:
                self.critic_lr_scheduler.step()
            if self.target_entropy_scheduler is not None:
                self.target_entropy_scheduler.step()
                self.target_entropy = self.target_entropy_scheduler.get_value()
            self.replay_buffer.reset()
            self.dog_cnt -= 1
        # self.logger.sync_log_to_remote(replace=False, trial_num=5)

    def save(self, model_dir=None):
        model_path = os.path.join(self.logger.output_dir, 'model') if model_dir is None else model_dir
        self.policy.save(model_path)
        for i in range(len(self.values)):
            self.values[i].save(model_path, index=f'{i}')
            self.target_values[i].save(model_path, index=f'{i}-target')
        torch.save(self.log_sac_alpha, os.path.join(model_path, 'log_sac_alpha.pt'))

    def load(self, model_dir=None, load_policy=True, load_value=True):
        model_path = os.path.join(self.logger.output_dir, 'model') if model_dir is None else model_dir
        if load_policy:
            self.policy.load(model_path, map_location=self.sample_device)
        if load_value:
            for i in range(len(self.values)):
                self.values[i].load(model_path, index=f'{i}', map_location=self.device)
                self.target_values[i].load(model_path, index=f'{i}-target', map_location=self.device)
        # self.policy.reset_target_encoder()
        self.log_sac_alpha = torch.load(os.path.join(model_path, 'log_sac_alpha.pt'), map_location=self.device)


