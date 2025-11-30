import datetime

import parser

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer
import argparse
from argparse import Namespace
import smart_logger
from offpolicy_rnn import init_smart_logger, Parameter, alg_init
from concurrent.futures import ThreadPoolExecutor
from offpolicy_rnn.env_utils.make_env import make_env
from legged_gym.utils.webviewer_local_save_paper_visual_uncertainty import WebViewer

from low_control_pkg.LowerPolicy import LowerPolicy
from low_control_pkg.LowerPolicyPitch import LowerPitchPolicy
from concurrent.futures import ProcessPoolExecutor
import random
import multiprocessing
import math
import pickle


task_name_to_log_name = {
    'four_corners': [
        'ParkourGo-v0_ppo_seperate_optim_seed_1-ParkourGo_ppo_0426_S_6',
        'ParkourGo-v0_ppo_seperate_optim_seed_2-ParkourGo_ppo_0426_S_6',
        'ParkourGo-v0_ppo_seperate_optim_seed_3-ParkourGo_ppo_0426_S_6',
        'ParkourGo-v0_ppo_seperate_optim_seed_4-ParkourGo_ppo_0426_S_6',
        'ParkourGo-v0_ppo_seperate_optim_seed_5-ParkourGo_ppo_0426_S_6',
    ],
    'left_right_choose': [
        'ParkourGo-v0_ppo_seperate_optim_seed_1-ParkourGo_ppo_0421_ver',
        'ParkourGo-v0_ppo_seperate_optim_seed_2-ParkourGo_ppo_0421_ver',
        'ParkourGo-v0_ppo_seperate_optim_seed_3-ParkourGo_ppo_0421_ver',
        'ParkourGo-v0_ppo_seperate_optim_seed_4-ParkourGo_ppo_0421_ver',
        'ParkourGo-v0_ppo_seperate_optim_seed_5-ParkourGo_ppo_0421_ver',
    ],
    'circle': [
        'ParkourGo-v0_ppo_seperate_optim_seed_1-ParkourGo_ppo_0421_plat_scandot',
        'ParkourGo-v0_ppo_seperate_optim_seed_2-ParkourGo_ppo_0421_plat_scandot',
        'ParkourGo-v0_ppo_seperate_optim_seed_3-ParkourGo_ppo_0421_plat_scandot',
        'ParkourGo-v0_ppo_seperate_optim_seed_4-ParkourGo_ppo_0421_plat_scandot',
        'ParkourGo-v0_ppo_seperate_optim_seed_5-ParkourGo_ppo_0421_plat_scandot',
    ],
    'middle_choose': [
        'ParkourGo-v0_ppo_seperate_optim_seed_1-ParkourGo_ppo_0427_ch',
        'ParkourGo-v0_ppo_seperate_optim_seed_2-ParkourGo_ppo_0427_ch',
        'ParkourGo-v0_ppo_seperate_optim_seed_3-ParkourGo_ppo_0427_ch',
        'ParkourGo-v0_ppo_seperate_optim_seed_4-ParkourGo_ppo_0427_ch',
        'ParkourGo-v0_ppo_seperate_optim_seed_5-ParkourGo_ppo_0427_ch',
    ],
    'stair_find_image': [
        'ParkourGo-v0_ppo_seperate_optim_seed_1-ParkourGo_ppo_0413_corner_2',
        # 'ParkourGo-v0_stair_find_image_ppo_seperate_optim_seed_1-ParkourGo_ppo_0506',
    ],
    'circle_image': [
        'ParkourGo-v0_ppo_seperate_optim_seed_1-ParkourGo_ppo_0421_plat',
    ],
    'middle_choose_image': [
        'ParkourGo-v0_ppo_seperate_optim_seed_1-ParkourGo_ppo_0415_ch_img'
    ]
}

RMA_task_name_to_log_name = {
    'four_corners': [
        'ParkourGo-v0_four_corners_ppo_seperate_optim_seed_1-ParkourGo_ppo_0428_RMA',
        'ParkourGo-v0_four_corners_ppo_seperate_optim_seed_2-ParkourGo_ppo_0428_RMA',
        'ParkourGo-v0_four_corners_ppo_seperate_optim_seed_3-ParkourGo_ppo_0428_RMA',
    ],
    'left_right_choose': [
        'ParkourGo-v0_left_right_choose_ppo_seperate_optim_seed_1-ParkourGo_ppo_0428_RMA',
        'ParkourGo-v0_left_right_choose_ppo_seperate_optim_seed_2-ParkourGo_ppo_0428_RMA',
        'ParkourGo-v0_left_right_choose_ppo_seperate_optim_seed_3-ParkourGo_ppo_0428_RMA',
    ],
    'circle': [
        'ParkourGo-v0_circle_ppo_seperate_optim_seed_1-ParkourGo_ppo_0428_RMA',
        'ParkourGo-v0_circle_ppo_seperate_optim_seed_2-ParkourGo_ppo_0428_RMA',
        'ParkourGo-v0_circle_ppo_seperate_optim_seed_3-ParkourGo_ppo_0428_RMA',
    ],
    'middle_choose': [
        'ParkourGo-v0_middle_choose_ppo_seperate_optim_seed_1-ParkourGo_ppo_0428_RMA',
        'ParkourGo-v0_middle_choose_ppo_seperate_optim_seed_2-ParkourGo_ppo_0428_RMA',
        'ParkourGo-v0_middle_choose_ppo_seperate_optim_seed_3-ParkourGo_ppo_0428_RMA',
    ],
    'stair_find_image': [
        'ParkourGo-v0_stair_find_image_ppo_seperate_optim_seed_1-ParkourGo_ppo_0428_RMA',
    ],
    'circle_image': [
        'ParkourGo-v0_ppo_seperate_optim_seed_1-ParkourGo_ppo_0426_RMA_plat_img',
    ],
    'middle_choose_image': [
        'ParkourGo-v0_middle_choose_image_ppo_seperate_optim_seed_1-ParkourGo_ppo_0428_RMA'
    ]

}

test_candidates = [
    # 'circle_image', # 'stair_find_image'
    # 'four_corners',
    # 'left_right_choose',
    # 'circle',
    # 'middle_choose',
    'stair_find_image',  # 'circle_image',
    # 'middle_choose_image',
    # 'stair_find_image',
    # 'circle_image',
]

test_alg = [
    'ours',
    # 'rma'
]

tasks = [(item1, item2) for item1 in test_alg for item2 in test_candidates]


target_mode = True
sample_std = 0.0

include_depth = True
max_difficulty = False

web = True
write_video = True
test_all_seeds = False
path_header = '/home/luofm'

# video_base_dir = '/home/ubuntu/luofanming/parkour_video' if path_header == '/home/ubuntu' else '/home/luofm/Downloads/parkour_video'
video_base_dir = '/mnt/parkour/parkour_video'
def system(cmd):
    print(cmd)
    os.system(cmd)
def prepare_model(log_name):
    log_full_path = os.path.join(smart_logger.get_base_path(), 'logfile', log_name)
    # if not os.path.exists(log_full_path):
    if os.path.exists(f'/mnt/parkour/Code/ParkourGo/logfile/{log_name}'):
        system(f'rsync -avz --progress /mnt/parkour/Code/ParkourGo/logfile/{log_name} {os.path.dirname(log_full_path)}/')
    else:
        system(f'rsync -avz --progress -e "ssh -p 31799" --exclude "*.mp4" --exclude "*.pkl" ubuntu@ServerIp:luofanming/Code/ParkourGo/logfile/{log_name} {os.path.dirname(log_full_path)}/')

def get_parameter(log_name):
    log_full_path = os.path.join(smart_logger.get_base_path(), 'logfile', log_name)
    config_dir = os.path.join(log_full_path, 'config')
    parameter = Parameter(config_path=config_dir)
    return parameter

def _seed(seed: int):
    np.random.seed(seed)
    random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_policy(log_name, env):
    log_full_path = os.path.join(smart_logger.get_base_path(), 'logfile', log_name)
    model_dir = os.path.join(log_full_path, 'model')
    config_dir = os.path.join(log_full_path, 'config')
    parameter = Parameter(config_path=config_dir)
    parameter.debug = True
    ppo = alg_init(parameter, env)
    ppo.load(model_dir=model_dir, load_value=False)
    policy = ppo.policy
    policy.eval()
    policy.target_mode = target_mode
    if parameter.directly_train_target:
        policy.no_logstd_output = True
    return policy, ppo.logger


def play(log_name, task_name, alg_name):
    prepare_model(log_name)
    parameter = get_parameter(log_name)
    _seed(parameter.seed)
    parameter.resume_log_name = None
    parameter.task_name = task_name
    parameter.information = f'{"target" if target_mode else "train"}_{alg_name}'
    print(parameter)
    faulthandler.enable()
    video_dir = os.path.join(video_base_dir, f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}-{parameter.short_name}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    remote_env_additional_configs = {
        'only_positive_rewards': False,
        'tracking_goal_vel_factor': 200.0,
        'curriculum': False,
        'depth': parameter.image_input,
        'num_envs': 5 if write_video or not web else 5,
        'no_privileged_info': parameter.no_privileged_info,
        'no_ext_privileged_info': parameter.no_ext_privileged_info,
        'headless': True,
        'camera_num_envs': 5 if write_video or not web else 5,
        # 'device_type': device.type,
        # 'device_index': device.index,
        'camera_setting_but_no_camera': False,
        'no_simplify': True,
        'action_delay': parameter.action_delay,
        'extreme_little_info': parameter.extreme_little_info,
        'include_absolute_position': parameter.include_absolute_position,
        'include_scan_dot': parameter.square_task,
        'plat_task': parameter.plat_task,
        'include_yaw': parameter.include_yaw,
        'save_dir': video_dir,
        'higher_mode': parameter.higher_mode,
        'higher_pitch': parameter.higher_pitch if hasattr(parameter, 'higher_pitch') else False,
        'env_num_rows': 7 if write_video or not web else 7,
        'env_num_cols': 7 if write_video or not web else 7,
        'max_difficulty': max_difficulty,
        # 'force_no_simplify': False,
        # 'terran_height_manual': [0.02, 0.06]
    }

    env_info = make_env(parameter.env_name, parameter.seed, env=None, **remote_env_additional_configs)
    for k, v in env_info.items():
        print(f'{k}: {v}')
    env = env_info['train_env']
    env_cfg, _ = env_info['additional_configs']
    if parameter.higher_mode:
        if hasattr(parameter, 'higher_pitch') and parameter.higher_pitch:
            lower_policy = LowerPitchPolicy()
            lower_policy.to_device(env.device)
        else:
            lower_policy = LowerPolicy()
            lower_policy.to_device(env.device)
    if web:
        if write_video:
            view_mode = 'square'
            if task_name == 'circle' or task_name == 'circle_image':
                view_mode = 'circle'
            elif task_name == 'middle_choose':
                view_mode = 'middle_choose'
            elif task_name == 'four_corners':
                view_mode = 'four_corners'
            elif task_name == 'left_right_choose':
                view_mode = 'left_right_choose'

            web_viewer = WebViewer(output_dir=video_dir, maximum_camera_num=20,
                                   view_mode=view_mode, include_depth=include_depth)
            faulthandler.enable()
            web_viewer.resize_ratio = 1.0
            web_viewer.setup(env)
        else:
            web_viewer = webviewer.WebViewer()
            faulthandler.enable()
            web_viewer.setup(env)
    env.reset()
    obs = env.get_processed_observation()
    obs_priv_np = env.get_processed_privileged_observation()
    policy, logger = get_policy(log_name, env)
    if sample_std > 0:
        policy.log_std = np.log(sample_std)
    device = policy.device
    last_action_np = torch.zeros((env.num_envs, (2 if hasattr(parameter, 'higher_pitch') and parameter.higher_pitch else 1) if parameter.higher_mode else 12), device=device)
    last_state_np = torch.zeros((env.num_envs, obs.shape[-1]), device=device)
    last_priv_state_np = torch.zeros((env.num_envs, obs_priv_np.shape[-1]), device=device)
    reward_np = torch.zeros((env.num_envs, 1), device=device)
    rnn_hidden = policy.make_init_state(env.num_envs, device=device)
    target_encoder_hidden = policy.target_encoder.make_init_state(env.num_envs, device=device)
    batch_last_smoothed_std = None
    ep_return = 0
    goal_reaching_each_env = {}
    ret_each_env = {}
    for i in range(int(env.max_episode_length) if write_video else 6 * int(env.max_episode_length)):
        with torch.no_grad():
            obs_np = env.get_processed_observation().to(device).clone()
            obs_priv_np = env.get_processed_privileged_observation().to(device).clone()
            target_embedding, target_logstd, target_encoder_hidden = policy.get_target_embedding(
                obs_np.unsqueeze(1).to(device), last_action_np.unsqueeze(1).to(device), target_encoder_hidden, )
            if batch_last_smoothed_std is None:
                batch_last_smoothed_std = torch.zeros_like(target_logstd) + 0.5
            batch_last_smoothed_std = batch_last_smoothed_std * 0.95 + 0.05 * target_logstd.exp()
            target_logstd = torch.log(batch_last_smoothed_std)
            MIN_LOGSTD, MAX_LOGSTD = -10, 1
            target_logstd_norm = (torch.clamp(target_logstd, min=MIN_LOGSTD, max=MAX_LOGSTD) - MIN_LOGSTD) / (
                    MAX_LOGSTD - MIN_LOGSTD) * 2 - 1
            if target_mode:
                privileged_embedding, _ = policy.get_privileged_embedding(obs_priv_np.unsqueeze(1).to(device), None)
                # privileged_embedding, _ = policy.privileged_encoder.forward(obs_priv_np.unsqueeze(1).to(device), None)
                act_mean, embedding_output, act_sample, _, rnn_hidden, _ = policy.forward(
                    state=obs_np.unsqueeze(1).to(device),
                    lst_state=last_state_np.unsqueeze(1).to(device),
                    lst_action=last_action_np.unsqueeze(1).to(device),
                    rnn_memory=rnn_hidden,
                    reward=reward_np.unsqueeze(1).to(device),
                    target_logstd=target_logstd,
                )

                ebd_diff = (privileged_embedding - target_embedding).pow(2).mean(dim=-1).sqrt()
            else:
                if parameter.mean_target_input:
                    target_logstd_norm = torch.cat((target_logstd_norm, target_embedding), dim=-1)
                act_mean, privileged_embedding, act_sample, _, rnn_hidden, _ = policy.forward(
                    state=obs_priv_np.unsqueeze(1).to(device),
                    lst_state=last_priv_state_np.unsqueeze(1).to(device),
                    lst_action=last_action_np.unsqueeze(1).to(device),
                    rnn_memory=rnn_hidden,
                    reward=reward_np.unsqueeze(1).to(device),
                    target_logstd=target_logstd_norm,
                )
                ebd_diff = (privileged_embedding - target_embedding).pow(2).mean(dim=-1).sqrt()
            act_mean, act_sample = map(lambda x: x[:, 0], [
                act_mean, act_sample
            ])
        if sample_std == 0:
            actions = act_mean
        else:
            actions = act_sample
        if parameter.higher_mode:
            if hasattr(parameter, 'higher_pitch') and parameter.higher_pitch:

                action_lower = lower_policy.forward(env,
                                                    actions[..., 0:1].clamp(-1, 1) * 3.14 / 2 + env.yaw.clone().unsqueeze(1),
                                                    # (act[..., 1:2].clamp(-1, 1) + 1) / 2 * 0.9 + 0.1,
                                                    actions[..., :1] * 0.0 + 0.5,
                                                    torch.clamp(actions[..., 1:2].clamp(-1, 1) * np.deg2rad(45.), -np.deg2rad(30.), np.deg2rad(30)))

            else:
                action_lower = lower_policy.forward(env, actions.clamp(-1, 1) * 3.14 + env.yaw.clone().unsqueeze(1),
                                                    torch.ones((env.num_envs, 1), device=env.device) * 0.7)
            # actions = torch.randn_like(actions)
            obs, _, rews, dones, infos = env.step(action_lower.detach().to(env.device))
        else:
            obs, _, rews, dones, infos = env.step(actions.detach().to(env.device))

        ep_return = ep_return + rews

        last_state_np = obs_np * (1 - dones.to(torch.float).unsqueeze(1))
        last_priv_state_np = obs_priv_np * (1 - dones.to(torch.float).unsqueeze(1))
        last_action_np = actions * (1 - dones.to(torch.float).unsqueeze(1))
        reward_np = (rews * (1 - dones.to(torch.float))).unsqueeze(1)
        rnn_hidden.hidden_state_mask_reset_(dones)
        target_encoder_hidden.hidden_state_mask_reset_(dones)
        if web:
            if write_video:
                web_viewer.render_all(torch.ones_like(dones) if i == int(env.max_episode_length) - 1 else torch.zeros_like(dones),
                                      fetch_results=True,
                                      step_graphics=True,
                                      render_all_camera_sensors=True,
                                      wait_for_page_load=True,
                                      ebd_diff=ebd_diff,
                                      ep_return=ep_return + rews,
                                      target_ebd=target_embedding,
                                      privileged_ebd=privileged_embedding,
                                      target_std=target_logstd.exp().mean(dim=-1),
                                      target_logstd=target_logstd
                                      )
            else:
                web_viewer.render(fetch_results=True,
                                  step_graphics=True,
                                  render_all_camera_sensors=True,
                                  wait_for_page_load=True)
        if torch.any(dones):
            logger(f'[{log_name}] average episode return: {(ep_return * dones).sum() / dones.sum()}')
            if batch_last_smoothed_std is not None:
                batch_last_smoothed_std[dones] = 0.5
            env_ids = [i for i in dones.nonzero(as_tuple=False).flatten().cpu().detach().numpy()]
            for env_id in env_ids:
                goal_idx = env.lst_goal_idx[env_id].float().item()
                if env_id not in goal_reaching_each_env:
                    goal_reaching_each_env[env_id] = []
                    ret_each_env[env_id] = []
                goal_reaching_each_env[env_id].append(goal_idx)
                ret_each_env[env_id].append(ep_return[env_id].item())
            ep_return = ep_return * (1 - dones.to(torch.float))

        # camera_id = web_viewer._camera_id
        # id = env.lookat_id
    logger(f'video save to {video_dir}')
    logger(f'summary:')

    logger(f'[{log_name}] goal_reaching_each_env: {goal_reaching_each_env}')
    logger(f'[{log_name}] ret_each_env: {ret_each_env}')
    total_traj_num = {k: len(v) for k, v in goal_reaching_each_env.items()}
    aver_goal_reaching_each_env = {k: sum(v) / len(v) for k, v in goal_reaching_each_env.items()}
    aver_ret_each_env = {k: sum(v) / len(v) for k, v in ret_each_env.items()}
    logger(f'[{log_name}] aver_goal_reaching_each_env: {aver_goal_reaching_each_env}')
    logger(f'[{log_name}] aver_ret_each_env: {aver_ret_each_env}')
    logger(f'[{log_name}] total_traj_num: {total_traj_num}')
    final_aver_goal = sum([v for k, v in aver_goal_reaching_each_env.items()]) / len(aver_goal_reaching_each_env)
    final_aver_ret = sum([v for k, v in aver_ret_each_env.items()]) / len(aver_ret_each_env)
    traj_sum = sum([v for k, v in total_traj_num.items()])
    logger(f'[{log_name}] final_aver_goal: {final_aver_goal}')
    logger(f'[{log_name}] final_aver_ret: {final_aver_ret}')
    logger(f'[{log_name}] traj_sum: {traj_sum}')
    summary = {
        'goal_reaching_each_env': goal_reaching_each_env,
        'ret_each_env': ret_each_env,
        'total_traj_num': total_traj_num,
        'aver_goal_reaching_each_env': aver_goal_reaching_each_env,
        'aver_ret_each_env': aver_ret_each_env,
        'final_aver_goal': final_aver_goal,
        'final_aver_ret': final_aver_ret,
        'traj_sum': traj_sum,
        'meta_data': (log_name, task_name, alg_name),
    }
    return summary

def evaluate_one_model(args):
    init_smart_logger(
    )
    model_log_name, task_name, alg_name = args
    return play(model_log_name, task_name, alg_name)

def parallel_evaluate_model(tasks):
    if not multiprocessing.get_start_method(allow_none=True) == 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    size_num = 4
    batch_num = int(math.ceil(len(tasks) / size_num))
    total_results = []
    if test_all_seeds:
        all_log_name = []
        for item in tasks:
            for ssss in get_log_name(item):
                all_log_name.append((ssss, item[1], item[0]))
        batch_num = int(math.ceil(len(all_log_name) / size_num))

        for i in range(batch_num):
            model_sub_names = [item for item in all_log_name[i * size_num:(i + 1) * size_num]]
            with ProcessPoolExecutor(max_workers=len(model_sub_names)) as executor:
                results = executor.map(evaluate_one_model, model_sub_names)
                results = list(results)
                total_results += results
        with open(f'all_seed_results.pkl', 'wb') as f:
            pickle.dump(total_results, f)
    else:

        for i in range(batch_num):
            model_sub_names = [(get_log_name(item), item[1], item[0]) for item in tasks[i * size_num:(i + 1) * size_num]]
            with ProcessPoolExecutor(max_workers=len(model_sub_names)) as executor:
                results = executor.map(evaluate_one_model, model_sub_names)
                results = list(results)

def get_log_name(task):
    alg_name, task_name = task
    if not test_all_seeds:
        if alg_name == 'ours':
            log_name = task_name_to_log_name[task_name][0]
        elif alg_name == 'rma':
            log_name = RMA_task_name_to_log_name[task_name][0]
        else:
            raise ValueError(f'unknown alg_name: {alg_name}')
        return log_name
    else:
        if alg_name == 'ours':
            log_name = task_name_to_log_name[task_name]
        elif alg_name == 'rma':
            log_name = RMA_task_name_to_log_name[task_name]
        else:
            raise ValueError(f'unknown alg_name: {alg_name}')
        return log_name


def get_summary():
    init_smart_logger()
    base_path_dir = smart_logger.get_base_path()
    for task in tasks:
        log_name = get_log_name(task)
        log_path = os.path.join(base_path_dir, 'logfile', f'{log_name}-debug', 'log.txt')
        print(f'[{log_name}]: {log_path}')
        os.system(f'tail -n 3 {log_path}')
def list_tasks():
    for idx, task in enumerate(tasks):
        log_name = get_log_name(task)
        print(f'task_id:{idx}, {log_name}: {task}')
if __name__ == '__main__':
    # evaluate_one_model(model_log_name=log_name)

    list_tasks()
    parallel_evaluate_model(tasks=tasks)
    #
    # get_summary()
    # list_tasks()