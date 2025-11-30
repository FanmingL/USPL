import copy

import yaml
import os
import math
from smart_logger.scripts.generate_tmuxp_base import generate_tmuxp_file, make_cmd_array
import argparse
import subprocess
from typing import Dict
MAX_SUBWINDOW = 2
# [MAX_PARALLEL]: The maximum number of experiments that can run in parallel. If you need to run 3 seeds in 5 environments, you will
# need to set up a total of 3x5=15 experiments. If you set `MAX_PARALLEL` to 4, it will start 4 task queues,
# each executing a roughly equal number of tasks sequentially. In this case, the four queues will execute tasks
# in the following quantities: [4, 4, 4, 3].
MAX_PARALLEL = 12

def get_gpu_count():
    sp = subprocess.Popen(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_list = out_str[0].decode('utf-8').split('\n')
    out_list = [x for x in out_list if x]
    return len(out_list)

def get_cmd_array(total_machine=8, machine_idx=0):
    """
    :return: cmd array: list[list[]], the item in the i-th row, j-th column of the return value denotes the
                        cmd in the i-th window and j-th sub-window
    """
    session_name = 'OffpolicyRNN'
    # 0. 代码运行路径
    current_path = os.path.dirname(os.path.abspath(__file__))
    # 1. GPU设置
    GPUS = [','.join([str(i) for i in range(get_gpu_count())])]
    # 2. 环境变量设置
    environment_dict = dict(
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=current_path,
        OMP_NUM_THREADS=1,
        TF_ENABLE_ONEDNN_OPTS=0,
        TF_CPP_MIN_LOG_LEVEL=2,
    )
    directory = current_path
    # 3. 启动脚本
    start_up_header = "/home/ubuntu/.conda/envs/py38/bin/python main.py"
    # 4. 基础参数
    basic_activation = 'elu'
    embedding_output_activation = f'elu'
    rnn_type_name = 'fc'
    rnn_type_name_value = 'fc'
    common_ndim = 256
    parameters_base = dict(
        alg_name='ppo_seperate_optim',
        base_algorithm='ppo',
        total_iteration=10000,
        step_per_iteration=1000,
        target_entropy_ratio=2.5,
        test_nprocess=5,
        test_nrollout=1,

        value_embedding_layer_type=[f'fc', f'{rnn_type_name_value}', f'fc'],
        value_embedding_activations=[f'{basic_activation}', f'{basic_activation}', embedding_output_activation],
        value_embedding_hidden_size=[common_ndim, common_ndim],

        value_hidden_size=[common_ndim, common_ndim],
        value_activations=[basic_activation, basic_activation, 'linear'],
        value_layer_type=[f'fc', f'fc', f'fc'],

        policy_embedding_layer_type=[ f'fc', f'{rnn_type_name}', 'fc'],
        policy_embedding_activations=[f'{basic_activation}', f'{basic_activation}',
                                      embedding_output_activation],
        policy_embedding_hidden_size=[common_ndim, common_ndim],

        policy_hidden_size=[common_ndim, common_ndim],
        policy_activations=[basic_activation, basic_activation, 'linear'],
        policy_layer_type=[f'fc', 'fc', f'fc'],
        sac_tau=0.995,  # influence value loss divergence
        value_net_num=1,
        cuda_inference=True,
        random_num=5000,
        max_buffer_traj_num=5000,
        policy_embedding_dim=128,
        value_embedding_dim=128,
        alpha_lr=1e-3,
        policy_uni_model_input_mapping_dim=128,
        value_uni_model_input_mapping_dim=128,
        policy_update_per=2,
        sac_batch_size=999,
        state_action_encoder=True,
        entropy_coeff=1e-3,
        policy_embedding_output_activation='tanh',
        target_encoder_rnn_type='cgpt_h16_l1_p0.2_ml1024_rms',
        target_encoder_lr=5e-5,
        target_encoder_l2_norm=0.0,
        target_encoder_learn_std=True,
        last_state_input=False,
        embedding_noise=True,
        no_privileged_info=True,
        include_absolute_position=True,
        include_yaw=True,
    )
    # 5. 遍历设置
    exclusive_candidates = dict(
        seed=[1],
        env_name=[
            'ParkourGo-v0',
                    ],
    )
    # 6. 单独设置
    aligned_candidates = dict(
        task_name=['left_right_choose'], # left_right_choose, four_corners, middle_choose, circle, middle_choose_image, circle_image, stair_find_image
        mean_target_input=[False],
        gamma=[0.998],
        higher_mode=[True],
        no_noise_perturbation=[True],
        no_reward_modification=[True],
        target_logstd_input=[False],
        encoder_bc_policy_rl=[False],
        ground_truth_mode=[False],
        directly_train_target=[True],

        information=['direct_PPO'],
    )

    def task_is_valid(_task):
        if _task['task_name'] == 'middle_choose':
            _task['image_input'] = False
            _task['square_task'] = True
            _task['plat_task'] = False
            _task['higher_pitch'] = False
            _task['higher_mode'] = True
        elif _task['task_name'] == 'middle_choose_image':
            _task['image_input'] = True
            _task['square_task'] = True
            _task['plat_task'] = False
            _task['higher_pitch'] = False
            _task['higher_mode'] = True
        elif _task['task_name'] == 'circle':
            _task['image_input'] = False
            _task['square_task'] = True
            _task['plat_task'] = True
            _task['higher_pitch'] = False
            _task['higher_mode'] = True
        elif _task['task_name'] == 'circle_image':
            _task['image_input'] = True
            _task['square_task'] = True
            _task['plat_task'] = True
            _task['higher_pitch'] = False
            _task['higher_mode'] = True
        elif _task['task_name'] == 'left_right_choose':
            _task['image_input'] = False
            _task['square_task'] = False
            _task['plat_task'] = False
            _task['higher_pitch'] = False
            _task['higher_mode'] = True
        elif _task['task_name'] == 'four_corners':
            _task['image_input'] = False
            _task['square_task'] = False
            _task['plat_task'] = True
            _task['higher_pitch'] = False
            _task['higher_mode'] = True
            _task['sample_std'] = 0.5
        elif _task['task_name'] == 'stair_find_image':
            _task['image_input'] = True
            _task['square_task'] = False
            _task['plat_task'] = True
            _task['higher_pitch'] = True
            _task['higher_mode'] = True
            _task['sample_std'] = 0.5
        return True

    # 从这里开始不用再修改了
    cmd_array, session_name = make_cmd_array(
        directory, session_name, start_up_header, parameters_base, environment_dict,
        aligned_candidates, exclusive_candidates, GPUS, MAX_PARALLEL, MAX_SUBWINDOW,
        machine_idx, total_machine, task_is_valid, split_all=True, sleep_before=0.0, sleep_after=0.0, rnd_seed=None, task_time_interval=60,
    )
    # 上面不用修改了

    # 7. 额外命令
    cmd_array.append(['htop', 'watch -n 1 nvidia-smi'])
    cmd_array.append(['python orphans_kill.py'])
    return cmd_array, session_name


def main():
    parser = argparse.ArgumentParser(description=f'generate parallel environment')
    parser.add_argument('--machine_idx', '-idx', type=int, default=-1,
                        help="Server port")
    parser.add_argument('--total_machine_num', '-tn', type=int, default=8,
                        help="Server port")
    args = parser.parse_args()
    cmd_array, session_name = get_cmd_array(args.total_machine_num, args.machine_idx)
    generate_tmuxp_file(session_name, cmd_array, use_json=True, layout='even-horizontal')


if __name__ == '__main__':
    main()
