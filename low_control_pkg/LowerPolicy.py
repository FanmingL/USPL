from .contextual_ppo_policy_mlp_encoder import ContextualPPOPolicy
import os
import torch


class LowerPolicy:
    def __init__(self, ):
        self.policy_args = {'state_dim': 172,
                            'action_dim': 12,
                            'priv_state_dim': 204,
                            'embedding_size': 128,
                            'embedding_hidden': [256, 256],
                            'embedding_activations': ['elu', 'elu', 'elu'],
                            'embedding_layer_type': ['fc', 'fc', 'fc'],
                            'uni_model_hidden': [256, 256],
                            'uni_model_activations': ['elu', 'elu', 'linear'],
                            'uni_model_layer_type': ['fc', 'fc', 'fc'],
                            'fix_rnn_length': 0,
                            'reward_input': False,
                            'last_action_input': True,
                            'last_state_input': False,
                            'uni_model_input_mapping_dim': 128,
                            'separate_encoder': True,
                            'use_camera': False,
                            'sample_std': 0.2,
                            'std_learnable': False,
                            'embedding_output_activation': 'tanh',
                            'target_rnn_type': 'fc',
                            'target_network_learn_std': True,
                            'target_logstd_input': False}
        self.action_dim = self.policy_args['action_dim']
        self.last_action_input = self.policy_args['last_action_input']
        self.last_state_input = self.policy_args['last_state_input']
        self.reward_input = self.policy_args['reward_input']

        self.base_algorithm = 'ppo'
        self.discrete_env = False
        self.policy = ContextualPPOPolicy(**self.policy_args)
        self.device = torch.device('cpu')
        self.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pt'), map_location=torch.device('cpu'))
        self.last_action_input = None
        self.last_state_input = None
        self.last_reward_input = None
        self.rnn_hidden = None
        self.reset()

    def to_device(self, device):
        if not self.device == device:
            self.device = device
            self.policy.to(self.device)
            if self.last_action_input is not None:
                self.last_action_input = self.last_action_input.to(self.device)
            if self.last_state_input is not None:
                self.last_state_input = self.last_state_input.to(self.device)
            if self.last_reward_input is not None:
                self.last_reward_input = self.last_reward_input.to(self.device)

    def load(self, dirname, index=0, **kwargs):
        for k, v in self.policy.contextual_modules.items():
            if k == 'target_encoder_subnet':
                continue
            name = f'{self.policy.name}-{index}-{k}.pt'
            full_path = os.path.join(dirname, name)
            if hasattr(v, 'load'):
                v.load(full_path, **kwargs)
            else:
                # print(f'loading from {full_path}..')
                item = torch.load(full_path, **kwargs)
                v.load_state_dict(item)

    @torch.no_grad()
    def reset(self, reset_buf=None):
        if reset_buf is None:
            self.last_action_input = None
            self.last_state_input = None
            self.last_reward_input = None
            self.rnn_hidden = None
        else:
            if self.last_action_input is not None:
                self.last_action_input = (1 - reset_buf.float().unsqueeze(1)) * self.last_action_input
            if self.last_state_input is not None:
                self.last_state_input = (1 - reset_buf.float().unsqueeze(1)) * self.last_state_input
            if self.last_reward_input is not None:
                self.last_reward_input = (1 - reset_buf.float().unsqueeze(1)) * self.last_reward_input
            if self.rnn_hidden is not None:
                self.rnn_hidden.hidden_state_mask_reset_(reset_buf)

    def get_obs(self, env, target_yaw_rad: torch.Tensor, forward_cmd: torch.Tensor) -> torch.Tensor:
        assert len(target_yaw_rad.shape) == 2
        assert target_yaw_rad.shape[-1] == 1, f'expected yaw shape [*, 1], got {target_yaw_rad}'
        assert len(forward_cmd.shape) == 2
        assert forward_cmd.shape[-1] == 1, f'expected forward shape [*, 1], got {forward_cmd}'
        ego_rotation = env.base_ang_vel * env.obs_scales.ang_vel
        imu_obs = torch.stack((env.roll, env.pitch), dim=1)
        ego_delta_yaw = target_yaw_rad - env.yaw.unsqueeze(1)
        cos_delta_yaw = torch.cos(ego_delta_yaw)
        sin_delta_yaw = torch.sin(ego_delta_yaw)
        zeros = torch.zeros((ego_delta_yaw.shape[0], 1), device=target_yaw_rad.device)
        zeros2 = torch.zeros((ego_delta_yaw.shape[0], 2), device=target_yaw_rad.device)
        zeros4 = torch.zeros((ego_delta_yaw.shape[0], 4), device=target_yaw_rad.device)
        cos_delta_next_yaw = zeros
        sin_delta_next_yaw = zeros
        ego_forward_vel_cmd = forward_cmd
        ego_hurdle_flag = zeros2
        ego_dof_pos = env.reindex((env.dof_pos - env.default_dof_pos_all) * env.obs_scales.dof_pos)
        ego_dof_vel = env.reindex(env.dof_vel * env.obs_scales.dof_vel)
        ego_feet_contact = zeros4
        moving_speed = env.base_lin_vel * env.obs_scales.lin_vel
        priv_latent = torch.cat((
            env.mass_params_tensor,  # 质量参数 # [0, 4]
            env.friction_coeffs_tensor,  # 摩擦力参数 # [4, 5]
            env.motor_strength[0] - 1,  # 电机力参数1，PD控制器的P的放缩系数 # [5, 17]
            env.motor_strength[1] - 1  # 电机力参数2，PD控制器的D的放缩系数 #  # [17, 29]
        ), dim=-1)
        ground_heights = torch.zeros((ego_delta_yaw.shape[0], 132), device=target_yaw_rad.device)
        obs = torch.cat((
            ego_rotation,
            imu_obs,
            cos_delta_yaw,
            sin_delta_yaw,
            cos_delta_next_yaw,
            sin_delta_next_yaw,
            ego_forward_vel_cmd,
            ego_hurdle_flag,
            ego_dof_pos,
            ego_dof_vel,
            ego_feet_contact,
            moving_speed,
            priv_latent,
            ground_heights,
        ), dim=-1)
        return obs

    @torch.no_grad()
    def forward(self, env, target_yaw_rad: torch.Tensor, forward_cmd: torch.Tensor):
        obs = self.get_obs(env, target_yaw_rad, forward_cmd)
        if self.last_action_input is None:
            self.last_state_input = torch.zeros_like(obs)
            self.last_reward_input = torch.zeros((obs.shape[0], 1), device=self.device)
            self.last_action_input = torch.zeros((obs.shape[0], self.action_dim), device=self.device)
        if self.rnn_hidden is None:
            self.rnn_hidden = self.policy.make_init_state(obs.shape[0], self.device)
        act_mean, _, _, _, self.rnn_hidden, _ = self.policy.forward(
            state=obs.unsqueeze(1),
            lst_state=self.last_state_input.unsqueeze(1),
            lst_action=self.last_action_input.unsqueeze(1),
            rnn_memory=self.rnn_hidden,
            reward=self.last_reward_input.unsqueeze(1),
        )

        self.last_state_input = obs.clone()
        self.last_action_input = act_mean.squeeze(1)
        return act_mean.squeeze(1)




