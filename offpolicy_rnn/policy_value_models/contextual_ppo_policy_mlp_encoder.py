import numpy as np
from ..models.contextual_model import ContextualModel
from ..models.mlp_base import MLPBase
import torch
from typing import List, Union, Tuple, Dict, Optional
from ..models.RNNHidden import RNNHidden
from .utils import nearest_power_of_two, nearest_power_of_two_half
from ..models.privileged_obs_encoder import PrivilegedObsEncoder, PrivilegedObsEncoderExpand, PrivilegedObsEncoderNoCommon, PrivilegedObsEncoderSmall
from ..models.target_obs_encoder import (TargetObsEncoder, TargetObsEncoderNoImage, TargetObsEncoderExtreme,
                                         TargetObsEncoderNoImage2Head, TargetObsEncoder2Head,
                                         TargetObsEncoderNoScanDot2Head, TargetObsEncoderNoImage2HeadStdDescend)


class ContextualPPOPolicy(ContextualModel):
    MAX_LOG_STD = 2.0
    MIN_LOG_STD = -20.0
    def __init__(self, state_dim, priv_state_dim, action_dim, embedding_size, embedding_hidden, embedding_activations,
                 embedding_layer_type, uni_model_hidden,
                 uni_model_activations, uni_model_layer_type, fix_rnn_length, uni_model_input_mapping_dim: int=0,
                 reward_input=False, last_action_input=True, last_state_input=False,
                 separate_encoder=False, output_logstd=False, sample_std=0.2,
                 use_camera=False, std_learnable=False, embedding_output_activation='elu', target_rnn_type='smamba_b1_c8_s64_ff',
                 target_network_learn_std=False, target_logstd_input=False, use_scan_dot=False,
                 continuous_vector=True, mean_target_input=False, baseline_mode=False, valid_priv_vector_dim=2, name='ContextualPPOPolicy'):
        # self.scan_ebd_dim = 32
        self.scan_ebd_dim = 2
        self.valid_priv_vector_dim = valid_priv_vector_dim
        self.target_mode = False
        self.no_logstd_output = False
        self.baseline_mode = baseline_mode
        self.continuous_privileged_vector = continuous_vector
        self.target_logstd_input = target_logstd_input
        self.use_scan_dot = use_scan_dot
        self.mean_target_input = mean_target_input
        # self.target_logstd_embedding_dim = self.scan_ebd_dim // 16
        self.target_logstd_embedding_dim = max(self.scan_ebd_dim * 2 // 4, 8)
        self.use_camera = use_camera
        self.original_image_dim_row, self.original_image_dim_col = 58, 87
        self.image_dim = 1 * self.original_image_dim_row * self.original_image_dim_col

        self.scan_origin_dim = self.image_dim if use_camera else 132
        self.target_state_dim = state_dim
        self.privileged_state_dim = priv_state_dim
        state_dim = state_dim - self.scan_origin_dim + self.scan_ebd_dim

        self.std_learnable = std_learnable
        if not uni_model_activations[-1] == 'tanh':
            uni_model_activations = uni_model_activations[:-1] + ['tanh']
        if not uni_model_layer_type[-1] == 'fc':
            raise NotImplementedError(f'It is not supported to construct {uni_model_layer_type[-1]} logstd and mean head! You can set the uni_model layer type to {uni_model_layer_type[:-1] + ["fc"]}')
        self.reward_input = reward_input
        self.last_action_input = last_action_input
        self.last_state_input = last_state_input
        self.reward_dim = 1 if self.reward_input else 0
        self.last_act_dim = action_dim if self.last_action_input else 0
        self.last_obs_dim = state_dim if self.last_state_input else 0
        if self.target_logstd_input:
            state_dim += self.target_logstd_embedding_dim
        if embedding_size == 'auto':
            embedding_size = nearest_power_of_two_half(state_dim)
        if uni_model_input_mapping_dim == 'auto':
            uni_model_input_mapping_dim = nearest_power_of_two(state_dim)
        self.separate_encoder = separate_encoder
        if separate_encoder:
            basic_embedding_dim = 128
            self.state_encoder = torch.nn.Linear(state_dim, basic_embedding_dim)
            self.last_act_encoder = torch.nn.Linear(self.last_act_dim, basic_embedding_dim) if self.last_act_dim else None
            self.reward_encoder = torch.nn.Linear(self.reward_dim, basic_embedding_dim) if self.reward_dim else None
            self.last_obs_encoder = torch.nn.Linear(self.last_obs_dim, basic_embedding_dim) if self.last_obs_dim else None

            cum_dim = basic_embedding_dim
            if self.last_act_encoder is not None:
                cum_dim += basic_embedding_dim
            if self.last_obs_encoder is not None:
                cum_dim += basic_embedding_dim
            if self.reward_encoder is not None:
                cum_dim += basic_embedding_dim
        else:
            cum_dim = state_dim + self.reward_dim + self.last_act_dim + self.last_obs_dim
            self.state_encoder = torch.nn.Identity()
            self.last_act_encoder = torch.nn.Identity()
            self.reward_encoder = torch.nn.Identity()
            self.last_obs_encoder = torch.nn.Identity()

        super(ContextualPPOPolicy, self).__init__(embedding_input_size=cum_dim,
                                                  embedding_size=embedding_size,
                                                  embedding_hidden=embedding_hidden,
                                                  embedding_activations=embedding_activations,
                                                  embedding_layer_type=embedding_layer_type,
                                                  uni_model_input_size=state_dim,
                                                  uni_model_output_size=action_dim * 2 if output_logstd else action_dim,
                                                  uni_model_hidden=uni_model_hidden,
                                                  uni_model_activations=uni_model_activations,
                                                  uni_model_layer_type=uni_model_layer_type,
                                                  fix_rnn_length=fix_rnn_length,
                                                  uni_model_input_mapping_dim=uni_model_input_mapping_dim,
                                                  uni_model_input_mapping_activation=embedding_activations[-1],
                                                  name=name)
        if separate_encoder:
            self.contextual_register_rnn_base_module(self.state_encoder, 'state_encoder')
            if self.last_act_encoder is not None:
                self.contextual_register_rnn_base_module(self.last_act_encoder, 'last_act_encoder')
            if self.last_obs_encoder is not None:
                self.contextual_register_rnn_base_module(self.last_obs_encoder, 'last_obs_encoder')
            if self.reward_encoder is not None:
                self.contextual_register_rnn_base_module(self.reward_encoder, 'reward_encoder')
        # image输入时，target_obs: [common_obs, target_image]
        # 非image输入时，target_obs: [common_obs]
        common_obs_dim = self.target_state_dim - self.image_dim if self.use_camera else self.target_state_dim - 132
        middle_dim = 128
        # 此处privileged obs不考虑scan dot
        scan_dot_dim = 132 if use_scan_dot else 0

        privileged_vector_dim = self.privileged_state_dim - common_obs_dim - scan_dot_dim

        rnn_type = target_rnn_type
        output_activation = embedding_output_activation
        # 5046, 2, 70, 202, 72, 0
        print(f'image dim: {self.image_dim}, '
              f'privileged_vector_dim: {privileged_vector_dim}, '
              f'common_obs_dim: {common_obs_dim}, '
              f'target_state_dim: {self.target_state_dim}, '
              f'privileged_state_dim: {self.privileged_state_dim}, '
              f'privileged_image_dim: {scan_dot_dim}')
        # make_target_encoder(self, image_dim, common_obs_dim, action_dim, ebd_dim, rnn_type, middle_dim, output_activation, target_network_learn_std, use_camera, use_scan_dot):
        self.target_encoder_params = {
            'image_dim': self.image_dim,
            'common_obs_dim': common_obs_dim,
            'action_dim': action_dim,
            'ebd_dim': self.scan_ebd_dim,
            'rnn_type': rnn_type,
            'middle_dim': middle_dim,
            'output_activation': output_activation,
            'target_network_learn_std': target_network_learn_std,
            'use_camera': self.use_camera,
            'use_scan_dot': use_scan_dot,
        }
        self.target_encoder = self.make_target_encoder(**self.target_encoder_params)
        self.privileged_encoder = PrivilegedObsEncoder(scan_dot_dim, privileged_vector_dim, common_obs_dim,
                                                       self.scan_ebd_dim, middle_dim, output_activation, no_common=False, layer_norm=False)
        if std_learnable:
            log_sample_std = torch.zeros((action_dim, )) + np.log(sample_std)
            self.log_std = torch.nn.Embedding(1, action_dim)
            with torch.no_grad():
                self.log_std.weight[:] = log_sample_std
            self.contextual_register_rnn_base_module(self.log_std, 'log_std')
        else:
            self.log_std = np.log(sample_std)

        self.contextual_register_rnn_base_module(self.target_encoder, 'target_encoder_subnet')
        self.contextual_register_rnn_base_module(self.privileged_encoder, 'privileged_encoder')
        if self.target_logstd_input:
            self.target_logstd_encoder = MLPBase(self.scan_ebd_dim * 2 if self.mean_target_input else self.scan_ebd_dim, self.target_logstd_embedding_dim, [128, 64], ['elu', 'elu', output_activation])
            self.contextual_register_rnn_base_module(self.target_logstd_encoder, 'target_logstd_encoder')
        else:
            self.target_logstd_encoder = None
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.soft_plus = torch.nn.Softplus()
        self.lst_processed_state = None

    def make_init_state(self, batch_size: int, device: torch.device, force_target=False) -> RNNHidden:
        target_mode = self.target_mode
        if force_target:
            target_mode = True
        if target_mode:
            target_encoder_hidden = self.target_encoder.make_init_state(batch_size, device)
            policy_hidden = super().make_init_state(batch_size, device)
            return target_encoder_hidden + policy_hidden
        else:
            return super().make_init_state(batch_size, device)

    def make_rnd_init_state(self, batch_size, device, force_target=False):
        target_mode = self.target_mode
        if force_target:
            target_mode = True
        if target_mode:
            target_encoder_hidden = self.target_encoder.make_rnd_init_state(batch_size, device)
            policy_hidden = super().make_rnd_init_state(batch_size, device)
            return target_encoder_hidden + policy_hidden
        else:
            return super().make_init_state(batch_size, device)

    @property
    def sample_std(self):
        if self.std_learnable:
            return self.log_std.weight[0].exp()
        else:
            return np.exp(self.log_std)

    def get_embedding_input(self, state, lst_state, lst_action, reward) -> torch.Tensor:
        embedding_inputs = [self.state_encoder(state)]
        if self.last_state_input:
            embedding_inputs.append(self.last_obs_encoder(lst_state))
        if self.last_action_input:
            embedding_inputs.append(self.last_act_encoder(lst_action))
        if self.reward_input:
            embedding_inputs.append(self.reward_encoder(reward))
        embedding_input = torch.cat(embedding_inputs, dim=-1)
        return embedding_input

    def get_privileged_embedding(self, privileged_state, hidden):
        privileged_embedding, hidden = self.privileged_encoder.forward(privileged_state, hidden)
        if not self.continuous_privileged_vector:
            return torch.sign(privileged_embedding) / 4, hidden
        return privileged_embedding, hidden

    def _soft_clamp(self, x: torch.Tensor, _min, _max) -> torch.Tensor:
        if _max is not None:
            x = _max - self.soft_plus(_max - x)
        if _min is not None:
            x = _min + self.soft_plus(x - _min)
        return x

    def get_target_embedding(self, target_state, last_action, hidden):
        target_embedding, logstd, hidden = self.target_encoder.forward(target_state, last_action, hidden)
        logstd = self._soft_clamp(logstd, -5.298, 0)
        return target_embedding, logstd, hidden

    def discretize_embedding(self, embedding):
        if self.continuous_privileged_vector:
            return embedding

        # embedding[torch.abs(embedding) < 1.5e-2] = 0.0
        embedding = torch.sign(embedding)
        if self.valid_priv_vector_dim == 1:
            embedding[..., -1] = 0
        return embedding

    def process_privileged_state(self, privileged_state, last_action, hidden, embedding_noise=None,
                                 target_logstd=None, target_flag=None):
        privileged_embedding, hidden = self.get_privileged_embedding(privileged_state, hidden)
        non_common_dim = self.privileged_encoder.privileged_image_dim + self.privileged_encoder.privileged_vector_dim
        if non_common_dim > 0:
            common_obs = privileged_state[..., :-non_common_dim]
        else:
            common_obs = privileged_state
        if embedding_noise is not None:
            if target_flag is not None:
                privileged_embedding_2 = (1 - target_flag) * (embedding_noise + privileged_embedding) + target_flag * embedding_noise
            else:
                embedding_noise_mask = embedding_noise.abs().sum(dim=-1, keepdim=True) > 0
                privileged_embedding_2 = (~embedding_noise_mask) * privileged_embedding + embedding_noise
        else:
            privileged_embedding_2 = privileged_embedding

        privileged_embedding_2 = self.discretize_embedding(privileged_embedding_2)

        if target_logstd is None or not self.target_logstd_input:
            processed_state = torch.cat((privileged_embedding_2, common_obs), dim=-1)
        else:
            if self.no_logstd_output:
                target_logstd = target_logstd * 0.0
            target_logstd = self.target_logstd_encoder.forward(target_logstd)
            processed_state = torch.cat((privileged_embedding_2, common_obs, target_logstd), dim=-1)

        return processed_state, privileged_embedding, hidden

    def process_target_state(self, target_state, last_action, hidden, embedding_noise=None, target_logstd=None, target_flag=None):
        target_embedding, logstd, hidden = self.get_target_embedding(target_state, last_action, hidden)
        if target_logstd is not None:
            logstd = target_logstd
        non_common_dim = self.target_encoder.target_image_dim + self.target_encoder.target_vector_dim
        if non_common_dim > 0:
            common_obs = target_state[..., :-(non_common_dim)]
        else:
            common_obs = target_state
        if self.target_logstd_input:
            MIN_LOGSTD, MAX_LOGSTD = -10, 1
            # MIN_LOGSTD, MAX_LOGSTD = -15, 2
            target_logstd_norm = (torch.clamp(logstd, min=MIN_LOGSTD, max=MAX_LOGSTD) - MIN_LOGSTD) / (
                        MAX_LOGSTD - MIN_LOGSTD) * 2 - 1
            target_logstd_norm = target_logstd_norm
            if self.no_logstd_output:
                target_logstd_norm = target_logstd_norm * 0.0
            # target_logstd_norm = target_logstd_norm
            if self.mean_target_input:
                target_logstd_norm = torch.cat((target_logstd_norm, target_embedding), dim=-1)
            target_logstd_norm = self.target_logstd_encoder.forward(target_logstd_norm)
            if not self.baseline_mode:
                target_embedding = self.discretize_embedding(target_embedding)
            processed_state = torch.cat((target_embedding, common_obs, target_logstd_norm), dim=-1)
        else:
            if not self.baseline_mode:
                target_embedding = self.discretize_embedding(target_embedding)
            processed_state = torch.cat((target_embedding, common_obs), dim=-1)

        return processed_state, target_embedding, hidden

    def make_target_encoder(self, image_dim, common_obs_dim, action_dim, ebd_dim, rnn_type, middle_dim,
                            output_activation, target_network_learn_std, use_camera, use_scan_dot):
        if use_camera:
            target_encoder = TargetObsEncoder2Head(image_dim, 0, common_obs_dim, action_dim,
                                                        ebd_dim, rnn_type, middle_dim, output_activation,
                                                        target_network_learn_std)
        else:
            if use_scan_dot:
                target_encoder = TargetObsEncoderNoImage2Head(132, 0, common_obs_dim, action_dim,
                                                               ebd_dim, rnn_type, middle_dim,
                                                               output_activation, target_network_learn_std, layer_norm=False,
                                                              maximum_image_encoder=False)
            else:
                target_encoder = TargetObsEncoderNoScanDot2Head(132, 0, common_obs_dim, action_dim,
                                                                     ebd_dim, rnn_type, middle_dim,
                                                                     output_activation, target_network_learn_std)
        return target_encoder

    def get_mean_std(self, privileged_state: torch.Tensor, lst_state: torch.Tensor,
                     lst_action: torch.Tensor, rnn_memory: Optional[RNNHidden],
                     reward: Optional[torch.Tensor]=None, detach_embedding: bool=False,
                     embedding_noise: Optional[torch.Tensor]=None,
                     target_logstd: Optional[torch.Tensor]=None, force_target=False,
                     target_flag: Optional[torch.Tensor]=None):
        target_mode = self.target_mode
        if force_target:
            target_mode = True
        encoder = self.privileged_encoder if not target_mode else self.target_encoder
        process_state_func = self.process_privileged_state if not target_mode else self.process_target_state

        hidden_encoder = rnn_memory[:encoder.rnn_num] if rnn_memory is not None else None
        hidden_policy = rnn_memory[encoder.rnn_num:] if rnn_memory is not None else None
        if embedding_noise is not None:
            last_embedding_noise, current_embedding_noise = torch.chunk(embedding_noise, 2, dim=-1)
        else:
            last_embedding_noise, current_embedding_noise = None, None
        state, privileged_embedding, hidden_encoder = process_state_func(privileged_state, lst_action,
                                                                         hidden_encoder, current_embedding_noise,
                                                                         target_logstd, target_flag)
        if self.last_state_input:
            lst_state_input = lst_state
            if target_mode:
                if state.shape[-2] == 1:
                    # TODO: check here
                    lst_state = self.lst_processed_state if self.lst_processed_state is not None else torch.zeros_like(state)
                    self.lst_processed_state = state
                    if self.target_logstd_input:
                        lst_state = lst_state[..., :self.last_obs_dim]
                else:
                    lst_state = torch.roll(state, 1, dims=(-2,))[..., :self.last_obs_dim]
            else:
                lst_state, _, _ = process_state_func(lst_state, lst_action, None,
                                                     last_embedding_noise, None, target_flag)
            lst_state_input_max = lst_state_input.abs().max(dim=-1, keepdim=True).values
            mask = lst_state_input_max == 0.0
            lst_state = lst_state * (~mask.detach())
        embedding_input = self.get_embedding_input(state, lst_state, lst_action, reward)
        model_output, hidden_policy, embedding_output, full_rnn_memory = self.meta_forward(embedding_input, state,
                                                                                        hidden_policy, detach_embedding)
        if hidden_encoder is not None:
            rnn_memory = hidden_encoder + hidden_policy
        else:
            rnn_memory = hidden_policy
        std = torch.ones_like(model_output) * self.sample_std
        return model_output, std, privileged_embedding, rnn_memory, full_rnn_memory

    def forward(self, state: torch.Tensor, lst_state: torch.Tensor, lst_action: torch.Tensor,
                rnn_memory: Optional[RNNHidden], reward: Optional[torch.Tensor]=None, detach_embedding: bool=False,
                embedding_noise: Optional[torch.Tensor]=None,
                target_logstd: Optional[torch.Tensor]=None, force_target=False, target_flag: Optional[torch.Tensor]=None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, RNNHidden, Optional[RNNHidden]
    ]:
        action_mean, std, embedding_output, rnn_memory, full_rnn_memory = self.get_mean_std(state, lst_state, lst_action, rnn_memory,
                                                                                            reward, detach_embedding, embedding_noise,
                                                                                            target_logstd, force_target, target_flag)
        action_mean, action_sample, log_prob = self.process_model_out(action_mean, std)
        return action_mean, embedding_output, action_sample, log_prob, rnn_memory, full_rnn_memory

    def entropy(self):
        if self.std_learnable:
            return 0.5 * (1 + np.log(2 * np.pi) + 2 * self.log_std.weight[0]).sum()
        else:
            return 0.5 * (1 + np.log(2 * np.pi) + 2 * self.log_std) * self.action_dim

    def process_model_out(self, action_mean, std):
        normal_dist = torch.distributions.Normal(action_mean, std)
        action_sample = normal_dist.sample()
        log_prob = normal_dist.log_prob(action_sample).sum(dim=-1, keepdim=True)
        return action_mean, action_sample, log_prob

    def logp(self, state: torch.Tensor, action: torch.Tensor, lst_state: torch.Tensor, lst_action: torch.Tensor,
             rnn_memory: Optional[RNNHidden], reward: Optional[torch.Tensor]=None, detach_embedding: bool=False,
             require_embedding=False, embedding_noise: Optional[torch.Tensor]=None, target_logstd: Optional[torch.Tensor]=None,
             target_flag: Optional[torch.Tensor]=None):
        action_mean, std, embedding_output, rnn_memory, full_rnn_memory = self.get_mean_std(state, lst_state, lst_action, rnn_memory, reward,
                                                                                            detach_embedding, embedding_noise, target_logstd, target_flag=target_flag)
        normal_dist = torch.distributions.Normal(action_mean, std)
        log_prob = normal_dist.log_prob(action)
        return log_prob.sum(-1, keepdim=True) if not require_embedding else log_prob.sum(-1, keepdim=True), embedding_output

    def forward_embedding(self, state: torch.Tensor, lst_state: torch.Tensor,
                          lst_action: torch.Tensor, rnn_memory: Optional[RNNHidden],
                          reward: Optional[torch.Tensor]):
        embedding_input = self.get_embedding_input(state, lst_state, lst_action, reward)
        embedding_output, embedding_rnn_memory, embedding_full_memory = self.get_embedding(embedding_input, rnn_memory)
        return embedding_output, embedding_rnn_memory, embedding_full_memory