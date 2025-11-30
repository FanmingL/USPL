from ..models.contextual_model import ContextualModel
import torch
from ..models.mlp_base import MLPBase
from ..models.RNNHidden import RNNHidden
from typing import List, Union, Tuple, Dict, Optional
from .utils import nearest_power_of_two, nearest_power_of_two_half


class ContextualPPOValue(ContextualModel):
    def __init__(self, state_dim, action_dim, embedding_size, embedding_hidden, embedding_activations,
                 embedding_layer_type, uni_model_hidden,
                 uni_model_activations, uni_model_layer_type, fix_rnn_length, uni_model_input_mapping_dim: int=0,
                 reward_input=False, last_action_input=True, last_state_input=False,
                 separate_encoder=False, use_camera=False, use_scan_dot=False, target_logstd_input=True, mean_target_input=False, name='ContextualSACValue'):
        self.scan_ebd_dim = 20
        self.use_camera = use_camera
        self.mean_target_input = mean_target_input
        self.original_image_dim_row, self.original_image_dim_col = 58, 87
        self.image_dim = 1 * self.original_image_dim_row * self.original_image_dim_col
        self.logstd_input = target_logstd_input
        self.scan_origin_dim = 132 if use_scan_dot else 0
        if use_scan_dot:
            state_dim = state_dim - self.scan_origin_dim + self.scan_ebd_dim
        else:
            state_dim = state_dim
        self.embedding_state_dim = state_dim
        self.use_scan_dot = use_scan_dot
        self.reward_input = reward_input
        self.last_action_input = last_action_input
        self.last_state_input = last_state_input
        self.reward_dim = 1 if self.reward_input else 0
        self.last_act_dim = action_dim if self.last_action_input else 0
        self.last_obs_dim = state_dim if self.last_state_input else 0
        if embedding_size == 'auto':
            embedding_size = nearest_power_of_two_half(state_dim)
        if uni_model_input_mapping_dim == 'auto':
            uni_model_input_mapping_dim = nearest_power_of_two(state_dim + action_dim)
        self.separate_encoder = separate_encoder
        if separate_encoder:
            basic_embedding_dim = 128
            self.state_encoder = torch.nn.Linear(state_dim, basic_embedding_dim)
            self.last_act_encoder = torch.nn.Linear(self.last_act_dim,
                                                    basic_embedding_dim) if self.last_act_dim else None
            self.reward_encoder = torch.nn.Linear(self.reward_dim, basic_embedding_dim) if self.reward_dim else None
            self.last_obs_encoder = torch.nn.Linear(self.last_obs_dim,
                                                    basic_embedding_dim) if self.last_obs_dim else None

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

        uni_model_input_size = state_dim
        self.state_input_encoder = torch.nn.Identity()
        if uni_model_input_mapping_dim > 0:
            if separate_encoder:
                self.state_input_encoder = torch.nn.Linear(state_dim, uni_model_input_mapping_dim)
                uni_model_input_size = uni_model_input_mapping_dim
                uni_model_input_mapping_dim = 0
        logstd_dim = 2 if not self.mean_target_input else 4
        if not target_logstd_input:
            logstd_dim = 0
        super(ContextualPPOValue, self).__init__(embedding_input_size=cum_dim,
                                                 embedding_size=embedding_size,
                                                 embedding_hidden=embedding_hidden,
                                                 embedding_activations=embedding_activations,
                                                 embedding_layer_type=embedding_layer_type,
                                                 uni_model_input_size=uni_model_input_size + logstd_dim,
                                                 uni_model_output_size=1,
                                                 uni_model_hidden=uni_model_hidden,
                                                 uni_model_activations=uni_model_activations,
                                                 uni_model_layer_type=uni_model_layer_type,
                                                 fix_rnn_length=fix_rnn_length,
                                                 uni_model_input_mapping_dim=uni_model_input_mapping_dim,
                                                 uni_model_input_mapping_activation=embedding_activations[-1],
                                                 name=name)
        self.uni_model_input_mapping_activation_func = self.embedding_network.activation_dict[embedding_activations[-1]]()
        if separate_encoder:
            self.contextual_register_rnn_base_module(self.state_encoder, 'state_encoder')
            if self.last_act_encoder is not None:
                self.contextual_register_rnn_base_module(self.last_act_encoder, 'last_act_encoder')
            if self.last_obs_encoder is not None:
                self.contextual_register_rnn_base_module(self.last_obs_encoder, 'last_obs_encoder')
            if self.reward_encoder is not None:
                self.contextual_register_rnn_base_module(self.reward_encoder, 'reward_encoder')
            if self.state_input_encoder is not None:
                self.contextual_register_rnn_base_module(self.state_input_encoder, 'state_input_encoder_q')
        if use_scan_dot:
            self.scan_encoder = MLPBase(self.scan_origin_dim, self.scan_ebd_dim, [256, 128], ['elu', 'elu', 'elu'])
            self.contextual_register_rnn_base_module(self.scan_encoder, 'scan_encoder')
        self.state_dim = state_dim
        self.action_dim = action_dim

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

    def state_encoder_func(self, state):
        s = self.state_input_encoder(state)
        if self.separate_encoder:
            return self.uni_model_input_mapping_activation_func(s)
        else:
            return s

    def process_state(self, state):
        if self.use_scan_dot:
            scan_obs = state[..., -self.scan_origin_dim:]
            scan_ebd = self.scan_encoder(scan_obs)
            state = torch.cat((state[..., :-self.scan_origin_dim], scan_ebd), dim=-1)
        return state

    def forward(self, state: torch.Tensor, lst_state: torch.Tensor,
                lst_action: torch.Tensor, action: torch.Tensor, rnn_memory: Optional[RNNHidden],
                reward: Optional[torch.Tensor], target_logstd: Optional[torch.Tensor], detach_embedding: bool=False) -> Tuple[
        torch.Tensor, torch.Tensor, RNNHidden, Optional[RNNHidden]
    ]:
        state = self.process_state(state)
        if self.last_state_input:
            lst_state = self.process_state(lst_state)
        embedding_input = self.get_embedding_input(state, lst_state, lst_action, reward)
        state_logstd = torch.cat((self.state_encoder_func(state), target_logstd), dim=-1) if self.logstd_input else self.state_encoder_func(state)
        value, rnn_memory, embedding_output, full_rnn_memory = \
            self.meta_forward(embedding_input, state_logstd, rnn_memory, detach_embedding)
        if self.use_camera:
            embedding_output = torch.cat((embedding_output, state[..., -self.scan_ebd_dim:]), dim=-1)
        return value, embedding_output, rnn_memory, full_rnn_memory

    def forward_embedding(self, state: torch.Tensor, lst_state: torch.Tensor,
                lst_action: torch.Tensor, rnn_memory: Optional[RNNHidden],
                reward: Optional[torch.Tensor]):
        embedding_input = self.get_embedding_input(state, lst_state, lst_action, reward)
        embedding_output, embedding_rnn_memory, embedding_full_memory = self.get_embedding(embedding_input, rnn_memory)
        return embedding_output, embedding_rnn_memory, embedding_full_memory
