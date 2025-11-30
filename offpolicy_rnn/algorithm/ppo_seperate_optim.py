from ..utility.sample_utility import unorm_act, norm_act, n2t, n2t_2dim, t2n, eval_inprocess, policy_eval
from typing import List, Union, Tuple, Dict, Optional, Callable
from tqdm import tqdm
import numpy as np
from legged_gym.utils import webviewer
import torch
from .ppo import PPO
from torch.cuda.amp import GradScaler
from ..utility.compute_utils import estimate_advantages

def _insert_rnn_parameter(module, rnn_lr, l2_norm, name):
    return {"params": list(module.parameters(True)),
         'lr': rnn_lr,
         'weight_decay': l2_norm,
         "name": f"rnn-{name}"}


def _insert_mlp_parameter(module, rnn_lr, l2_norm, name):
    return {"params": list(module.parameters(True)),
            # 'lr': rnn_lr,
            # 'weight_decay': l2_norm,
            "name": f"mlp-{name}"}
def prepare_param_list(model, rnn_lr, l2_norm, exclude=None):
    param_list = []
    for k, v in model.contextual_modules.items():
        if exclude is not None and k in exclude:
            continue
        def rnn_parameter(mod):
            return _insert_rnn_parameter(mod, rnn_lr, l2_norm, type(v).__name__)
        def mlp_parameter(mod):
            return _insert_mlp_parameter(mod, rnn_lr, l2_norm, type(v).__name__)

        # This is paper implementation.
        if k.endswith('encoder'):
            param_list.append(rnn_parameter(v))
        elif k == 'embedding_model':
            param_list.append(rnn_parameter(v.layer_list[0]))
            for i in range(1, len(model.embedding_network.layer_list) - 1):
                param_list.append(rnn_parameter(v.layer_list[i]))
            param_list.append(rnn_parameter(v.layer_list[-1]))
            param_list.append(rnn_parameter(v.activation_list))
        else:
            param_list.append(mlp_parameter(v))

    return param_list


class PPO_SEPERATE_OPTIM(PPO):
    def __init__(self, parameter, env=None):
        super().__init__(parameter, env)
        assert self.parameter.value_net_num == 1

        for network in (self.values[0].embedding_network.layer_list + self.target_values[0].embedding_network.layer_list
                        + self.values[0].uni_network.layer_list + self.target_values[0].uni_network.layer_list):
            if hasattr(network, 'desire_ndim'):
                print(f'set desire_ndim')
                network.desire_ndim = 4
            if hasattr(network, 'in_proj') and hasattr(network.in_proj, 'desire_ndim'):
                print(f'set desire_ndim')
                network.in_proj.desire_ndim = 4
        self.logger(f'replay buffer skip len: {self._get_skip_len()}')
        if self._get_whether_require_amp():
            self.logger(f'enable AMP, introducing grad scalar!!')
            self.amp_scalar = GradScaler()
            self.amp_scalar_critic = GradScaler()
        else:
            self.amp_scalar = None
            self.amp_scalar_critic = None

        self.logger(f'max trajectory len: {self.env_info["max_trajectory_len"]}')


        self.logger(
            f'random test: policy L2 norm: {self.policy.l2_norm_square()}, value[0] L2 norm: {self.values[0].l2_norm_square()}')

        policy_param_list = prepare_param_list(self.policy, self.parameter.rnn_policy_lr, self.parameter.policy_l2_norm,
                                               exclude=['target_encoder_subnet', 'privileged_encoder_moving_aver'])

        self.optimizer_policy = self.optim_class(policy_param_list,
                                                 lr=self.parameter.policy_lr,
                                                 weight_decay=self.parameter.policy_l2_norm)
        self.optimizer_target_encoder = self.optim_class(self.policy.target_encoder.parameters(), lr=self.parameter.target_encoder_lr,
                                                         weight_decay=self.parameter.target_encoder_l2_norm)
        self.target_encoder_optimization_cnt = 0
        self.target_encoder_accumulation_steps = 1

        value_param_list = [item for value in self.values for item in
                            prepare_param_list(value, self.parameter.rnn_value_lr, self.parameter.value_l2_norm)]

        self.optimizer_value = self.optim_class(value_param_list, lr=self.parameter.value_lr,
                                                weight_decay=self.parameter.value_l2_norm)

    def _get_whether_require_amp(self):
        for rnn_base in [self.values[0].uni_network, self.values[0].embedding_network,
                         self.policy.uni_network, self.policy.embedding_network]:
            for i in range(len(rnn_base.layer_type)):
                if 'gpt' in rnn_base.layer_type[i]:
                    return True
                    # try disabling GradScaler for GPT
                    # return False
        return False

    def _get_gae(self):
        pass

    def _ppo_batch_train_direct_target(self, state: torch.Tensor, priv_state: torch.Tensor, lst_state: torch.Tensor, lst_priv_state: torch.Tensor, lst_action: torch.Tensor, action: torch.Tensor,
                     adv: torch.Tensor, reward_input: torch.Tensor, rets: torch.Tensor, old_logp: torch.Tensor, mask: torch.Tensor,
                         policy_hidden, value_hidden, old_policy_embedding, old_value_embedding, target_embedding,
                         target_encoder_hidden, embedding_noise, target_logstd_input, priv_state_offpi, state_offpi, lst_action_offpi,
                         target_encoder_hidden_offpi, embedding_noise_offpi, mask_offpi, train_target_only):
        value, value_embedding = self.values[0].forward(priv_state, lst_priv_state, lst_action, None, value_hidden,
                                                        reward_input, target_logstd_input)[:2]
        valid_num = mask.sum().detach()
        # 值函数损失函数
        value_loss = ((value - rets) * mask).pow(2).sum() / valid_num

        # 值函数网络权重更新
        self.optimizer_value.zero_grad()
        value_loss.backward()
        # 梯度裁剪
        value_norm = torch.nn.utils.clip_grad_norm_(self.values[0].parameters(), 40.)
        self.optimizer_value.step()
        value_loss = value_loss.item()

        log_probs, _ = self.policy.logp(state, action, lst_state, lst_action, policy_hidden,
                                        reward_input, embedding_noise=embedding_noise,
                                        target_logstd=target_logstd_input,
                                        require_embedding=True, )
        # 新旧策略的KL散度
        kl = old_logp - log_probs

        # exp(-KL)
        ratio = torch.exp(-kl)
        # KL散度均值，用于监控新旧策略之间的差异
        kl_mean = (kl * mask).sum() / valid_num
        if kl_mean.item() > 0.1:
            return {}
        # PPO actor损失计算
        surr1 = ratio * adv
        # TODO: epsilon
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * adv
        policy_loss = -(torch.min(surr1,
                                  surr2) * mask).sum() / valid_num

        # 策略熵计算
        # policy_std = self.policy.logstd.exp().mean()
        if self.parameter.std_learnable and self.parameter.entropy_coeff > 0:
            entropy = self.policy.entropy()
            policy_loss = policy_loss - entropy * self.parameter.entropy_coeff
            entropy = entropy.item()
        else:
            entropy = self.policy.entropy()

        # 策略网络权重更新
        self.optimizer_policy.zero_grad()
        self.optimizer_target_encoder.zero_grad()
        policy_loss.backward()
        policy_parameter = []
        for k, v in self.policy.contextual_modules.items():
            if not k == 'target_encoder_subnet':
                policy_parameter += list(v.parameters(True))
        target_encoder_parameter = list(self.policy.target_encoder.parameters())
        target_encoder_l2norm = sum([torch.sum(param ** 2) for param in target_encoder_parameter])
        policy_norm = torch.nn.utils.clip_grad_norm_(policy_parameter, 40.)
        policy_norm_target = torch.nn.utils.clip_grad_norm_(target_encoder_parameter, 40.)
        self.optimizer_policy.step()
        self.optimizer_target_encoder.step()
        return dict(
            policy_loss=policy_loss.item(),
            value_loss=value_loss,
            kl=kl_mean.item(),
            entropy=entropy,
            std=self.policy.sample_std.mean().item(),
            value_norm=float(value_norm),
            policy_norm=float(policy_norm),
            policy_norm_target=float(policy_norm_target),
            target_encoder_l2norm=float(target_encoder_l2norm)
        )


    def _ppo_batch_train(self, state: torch.Tensor, priv_state: torch.Tensor, lst_state: torch.Tensor, lst_priv_state: torch.Tensor, lst_action: torch.Tensor, action: torch.Tensor,
                     adv: torch.Tensor, reward_input: torch.Tensor, rets: torch.Tensor, old_logp: torch.Tensor, mask: torch.Tensor,
                         policy_hidden, value_hidden, old_policy_embedding, old_value_embedding, target_embedding, target_flag,
                         target_encoder_hidden, embedding_noise, target_logstd_input, priv_state_offpi, state_offpi, lst_action_offpi,
                         target_encoder_hidden_offpi, embedding_noise_offpi, mask_offpi, train_target_only):
        """
        进行一个batch的PPO优化
        Args:
            state: 状态
            lst_action: 上一时刻动作
            action: 动作
            adv: 优势函数（GAE）
            rets: 带gamma衰减的return
            old_logp: sample时action被采到的概率的对数值
            mask: 数据有效标志
        Returns:
            训练指标，各种loss
        """
        policy_embedding_before_training = old_policy_embedding
        with torch.no_grad():
            old_policy_embedding, _ = self.policy.get_privileged_embedding(priv_state_offpi, None)
            old_policy_embedding = old_policy_embedding.detach()
        outlier_clip_factor = 100.0
        if train_target_only:
            with torch.no_grad():
                last_embedding_noise_, current_embedding_noise_ = torch.chunk(embedding_noise_offpi, 2, dim=-1)
                current_embedding_noise_ = ((current_embedding_noise_.abs() + 1) * mask_offpi).mean(dim=-1, keepdim=True)
                target_policy_mask = (current_embedding_noise_ > 1)
                non_target_policy_mask = (current_embedding_noise_ == 1)
                target_num_cnt = target_policy_mask.sum()
                non_target_num_cnt = non_target_policy_mask.sum()

            target_embedding, target_logstd, _ = self.policy.get_target_embedding(state_offpi, lst_action_offpi,
                                                                   target_encoder_hidden_offpi)

            if self.target_encoder_learn_std:
                target_std = target_logstd.exp()
                target_mask = mask_offpi[:target_embedding.shape[0]]
                old_policy_embedding_clamp = torch.clamp(
                    old_policy_embedding[:target_embedding.shape[0]] - target_embedding, -outlier_clip_factor * target_std, outlier_clip_factor * target_std) + target_embedding
                old_policy_embedding_clamp = old_policy_embedding_clamp.detach()
                target_ebd_logp = (torch.distributions.Normal(target_embedding, target_std).log_prob(old_policy_embedding_clamp) * target_mask).mean(dim=-1)
                target_encoder_loss = -target_ebd_logp.sum() / target_mask.sum()
                mean_std = (target_std * target_mask).mean(dim=-1).sum() / target_mask.sum()
                target_policy_average_std = (target_std * target_policy_mask).mean(dim=-1).sum() / target_num_cnt if target_num_cnt > 0 else (target_std * target_policy_mask).mean(dim=-1).sum()
                non_target_policy_average_std = (target_std * non_target_policy_mask).mean(dim=-1).sum() / non_target_num_cnt if non_target_num_cnt else (target_std * non_target_policy_mask).mean(dim=-1).sum()
            else:
                target_mask = mask[:target_embedding.shape[0]]
                target_ebd_logp = ((target_embedding - old_policy_embedding[:target_embedding.shape[0]]) * target_mask).pow(
                    2).mean(dim=-1)
                target_encoder_loss = target_ebd_logp.sum() / target_mask.sum()
                target_policy_average_std = non_target_policy_average_std = target_encoder_loss * 0
            if target_mask.sum() > 0:
                (target_encoder_loss / self.target_encoder_accumulation_steps).backward()
                target_encoder_parameter = list(self.policy.target_encoder.parameters())
                policy_norm_target = torch.nn.utils.clip_grad_norm_(target_encoder_parameter, 40.)
                if self.get_schedule_weight() > 0:
                    self.target_encoder_optimization_cnt += 1
                    if self.target_encoder_accumulation_steps == self.target_encoder_optimization_cnt:
                        self.optimizer_target_encoder.step()
                        self.optimizer_target_encoder.zero_grad()
                        self.target_encoder_optimization_cnt = 0
                else:
                    self.target_encoder_optimization_cnt = 0
                    self.optimizer_target_encoder.zero_grad()
            else:
                policy_norm_target = 0.0
            # embedding_noise_offpi (B, L, C)
            target_policy_ebd_logp = (target_ebd_logp.unsqueeze(-1) * target_policy_mask).sum() / target_num_cnt if target_num_cnt > 0 else (target_ebd_logp.unsqueeze(-1) * target_policy_mask).sum()
            non_target_policy_ebd_logp = (target_ebd_logp.unsqueeze(-1) * non_target_policy_mask).sum() / non_target_num_cnt if non_target_num_cnt > 0 else (target_ebd_logp.unsqueeze(-1) * non_target_policy_mask).sum()
            logs = dict(
                # target_encoder_loss=target_encoder_loss.item(),
                policy_norm_target=float(policy_norm_target),
                # target_encoder_std=mean_std.item() if self.target_encoder_learn_std else 0,
                target_encoder_std_max=(target_logstd * target_mask).max().exp().item() if self.target_encoder_learn_std else 0,
                target_encoder_std_min=(target_logstd * target_mask).min().exp().item() if self.target_encoder_learn_std else 0,
                # target_num_cnt=target_num_cnt.item(),
                # non_target_num_cnt=non_target_num_cnt.item(),
                # target_policy_average_std=target_policy_average_std.item(),
                # non_target_policy_average_std=non_target_policy_average_std.item(),
                # target_policy_ebd_logp=target_policy_ebd_logp.item(),
                # non_target_policy_ebd_logp=non_target_policy_ebd_logp.item()
            )
            if target_mask.sum() > 0:
                logs['target_encoder_loss'] = target_encoder_loss.item()
                logs['target_encoder_std'] = mean_std.item() if self.target_encoder_learn_std else 0
            if non_target_num_cnt > 0:
                logs['non_target_policy_average_std'] = non_target_policy_average_std.item()
                logs['non_target_policy_ebd_logp'] = non_target_policy_ebd_logp.item()
                logs['non_target_num_cnt'] = non_target_num_cnt.item()
            if target_num_cnt > 0:
                logs['target_policy_average_std'] = target_policy_average_std.item()
                logs['target_policy_ebd_logp'] = target_policy_ebd_logp.item()
                logs['target_num_cnt'] = target_num_cnt.item()
            return logs
        value_ebd_factor = 0.0
        policy_ebd_factor = 0.0
        if self.get_schedule_weight() > 0:
            policy_ebd_factor = self.get_schedule_weight() * self.parameter.policy_embedding_loss_factor
        # 值函数计算
        value, value_embedding = self.values[0].forward(priv_state, lst_priv_state, lst_action, None, value_hidden, reward_input, target_logstd_input)[:2]
        valid_num = mask.sum().detach()
        # 值函数损失函数
        value_embedding_loss = torch.clamp_min(((value_embedding - old_value_embedding) * mask).pow(2), min=0.01 ** 2).mean(dim=-1).sum() / valid_num
        value_loss = ((value - rets) * mask).pow(2).sum() / valid_num

        # 值函数网络权重更新
        self.optimizer_value.zero_grad()
        value_loss.backward()
        # 梯度裁剪
        value_norm = torch.nn.utils.clip_grad_norm_(self.values[0].parameters(), 40.)
        self.optimizer_value.step()
        value_loss = value_loss.item()
        value_embedding_loss = value_embedding_loss.item()

        # action在当前policy下被执行的概率
        log_probs, _ = self.policy.logp(priv_state, action, lst_priv_state, lst_action, policy_hidden,
                                                       reward_input, embedding_noise=embedding_noise,
                                                       target_logstd=target_logstd_input,
                                                       require_embedding=True, target_flag=target_flag)
        policy_embedding_current_batch, _ = self.policy.get_privileged_embedding(priv_state, None)
        policy_embedding, _ = self.policy.get_privileged_embedding(priv_state_offpi, None)
        # 新旧策略的KL散度
        kl = old_logp - log_probs

        # exp(-KL)
        ratio = torch.exp(-kl)
        # KL散度均值，用于监控新旧策略之间的差异
        kl_mean = (kl * mask).sum() / valid_num
        # PPO actor损失计算
        surr1 = ratio * adv
        # TODO: epsilon
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * adv
        # policy_embedding_loss = torch.clamp_min(((policy_embedding - old_policy_embedding) * mask).pow(2), min=0.01 ** 2).mean(dim=-1).sum() / valid_num

        target_embedding, target_logstd, _ = self.policy.get_target_embedding(state_offpi, lst_action_offpi, target_encoder_hidden_offpi)

        if self.target_encoder_learn_std:
            target_std = target_logstd.exp()
            old_policy_embedding_clamp = torch.clamp(
                old_policy_embedding[:target_embedding.shape[0]] - target_embedding, -outlier_clip_factor * target_std,
                outlier_clip_factor * target_std) + target_embedding
            old_policy_embedding_clamp = old_policy_embedding_clamp.detach()
            target_mask = mask_offpi
            target_encoder_loss = -(torch.distributions.Normal(target_embedding, target_std).log_prob(old_policy_embedding_clamp) * target_mask).mean(dim=-1).sum() / target_mask.sum()
            mean_std = (target_std * target_mask).mean(dim=-1).sum() / target_mask.sum()
        else:
            target_mask = mask[:target_embedding.shape[0]]
            target_encoder_loss = ((target_embedding - old_policy_embedding[:target_embedding.shape[0]]) * target_mask).pow(
                2).mean(dim=-1).sum() / target_mask.sum()

        policy_embedding_loss = ((policy_embedding[:target_embedding.shape[0]] - target_embedding.detach()) * target_mask).pow(2).mean(dim=-1).sum() / target_mask.sum()
        if target_mask.sum() == 0:
            policy_embedding_loss = (
                        (policy_embedding[:target_embedding.shape[0]] - target_embedding.detach()) * target_mask).pow(
                2).mean(dim=-1).sum()
            target_encoder_loss = -(torch.distributions.Normal(target_embedding, target_std).log_prob(old_policy_embedding_clamp) * target_mask).mean(dim=-1).sum()
        policy_divergence_loss = (((policy_embedding_current_batch - policy_embedding_before_training) * mask).pow(2).mean(dim=-1)).sum() / mask.sum()
        policy_divergence_loss_factor = self.parameter.policy_divergence_loss_factor if self.iter_train >= self.threshold_iter else 0
        embedding_norm_loss = (policy_embedding_current_batch * mask).pow(2).mean(dim=-1).sum() / mask.sum()
        policy_loss = -(torch.min(surr1, surr2) * mask).sum() / valid_num + policy_ebd_factor * policy_embedding_loss + target_encoder_loss / self.target_encoder_accumulation_steps + policy_divergence_loss * policy_divergence_loss_factor #  + embedding_norm_loss * 0.002

        # 策略熵计算
        # policy_std = self.policy.logstd.exp().mean()
        if self.parameter.std_learnable and self.parameter.entropy_coeff > 0:
            entropy = self.policy.entropy()
            policy_loss = policy_loss - entropy * self.parameter.entropy_coeff
            entropy = entropy.item()
        else:
            entropy = self.policy.entropy()

        # 策略网络权重更新
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        policy_parameter = []
        for k, v in self.policy.contextual_modules.items():
            if not k == 'target_encoder_subnet':
                policy_parameter += list(v.parameters(True))
        target_encoder_parameter = list(self.policy.target_encoder.parameters())
        target_encoder_l2norm = sum([torch.sum(param ** 2) for param in target_encoder_parameter])

        policy_norm = torch.nn.utils.clip_grad_norm_(policy_parameter, 40.)
        policy_norm_target = torch.nn.utils.clip_grad_norm_(target_encoder_parameter, 40.)
        self.optimizer_policy.step()
        if self.get_schedule_weight() > 0:
            self.target_encoder_optimization_cnt += 1
            if self.target_encoder_accumulation_steps == self.target_encoder_optimization_cnt:
                self.optimizer_target_encoder.step()
                self.optimizer_target_encoder.zero_grad()
                self.target_encoder_optimization_cnt = 0
        else:
            self.optimizer_target_encoder.zero_grad()
            self.target_encoder_optimization_cnt = 0

        return dict(
            policy_loss=policy_loss.item(),
            value_loss=value_loss,
            kl=kl_mean.item(),
            entropy=entropy,
            std=self.policy.sample_std.mean().item(),
            # value_norm=value_norm.item(),
            # policy_norm=policy_norm.item()
            value_norm=float(value_norm),
            policy_norm=float(policy_norm),
            policy_norm_target=float(policy_norm_target),
            policy_embedding_loss=policy_embedding_loss.item(),
            value_embedding_loss=value_embedding_loss,
            target_encoder_loss=target_encoder_loss.item(),
            target_encoder_std=mean_std.item() if self.target_encoder_learn_std else 0,
            policy_ebd_factor=policy_ebd_factor,
            policy_divergence_loss=policy_divergence_loss.item(),
            embedding_norm_loss=embedding_norm_loss.item(),
            target_encoder_l2norm=float(target_encoder_l2norm)
        )


    def sample_offpolicy_batch_old(self, batch_size, device, max_traj_num=None, uncompress_state=True):
        batch_data, batch_size, traj_valid_indicators, traj_len_array = self.offpolicy_replay_buffer.sample_trajs(
            batch_size / 2,
            None, randomize_mask=self.parameter.randomize_mask,
            valid_number_post_randomized=self.parameter.valid_number_post_randomized,
            equalize_data_of_each_traj=True,
            get_all=False,
            random_trunc_traj=self.parameter.random_trunc_traj,
            nest_stack_trajs=self.allow_nest_stack)
        batch_data_2, _, traj_valid_indicators_2, traj_len_array_2 = self.replay_buffer.sample_trajs(
            batch_size / 2,
            None, randomize_mask=self.parameter.randomize_mask,
            valid_number_post_randomized=self.parameter.valid_number_post_randomized,
            equalize_data_of_each_traj=True,
            get_all=False,
            random_trunc_traj=self.parameter.random_trunc_traj,
            nest_stack_trajs=self.allow_nest_stack)
        state, priv_state, last_state, last_priv_state, last_action, mask, reward_input, embedding_noise, rnn_start, target_flag = map(
            lambda x: getattr(batch_data, x), [
                'state', 'priv_state', 'last_state', 'last_priv_state', 'last_action', 'mask', 'reward_input', 'embedding_noise',
                'start', 'target_flag'
            ])
        state_2, priv_state_2, last_state_2, last_priv_state_2, last_action_2, mask_2, reward_input_2, embedding_noise_2, rnn_start_2, target_flag_2 = map(
            lambda x: getattr(batch_data_2, x), [
                'state', 'priv_state', 'last_state', 'last_priv_state', 'last_action', 'mask', 'reward_input',
                'embedding_noise',
                'start', 'target_flag'
            ])
        cpu_device = device

        if self.parameter.image_input and uncompress_state:
            state = n2t(state, cpu_device)
            state_2 = n2t(state_2, cpu_device)
            state = self.uncompress_state_offpolicy(state)
            state_2 = self.uncompress_state(state_2)
            state = t2n(state)
            state_2 = t2n(state_2)
        if traj_len_array.shape[-1] > traj_len_array_2.shape[-1]:
            traj_len_array_2 = np.concatenate((traj_len_array_2, np.zeros((traj_len_array_2.shape[0], traj_len_array.shape[-1] - traj_len_array_2.shape[-1]))), axis=-1)
        elif traj_len_array.shape[-1] < traj_len_array_2.shape[-1]:
            traj_len_array = np.concatenate((traj_len_array, np.zeros((traj_len_array.shape[0], traj_len_array_2.shape[-1] - traj_len_array.shape[-1]))), axis=-1)

        rnd_idx = np.random.permutation(state.shape[0] + state_2.shape[0])
        if max_traj_num is not None:
            rnd_idx = rnd_idx[:max_traj_num]
        state, priv_state, last_state, last_priv_state, last_action, mask, reward_input, embedding_noise, rnn_start, target_flag, traj_valid_indicators, traj_len_array = map(
            lambda x, y: np.concatenate((x, y), axis=0)[rnd_idx], [
                state, priv_state, last_state, last_priv_state, last_action, mask, reward_input, embedding_noise,
                rnn_start, target_flag, traj_valid_indicators, traj_len_array
            ],
            [
                state_2, priv_state_2, last_state_2, last_priv_state_2, last_action_2, mask_2, reward_input_2,
                embedding_noise_2, rnn_start_2, target_flag_2, traj_valid_indicators_2, traj_len_array_2
            ]
        )
        with torch.no_grad():
            state, priv_state, last_state, last_priv_state, last_action, mask, reward_input, embedding_noise, rnn_start, target_flag, traj_valid_indicators = map(
                lambda x: n2t(x, cpu_device), [
                    state, priv_state, last_state, last_priv_state, last_action, mask, reward_input, embedding_noise,
                    rnn_start, target_flag, traj_valid_indicators
                ])
            total_rnn_start = rnn_start.clone()
            total_valid_indicators = traj_valid_indicators.clone()
            total_valid_indicators[torch.where(torch.diff(traj_valid_indicators, dim=-2) == 1)] = 1
            total_rnn_start[torch.where(torch.diff(total_rnn_start, dim=-2) == -1)] = 0
            if self.parameter.randomize_first_hidden:
                policy_hidden = self.policy.make_rnd_init_state(state.shape[0], device=self.device)
            else:
                policy_hidden = self.policy.make_init_state(state.shape[0], device=self.device)
            with torch.no_grad():
                attention_mask = torch.from_numpy(traj_len_array).to(torch.get_default_dtype()).to(self.device)
                attention_mask = torch.cat((attention_mask, torch.zeros(
                    (attention_mask.shape[0], state.shape[-2] - attention_mask.shape[1]), device=self.device)), dim=-1)
                attention_mask = attention_mask.clone()
                attention_mask = attention_mask.to(torch.int)

            policy_hidden.set_rnn_start(rnn_start)
            policy_hidden.set_mask(traj_valid_indicators)
            policy_hidden.set_attention_concat_mask(attention_mask)

            target_encoder_hidden = self.policy.target_encoder.make_init_state(state.shape[0],
                                                                               device=device)
            target_encoder_hidden.copy_attachment_from_(policy_hidden)
        # embedding_noise_mask = (embedding_noise.abs().sum(dim=-1, keepdim=True) > 0).float()
        # mask = mask * target_flag
        return state, priv_state, last_priv_state, last_action, embedding_noise, mask, reward_input, policy_hidden, target_encoder_hidden


    def sample_offpolicy_batch(self, batch_size, device, max_traj_num=None):
        batch_data, batch_size, traj_valid_indicators, traj_len_array = self.offpolicy_replay_buffer.sample_trajs(
            batch_size,
            None, randomize_mask=self.parameter.randomize_mask,
            valid_number_post_randomized=self.parameter.valid_number_post_randomized,
            equalize_data_of_each_traj=True,
            get_all=False,
            random_trunc_traj=self.parameter.random_trunc_traj,
            nest_stack_trajs=self.allow_nest_stack)

        state, priv_state, last_state, last_priv_state, last_action, mask, reward_input, embedding_noise, rnn_start = map(
            lambda x: getattr(batch_data, x), [
                'state', 'priv_state', 'last_state', 'last_priv_state', 'last_action', 'mask', 'reward_input', 'embedding_noise',
                'start'
            ])

        cpu_device = device

        if self.parameter.image_input:
            state = n2t(state, cpu_device)
            state = self.uncompress_state_offpolicy(state)
            state = t2n(state)
        rnd_idx = np.random.permutation(state.shape[0])
        if max_traj_num is not None:
            rnd_idx = rnd_idx[:max_traj_num]
        state, priv_state, last_state, last_priv_state, last_action, mask, reward_input, embedding_noise, rnn_start, traj_valid_indicators, traj_len_array = map(
            lambda x: x[rnd_idx], [
                state, priv_state, last_state, last_priv_state, last_action, mask, reward_input, embedding_noise,
                rnn_start, traj_valid_indicators, traj_len_array
            ]
        )
        with torch.no_grad():
            state, priv_state, last_state, last_priv_state, last_action, mask, reward_input, embedding_noise, rnn_start, traj_valid_indicators = map(
                lambda x: n2t(x, cpu_device), [
                    state, priv_state, last_state, last_priv_state, last_action, mask, reward_input, embedding_noise,
                    rnn_start, traj_valid_indicators
                ])
            total_rnn_start = rnn_start.clone()
            total_valid_indicators = traj_valid_indicators.clone()
            total_valid_indicators[torch.where(torch.diff(traj_valid_indicators, dim=-2) == 1)] = 1
            total_rnn_start[torch.where(torch.diff(total_rnn_start, dim=-2) == -1)] = 0

            # valid_num = mask.sum().item()
            # set RNN termination for LRU
            if self.parameter.randomize_first_hidden:
                policy_hidden = self.policy.make_rnd_init_state(state.shape[0], device=self.device)
            else:
                policy_hidden = self.policy.make_init_state(state.shape[0], device=self.device)
            with torch.no_grad():
                attention_mask = torch.from_numpy(traj_len_array).to(torch.get_default_dtype()).to(self.device)
                attention_mask = torch.cat((attention_mask, torch.zeros(
                    (attention_mask.shape[0], state.shape[-2] - attention_mask.shape[1]), device=self.device)), dim=-1)
                attention_mask = attention_mask.clone()
                attention_mask = attention_mask.to(torch.int)

            policy_hidden.set_rnn_start(rnn_start)
            policy_hidden.set_mask(traj_valid_indicators)
            policy_hidden.set_attention_concat_mask(attention_mask)

            target_encoder_hidden = self.policy.target_encoder.make_init_state(state.shape[0],
                                                                               device=device)
            target_encoder_hidden.copy_attachment_from_(policy_hidden)
        # embedding_noise_mask = (embedding_noise.abs().sum(dim=-1, keepdim=True) > 0).float()
        # mask = mask * embedding_noise_mask
        return state, priv_state, last_priv_state, last_action, embedding_noise, mask, reward_input, policy_hidden, target_encoder_hidden


    def sample_offpolicy_batch_on_policy(self, batch_size, device, max_traj_num=None):
        batch_data, batch_size, traj_valid_indicators, traj_len_array = self.replay_buffer.sample_trajs(
            batch_size,
            None, randomize_mask=self.parameter.randomize_mask,
            valid_number_post_randomized=self.parameter.valid_number_post_randomized,
            equalize_data_of_each_traj=True,
            get_all=False,
            random_trunc_traj=self.parameter.random_trunc_traj,
            nest_stack_trajs=self.allow_nest_stack)

        state, priv_state, last_state, last_priv_state, last_action, mask, reward_input, embedding_noise, rnn_start, target_flag = map(
            lambda x: getattr(batch_data, x), [
                'state', 'priv_state', 'last_state', 'last_priv_state', 'last_action', 'mask', 'reward_input', 'embedding_noise',
                'start', 'target_flag'
            ])

        cpu_device = device

        if self.parameter.image_input:
            state = n2t(state, cpu_device)
            state = self.uncompress_state(state)
            state = t2n(state)
        rnd_idx = np.random.permutation(state.shape[0])
        if max_traj_num is not None:
            rnd_idx = rnd_idx[:max_traj_num]
        state, priv_state, last_state, last_priv_state, last_action, mask, reward_input, embedding_noise, target_flag, rnn_start, traj_valid_indicators, traj_len_array = map(
            lambda x: x[rnd_idx], [
                state, priv_state, last_state, last_priv_state, last_action, mask, reward_input, embedding_noise, target_flag,
                rnn_start, traj_valid_indicators, traj_len_array
            ]
        )
        with torch.no_grad():
            state, priv_state, last_state, last_priv_state, last_action, mask, reward_input, embedding_noise, target_flag, rnn_start, traj_valid_indicators = map(
                lambda x: n2t(x, cpu_device), [
                    state, priv_state, last_state, last_priv_state, last_action, mask, reward_input, embedding_noise, target_flag,
                    rnn_start, traj_valid_indicators
                ])
            total_rnn_start = rnn_start.clone()
            total_valid_indicators = traj_valid_indicators.clone()
            total_valid_indicators[torch.where(torch.diff(traj_valid_indicators, dim=-2) == 1)] = 1
            total_rnn_start[torch.where(torch.diff(total_rnn_start, dim=-2) == -1)] = 0

            # valid_num = mask.sum().item()
            # set RNN termination for LRU
            if self.parameter.randomize_first_hidden:
                policy_hidden = self.policy.make_rnd_init_state(state.shape[0], device=self.device)
            else:
                policy_hidden = self.policy.make_init_state(state.shape[0], device=self.device)
            with torch.no_grad():
                attention_mask = torch.from_numpy(traj_len_array).to(torch.get_default_dtype()).to(self.device)
                attention_mask = torch.cat((attention_mask, torch.zeros(
                    (attention_mask.shape[0], state.shape[-2] - attention_mask.shape[1]), device=self.device)), dim=-1)
                attention_mask = attention_mask.clone()
                attention_mask = attention_mask.to(torch.int)

            policy_hidden.set_rnn_start(rnn_start)
            policy_hidden.set_mask(traj_valid_indicators)
            policy_hidden.set_attention_concat_mask(attention_mask)

            target_encoder_hidden = self.policy.target_encoder.make_init_state(state.shape[0],
                                                                               device=device)
            target_encoder_hidden.copy_attachment_from_(policy_hidden)
        # embedding_noise_mask = (embedding_noise.abs().sum(dim=-1, keepdim=True) > 0).float()
        # mask = mask * embedding_noise_mask
        mask = mask * target_flag
        return state, priv_state, last_priv_state, last_action, embedding_noise, mask, reward_input, policy_hidden, target_encoder_hidden

    def train_one_batch(self) -> Dict:
        rl_train_threshold = 10000
        batch_data, batch_size, traj_valid_indicators, traj_len_array = self.replay_buffer.sample_trajs(
            self.parameter.sac_batch_size,
            None, randomize_mask=self.parameter.randomize_mask,
            valid_number_post_randomized=self.parameter.valid_number_post_randomized,
            equalize_data_of_each_traj=True,
            get_all=True,
            random_trunc_traj=self.parameter.random_trunc_traj,
            nest_stack_trajs=self.allow_nest_stack)
        self.timer.register_end(level=2)

        # ('state', 'last_action', 'action', 'next_state', 'reward', 'logp', 'mask', 'done')
        max_trajectory_nums = 1300 # 800 for b4_mamba
        traj_id = np.random.permutation(batch_data.state.shape[0])[:max_trajectory_nums]
        (state, priv_state, last_state, last_priv_state, action, last_action, next_state, next_priv_state, done, mask, target_flag, reward,
         reward_input, timeout, rnn_start, logp_old, target_embedding, embedding_noise, target_logstd, next_target_logstd) = map(
            lambda x: getattr(batch_data, x)[traj_id, :], [
                'state', 'priv_state', 'last_state', 'last_priv_state', 'action', 'last_action',
                'next_state', 'next_priv_state', 'done', 'mask', 'target_flag', 'reward', 'reward_input',
                'timeout', 'start', 'logp', 'target_embedding', 'embedding_noise', 'target_logstd', 'next_target_logstd'
            ])
        traj_valid_indicators = traj_valid_indicators[traj_id, :]
        cpu_device = self.device

        with torch.no_grad():
            state, priv_state, last_state, last_priv_state, action, last_action, next_state, next_priv_state, done, mask, target_flag, reward, reward_input, timeout, target_embedding, embedding_noise, target_logstd, next_target_logstd, rnn_start, logp_old, traj_valid_indicators = map(
                lambda x: n2t(x, cpu_device), [
                    state, priv_state, last_state, last_priv_state, action, last_action, next_state, next_priv_state, done, mask, target_flag, reward, reward_input, timeout, target_embedding, embedding_noise, target_logstd, next_target_logstd,
                    rnn_start, logp_old, traj_valid_indicators
                ])
            total_rnn_start = rnn_start.clone()
            total_valid_indicators = traj_valid_indicators.clone()
            total_valid_indicators[torch.where(torch.diff(traj_valid_indicators, dim=-2) == 1)] = 1
            total_rnn_start[torch.where(torch.diff(total_rnn_start, dim=-2) == -1)] = 0
            done[timeout > 0] = 0

            # valid_num = mask.sum().item()
            # set RNN termination for LRU
            if self.parameter.randomize_first_hidden:
                policy_hidden = self.policy.make_rnd_init_state(state.shape[0], device=self.device)
                target_policy_hidden = policy_hidden
                target_hiddens = [target_value.make_rnd_init_state(state.shape[0], device=self.device) for target_value in
                                  self.target_values]
                value_hiddens = [value.make_rnd_init_state(state.shape[0], device=self.device) for value in self.values]
            else:
                policy_hidden = self.policy.make_init_state(state.shape[0], device=self.device)
                target_policy_hidden = self.policy.make_init_state(state.shape[0], device=self.device)
                target_hiddens = [target_value.make_init_state(state.shape[0], device=self.device) for target_value in
                                  self.target_values]
                value_hiddens = [value.make_init_state(state.shape[0], device=self.device) for value in self.values]
            with torch.no_grad():
                attention_mask = torch.from_numpy(traj_len_array).to(torch.get_default_dtype()).to(self.device)
                attention_mask = torch.cat((attention_mask, torch.zeros(
                    (attention_mask.shape[0], state.shape[-2] - attention_mask.shape[1]), device=self.device)), dim=-1)
                target_attention_mask = torch.cat((attention_mask[..., 1:], torch.zeros(
                    (attention_mask.shape[0], 1), device=self.device)), dim=-1)
                attention_mask = attention_mask.clone()
                attention_mask = attention_mask.to(torch.int)
                target_attention_mask = target_attention_mask.to(torch.int)

            target_policy_hidden.set_rnn_start(total_rnn_start)
            target_policy_hidden.set_mask(total_valid_indicators)
            target_policy_hidden.set_attention_concat_mask(target_attention_mask)
            for item in value_hiddens:
                item.set_rnn_start(rnn_start)
                item.set_attention_concat_mask(attention_mask)
                item.set_mask(traj_valid_indicators)
            for item in target_hiddens:
                item.set_rnn_start(total_rnn_start)
                item.set_attention_concat_mask(target_attention_mask)
                item.set_mask(total_valid_indicators)
            policy_hidden.set_rnn_start(rnn_start)
            policy_hidden.set_mask(traj_valid_indicators)
            policy_hidden.set_attention_concat_mask(attention_mask)
            self.policy.eval()

            next_state_value_list, state_value_list, value_embedding_list, policy_embedding_list = [], [], [], []
            batch_size_chunk = 100
            for i in range(int(np.ceil(next_state.shape[0] / batch_size_chunk))):
                end_idx = min((i + 1) * batch_size_chunk, next_state.shape[0])
                start_idx = i * batch_size_chunk
                indices = np.arange(start_idx, end_idx)
                target_hidden_tmp = target_hiddens[0].hidden_state_sample(indices)
                hidden_tmp = value_hiddens[0].hidden_state_sample(indices)
                policy_hidden_tmp = policy_hidden.hidden_state_sample(indices)

                next_state_value_list.append(
                    self.values[0].forward(next_priv_state[start_idx: end_idx].to(self.device),
                                           priv_state[start_idx: end_idx].to(self.device),
                                           action[start_idx: end_idx].to(self.device), None,
                                           target_hidden_tmp,
                                           reward[start_idx: end_idx].to(self.device),
                                           next_target_logstd[start_idx: end_idx].to(self.device)
                                           )[0],

                )
                state_value, value_embedding_output = self.values[0].forward(priv_state[start_idx: end_idx].to(self.device),
                                                                             last_priv_state[start_idx: end_idx].to(self.device),
                                                                             last_action[start_idx: end_idx].to(self.device),
                                                                             None,
                                                                             hidden_tmp,
                                                                             reward_input[start_idx: end_idx].to(self.device),
                                                                             target_logstd[start_idx: end_idx].to(self.device))[:2]
                state_value_list.append(
                    state_value
                )
                value_embedding_list.append(value_embedding_output)
                if self.parameter.directly_train_target:
                    policy_embedding_list.append(torch.zeros_like(value_embedding_output))
                else:
                    _, _, policy_embedding_output, _, _ = self.policy.get_mean_std(priv_state[start_idx: end_idx].to(self.device),
                                             last_priv_state[start_idx: end_idx].to(self.device),
                                             last_action[start_idx: end_idx].to(self.device), policy_hidden_tmp,
                                                                                   reward_input[start_idx: end_idx].to(self.device), True,
                                                                                   embedding_noise[start_idx: end_idx].to(self.device),
                                                                                   target_logstd[start_idx: end_idx].to(self.device), target_flag=target_flag[start_idx: end_idx].to(self.device))
                    policy_embedding_list.append(policy_embedding_output)
            next_state_value = torch.cat(next_state_value_list, dim=0)
            state_value = torch.cat(state_value_list, dim=0)
            value_embedding = torch.cat(value_embedding_list, dim=0).to(cpu_device)
            policy_embedding = torch.cat(policy_embedding_list, dim=0).to(cpu_device)
            # next_state_value = self.values[0].forward(next_state, state, action, None, target_hiddens[0], reward)[0]
            # state_value = self.values[0].forward(state, last_state, last_action, None, value_hiddens[0], reward_input)[0]
            # print(f'state shape: {state.shape}, action shape: {action.shape}, last_action shape: {last_action.shape}, '
            #       f'next_state shape: {next_state_value.shape}, reward shape: {reward_input.shape},state_value shape: {state_value.shape}, done shape: {done.shape}, mask shape: {mask.shape}')
            if self.iter_train <= rl_train_threshold:
                mask_adv = mask * (1 - target_flag)
            else:
                mask_adv = mask

            advantages, returns, adv_mean, adv_std = estimate_advantages(reward.to(self.device), (1 - done).to(self.device), state_value.to(self.device), mask_adv.to(self.device), next_state_value.to(self.device), self.parameter.gamma, 0.97)
            self.logger.add_tabular_data(tb_prefix='train', adv_mean=adv_mean.item(), adv_std=adv_std.item())
            advantages_mask = advantages * mask_adv
            positive_adv_num = (advantages_mask > 0).sum()
            negative_adv_num = (advantages_mask < 0).sum()
            self.logger.add_tabular_data(tb_prefix='train', positive_adv_num=positive_adv_num.item(), negative_adv_num=negative_adv_num.item(), adv_positive_ratio=(positive_adv_num / (negative_adv_num + positive_adv_num)).item())

            # logp_old_2 = self.policy.logp(state, action, last_state, last_action, policy_hidden, reward_input)
            # self.logger.add_tabular_data(tb_prefix='train', full_kl=(((logp_old - logp_old_2) * mask).sum() / mask.sum()).item())
            # logp_old = logp_old_2
        # print(f'state shape: {state.shape}, action shape: {action.shape}, last_action shape: {last_action.shape}, '
        #       f'next_state shape: {next_state_value.shape}, advantages shape: {advantages.shape}, returns shape: {returns.shape}, logp_old: {logp_old.shape}')
        self.logger(f'training state shape: {state.shape}, device: {state.device}')
        torch.cuda.empty_cache()
        epoch_num = 15
        # target_network_additional_epoch_num = min(85, self.iter_train * 3) #  45 if self.parameter.image_input else 30
        # if (self.iter_train + 1) % 20 == 0:
        #     target_network_additional_epoch_num *= 20
        if self.iter_train < self.threshold_iter or self.parameter.directly_train_target:
            target_network_additional_epoch_num = 0
        else:
            target_network_additional_epoch_num = 30
        target_encoder_batch_size = 4 if self.parameter.image_input else 25
        if self.parameter.directly_train_target:
            target_network_additional_epoch_num = 0
        self.optimizer_target_encoder.zero_grad()
        self.target_encoder_optimization_cnt = 0
        for epoch in tqdm(range(epoch_num + target_network_additional_epoch_num)):
            # 打乱数据集
            permutation = torch.randperm(state.shape[0])
            # 计算batch_size的大小
            # batch_size = int(np.ceil(state.shape[0] / 10)) if self.parameter.image_input else int(np.ceil(state.shape[0] / 10))
            batch_size = int(np.ceil(state.shape[0] / 10))
            if self.parameter.directly_train_target:
                batch_size = int(np.ceil(state.shape[0] / 20))
            # 打印一次日志
            # print(
            #     f'dataset size {dataset_size}, mini_batch_num: {self.parameter.ppo_minibatch_num}, batch_size: {batch_size}')
            # 将数据分为多个mini_batch，分批次经过神经网络
            if not self.parameter.directly_train_target and False:
                state_offpi_full, priv_state_offpi_full, last_priv_state_offpi_full, last_action_offpi_full, embedding_noise_offpi_full, mask_offpi_full, reward_input_offpi_full, policy_hidden_offpi_full, target_encoder_hidden_offpi_full = self.sample_offpolicy_batch_old(
                    80000, self.device,
                    max_traj_num=target_encoder_batch_size * 10 if self.parameter.image_input else None,
                    uncompress_state=True,
                )
                permutation_offpolicy = torch.randperm(state_offpi_full.shape[0])
                offpolicy_batch_size = target_encoder_batch_size if self.parameter.image_input else (state_offpi_full.shape[0])//10
            for i in range(0, state.shape[0], batch_size):
                # 取出数据下标
                with (torch.no_grad()):
                    indices = permutation[i:i + batch_size]

                    value_hidden_batch = value_hiddens[0].hidden_state_sample(indices)
                    policy_hidden_batch = policy_hidden.hidden_state_sample(indices)

                    # 将数据取出
                    batch_obs, batch_priv_obs, batch_act, batch_last_act, batch_last_obs, batch_last_priv_obs, batch_adv, batch_mask, batch_target_flag,\
                    batch_reward_input, batch_old_logp, batch_ret, batch_policy_embedding, batch_value_embedding, batch_target_embedding, batch_embedding_noise, batch_target_logstd = state[indices], priv_state[indices], \
                        action[indices], last_action[indices], last_state[indices], last_priv_state[indices], \
                        advantages[indices], mask[indices], target_flag[indices], reward_input[indices], logp_old[indices], returns[indices], policy_embedding[indices], value_embedding[indices], target_embedding[indices], embedding_noise[indices], target_logstd[indices]
                    batch_obs, batch_priv_obs, batch_act, batch_last_act, batch_last_obs, batch_last_priv_obs, batch_adv, batch_mask, batch_target_flag, \
                        batch_reward_input, batch_old_logp, batch_ret, batch_policy_embedding, batch_value_embedding, batch_target_embedding, batch_embedding_noise, batch_target_logstd = map(lambda x: x.to(self.device),
                                                                                                                           [batch_obs, batch_priv_obs, batch_act, batch_last_act, batch_last_obs, batch_last_priv_obs, batch_adv, batch_mask, batch_target_flag,\
                    batch_reward_input, batch_old_logp, batch_ret, batch_policy_embedding, batch_value_embedding, batch_target_embedding, batch_embedding_noise, batch_target_logstd])
                    if self.parameter.directly_train_target or (epoch == 0 and i == 0 and not self.parameter.image_input):
                        batch_obs, batch_last_obs = map(lambda x: self.uncompress_state(x), [
                            batch_obs, batch_last_obs
                        ])
                        # tmp_target_batch_size = min(target_encoder_batch_size, batch_obs.shape[0])
                        # target_encoder_hidden = self.policy.target_encoder.make_init_state(tmp_target_batch_size, device=batch_obs.device)
                        # target_encoder_hidden.copy_attachment_from_(policy_hidden_batch)
                        # target_encoder_hidden = target_encoder_hidden.hidden_state_sample(torch.arange(tmp_target_batch_size, device=batch_obs.device))
                        # batch_obs, batch_last_obs = map(lambda x: self.uncompress_state(x), [
                        #     batch_obs[:tmp_target_batch_size], batch_last_obs[:tmp_target_batch_size]
                        # ])

                # 进行一步PPO训练
                train_target_only = False
                if self.parameter.iterative_training:
                    if self.iter_train % 2 == 1:
                        train_target_only = True
                if epoch >= epoch_num:
                    train_target_only = True
                if self.parameter.directly_train_target:
                    logs = self._ppo_batch_train_direct_target(batch_obs, batch_priv_obs, batch_last_obs, batch_last_priv_obs, batch_last_act, batch_act, batch_adv, batch_reward_input, batch_ret, batch_old_logp,
                                         batch_mask, policy_hidden_batch, value_hidden_batch,
                                             batch_policy_embedding, batch_value_embedding, batch_target_embedding, None, batch_embedding_noise, batch_target_logstd,
                                             None, None, None, None, None, None, train_target_only=train_target_only)
                else:
                    if self.iter_train <= rl_train_threshold:
                        batch_mask = batch_mask * (1 - batch_target_flag)
                    if False and offpolicy_indices.shape[0] > 1:
                        offpolicy_indices = permutation_offpolicy[(i // batch_size) * offpolicy_batch_size:((i // batch_size) + 1) * offpolicy_batch_size]
                        state_offpi, priv_state_offpi, last_priv_state_offpi, last_action_offpi, embedding_noise_offpi, mask_offpi, reward_input_offpi = map(lambda x: x[offpolicy_indices],[
                            state_offpi_full, priv_state_offpi_full, last_priv_state_offpi_full, last_action_offpi_full, embedding_noise_offpi_full,
                            mask_offpi_full, reward_input_offpi_full
                        ])
                        # policy_hidden_offpi = policy_hidden_offpi_full.hidden_state_sample(offpolicy_indices)
                        target_encoder_hidden_offpi = target_encoder_hidden_offpi_full.hidden_state_sample(offpolicy_indices)
                    else:
                        # self.logger(f'additionally sample batch..')
                        state_offpi, priv_state_offpi, last_priv_state_offpi, last_action_offpi, embedding_noise_offpi, mask_offpi, reward_input_offpi, policy_hidden_offpi, target_encoder_hidden_offpi = self.sample_offpolicy_batch_old(
                            8000 if self.parameter.image_input else 8000, self.device,
                            max_traj_num=target_encoder_batch_size if self.parameter.image_input else None,
                            uncompress_state=True,
                        )
                    try:
                        if batch_mask.sum() == 0:
                            logs = {}
                        else:
                            logs = self._ppo_batch_train(batch_obs, batch_priv_obs, batch_last_obs, batch_last_priv_obs, batch_last_act, batch_act, batch_adv, batch_reward_input, batch_ret, batch_old_logp,
                                                     batch_mask, policy_hidden_batch, value_hidden_batch,
                                                         batch_policy_embedding, batch_value_embedding, batch_target_embedding, batch_target_flag, None, batch_embedding_noise, batch_target_logstd,
                                                         priv_state_offpi, state_offpi, last_action_offpi, target_encoder_hidden_offpi, embedding_noise_offpi, mask_offpi, train_target_only=train_target_only)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        logs = {}
                self.logger.add_tabular_data(tb_prefix='train', **logs)
                if epoch == 0 and i == 0 and 'kl' in logs:
                    self.logger.add_tabular_data(tb_prefix='train', start_kl=logs['kl'])
        self.optimizer_target_encoder.zero_grad()
        return {}
