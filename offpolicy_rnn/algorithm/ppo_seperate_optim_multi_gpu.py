from ..utility.sample_utility import unorm_act, norm_act, n2t, n2t_2dim, t2n, eval_inprocess, policy_eval
from typing import List, Union, Tuple, Dict, Optional, Callable
from tqdm import tqdm
import numpy as np
from legged_gym.utils import webviewer
import torch
from ..policy_value_models.make_models import make_policy_model, make_value_model
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
def prepare_param_list(model, rnn_lr, l2_norm):
    param_list = []
    for k, v in model.contextual_modules.items():
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


class PPO_SEPERATE_OPTIM_MULTI_GPU(PPO):
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

        policy_param_list = prepare_param_list(self.policy, self.parameter.rnn_policy_lr, self.parameter.policy_l2_norm)
        self.optimizer_policy = self.optim_class(policy_param_list,
                                                 lr=self.parameter.policy_lr,
                                                 weight_decay=self.parameter.policy_l2_norm)
        value_param_list = prepare_param_list(self.values[0], self.parameter.rnn_value_lr, self.parameter.value_l2_norm)

        self.optimizer_value = self.optim_class(value_param_list, lr=self.parameter.value_lr,
                                                weight_decay=self.parameter.value_l2_norm)
        # self.policy = make_policy_model(self.policy_args, self.base_algorithm, self.discrete_env)
        # self.values: List[ContextualPPOValue] = [
        #     make_value_model(self.value_args, self.base_algorithm, self.discrete_env) for _ in
        #     range(self.parameter.value_net_num)]
        self.total_gpus = total_gpus = torch.cuda.device_count()

        self.training_policies = [self.policy] + [
            make_policy_model(self.policy_args, self.base_algorithm, self.discrete_env) for _ in range(total_gpus - 1)
        ]
        self.training_values = [self.values[0]] + [
            make_value_model(self.value_args, self.base_algorithm, self.discrete_env) for _ in range(total_gpus - 1)
        ]
        self.training_devices = [self.device]
        for i in range(self.total_gpus):
            device = torch.device(f'cuda:{i}')
            if not device == self.training_devices[0]:
                self.training_devices.append(device)

        self.training_policy_parameters = [policy_param_list] + [prepare_param_list(policy, self.parameter.rnn_policy_lr, self.parameter.policy_l2_norm) for policy in self.training_policies[1:]]
        self.training_policy_optimizers = [self.optimizer_policy] + [
            self.optim_class(policy_param,
                             lr=self.parameter.policy_lr,
                             weight_decay=self.parameter.policy_l2_norm) for policy_param in self.training_policy_parameters[1:]
        ]
        self.training_value_parameters = [value_param_list] + [prepare_param_list(value, self.parameter.rnn_value_lr, self.parameter.value_l2_norm) for value in self.training_values[1:]]

        self.training_value_optimizers = [self.optimizer_value] + [
            self.optim_class(value_param,
                             lr=self.parameter.value_lr,
                             weight_decay=self.parameter.value_l2_norm) for value_param in self.training_value_parameters[1:]
        ]
        for device, policy, value in zip(self.training_devices, self.training_policies, self.training_values):
            policy.to(device)
            value.to(device)

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

    def _ppo_batch_train(self, state: torch.Tensor, priv_state: torch.Tensor, lst_state: torch.Tensor, lst_priv_state: torch.Tensor, lst_action: torch.Tensor, action: torch.Tensor,
                     adv: torch.Tensor, reward_input: torch.Tensor, rets: torch.Tensor, old_logp: torch.Tensor, mask: torch.Tensor,
                         policy_hidden, value_hidden, old_policy_embedding, old_value_embedding):
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
        value_ebd_factor = 0.0
        policy_ebd_factor = 0.0
        valid_nums = [item.sum().detach() for item in mask]
        batch_nums = [item.shape[0] if np.prod(item.shape) > 0 else 0 for item in state]
        sum_weights = sum(batch_nums)
        weights = [item / sum_weights for item in batch_nums]
        def all_reduce_1(data):
            if data[0].grad is not None:
                data[0].grad[:] *= weights[0]
            else:
                data[0].grad = torch.zeros_like(data[0])
            for i in range(1, len(data)):
                data[0].grad[:] += data[i].grad.to(data[0].device) * weights[i]
        def all_reduce_2(data):
            for i in range(1, len(data)):
                data[i].grad[:] = data[0].grad.to(data[i].device)

        # 值函数计算
        for value_optim, value_net, state_item, lst_state_item, lst_action_item, value_hidden_item, reward_input_item, old_value_embedding_item, valid_num, mask_item, rets_item in zip(
            self.training_value_optimizers, self.training_values, priv_state, lst_priv_state, lst_action, value_hidden, reward_input, old_value_embedding, valid_nums, mask, rets
        ):
            if np.prod(state_item.shape) == 0:
                continue
            value, value_embedding = value_net.forward(state_item, lst_state_item, lst_action_item, None, value_hidden_item, reward_input_item)[:2]
            # 值函数损失函数
            value_embedding_loss = torch.clamp_min(((value_embedding - old_value_embedding_item) * mask_item).pow(2), min=0.01 ** 2).mean(dim=-1).sum() / valid_num
            value_loss = ((value - rets_item) * mask_item).pow(2).sum() / valid_num + value_ebd_factor * value_embedding_loss

            # 值函数网络权重更新
            value_optim.zero_grad()
            value_loss.backward()
            # 梯度裁剪
        with torch.no_grad():
            for params in zip(*[net.parameters() for net in self.training_values]):
                all_reduce_1(params)
            value_norm = torch.nn.utils.clip_grad_norm_(self.values[0].parameters(), 40.)
            for params in zip(*[net.parameters() for net in self.training_values]):
                all_reduce_2(params)
        for value_optim in self.training_value_optimizers:
            value_optim.step()
        value_loss = value_loss.item()
        value_embedding_loss = value_embedding_loss.item()
        for policy_optim, policy_net, state_item, action_item, lst_state_item, lst_action_item, policy_hidden_item, reward_input_item, old_policy_embedding_item, valid_num, mask_item, rets_item, old_logp_item, adv_item in zip(
            self.training_policy_optimizers, self.training_policies, state, action, lst_state, lst_action, policy_hidden, reward_input, old_policy_embedding, valid_nums, mask, rets, old_logp, adv
        ):
            if np.prod(state_item.shape) == 0:
                continue
            # action在当前policy下被执行的概率
            log_probs, policy_embedding = policy_net.logp(state_item, action_item, lst_state_item, lst_action_item, policy_hidden_item, reward_input_item, require_embedding=True)
            # 新旧策略的KL散度
            kl = old_logp_item - log_probs

            # exp(-KL)
            ratio = torch.exp(-kl)
            # KL散度均值，用于监控新旧策略之间的差异
            kl_mean = (kl * mask_item).sum() / valid_num
            # PPO actor损失计算
            surr1 = ratio * adv_item
            # TODO: epsilon
            surr2 = torch.clamp(ratio, 1.0 - 0.15, 1.0 + 0.15) * adv_item
            policy_embedding_loss = torch.clamp_min(((policy_embedding - old_policy_embedding_item) * mask_item).pow(2), min=0.01 ** 2).mean(dim=-1).sum() / valid_num
            policy_loss = -(torch.min(surr1, surr2) * mask_item).sum() / valid_num + policy_ebd_factor * policy_embedding_loss

            # 策略熵计算
            # policy_std = self.policy.logstd.exp().mean()
            if self.parameter.std_learnable and self.parameter.entropy_coeff > 0:
                entropy = self.policy.entropy()
                policy_loss = policy_loss - entropy * self.parameter.entropy_coeff
                entropy = entropy.item()
            else:
                entropy = self.policy.entropy()
            # 策略网络权重更新
            policy_optim.zero_grad()
            policy_loss.backward()
        with torch.no_grad():
            for params in zip(*[net.parameters() for net in self.training_policies]):
                all_reduce_1(params)
            policy_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40.)
            for params in zip(*[net.parameters() for net in self.training_policies]):
                all_reduce_2(params)
        for policy_optim in self.training_policy_optimizers:
            policy_optim.step()
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
            policy_embedding_loss=policy_embedding_loss.item(),
            value_embedding_loss=value_embedding_loss,
        )


    def train_one_batch(self) -> Dict:
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
        max_trajectory_nums = 1500
        traj_id = np.random.permutation(batch_data.state.shape[0])[:max_trajectory_nums]
        state, priv_state, last_state, last_priv_state, action, last_action, next_state, next_priv_state, done, mask, reward, reward_input, timeout, rnn_start, logp_old = map(
            lambda x: getattr(batch_data, x)[traj_id, :], [
                'state', 'priv_state', 'last_state', 'last_priv_state', 'action', 'last_action', 'next_state', 'next_priv_state', 'done', 'mask', 'reward', 'reward_input',
                'timeout', 'start', 'logp'
            ])
        traj_valid_indicators = traj_valid_indicators[traj_id, :]
        # self.logger(state.shape, action.shape, last_action.shape, next_state.shape, done.shape, mask.shape, reward.shape)
        # (1, 1000, 17) (1, 1000, 6) (1, 1000, 6) (1, 1000, 17) (1, 1000, 1) (1, 1000, 1) (1, 1000, 1)
        # 2. update networks
        # print(state.shape, last_state.shape, action.shape, next_state.shape, done.shape, reward.shape, logp_old.shape, mask.shape, reward_input.shape, rnn_start.shape, logp_old.shape)
        cpu_device = self.device
        cpu_device = torch.device('cpu') if self.parameter.image_input else self.device

        with torch.no_grad():
            for value, policy in zip(self.training_values[1:], self.training_policies[1:]):
                value.load_state_dict(self.training_values[0].state_dict())
                policy.load_state_dict(self.training_policies[0].state_dict())
            device = self.training_devices[0]
            policy_network = self.training_policies[0]
            value_network = self.training_values[0]

            state, priv_state, last_state, last_priv_state, action, last_action, next_state, next_priv_state, done, mask, reward, reward_input, timeout, rnn_start, logp_old, traj_valid_indicators = map(
                lambda x: n2t(x, cpu_device), [
                    state, priv_state, last_state, last_priv_state, action, last_action, next_state, next_priv_state, done, mask, reward, reward_input, timeout,
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
                policy_hidden = policy_network.make_rnd_init_state(state.shape[0], device=device)
                target_policy_hidden = policy_hidden
                target_hiddens = [value_network.make_rnd_init_state(state.shape[0], device=device)]
                value_hiddens = [value_network.make_rnd_init_state(state.shape[0], device=device)]
            else:
                policy_hidden = policy_network.make_init_state(state.shape[0], device=device)
                target_policy_hidden = policy_network.make_init_state(state.shape[0], device=device)
                target_hiddens = [value_network.make_init_state(state.shape[0], device=device)]
                value_hiddens = [value_network.make_init_state(state.shape[0], device=device)]
            with torch.no_grad():
                attention_mask = torch.from_numpy(traj_len_array).to(torch.get_default_dtype()).to(device)
                attention_mask = torch.cat((attention_mask, torch.zeros(
                    (attention_mask.shape[0], state.shape[-2] - attention_mask.shape[1]), device=device)), dim=-1)
                target_attention_mask = torch.cat((attention_mask[..., 1:], torch.zeros(
                    (attention_mask.shape[0], 1), device=device)), dim=-1)
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
            for policy in self.training_policies:
                policy.eval()

            next_state_value_list, state_value_list, value_embedding_list, policy_embedding_list = [], [], [], []
            batch_size_chunk = 5 if self.parameter.image_input else 100
            for i in range(int(np.ceil(next_state.shape[0] / batch_size_chunk))):
                end_idx = min((i + 1) * batch_size_chunk, next_state.shape[0])
                start_idx = i * batch_size_chunk
                indices = np.arange(start_idx, end_idx)
                target_hidden_tmp = target_hiddens[0].hidden_state_sample(indices)
                hidden_tmp = value_hiddens[0].hidden_state_sample(indices)
                policy_hidden_tmp = policy_hidden.hidden_state_sample(indices)

                next_state_value_list.append(
                    value_network.forward(next_priv_state[start_idx: end_idx].to(self.device),
                                           priv_state[start_idx: end_idx].to(self.device),
                                           action[start_idx: end_idx].to(self.device), None,
                                           target_hidden_tmp,
                                           reward[start_idx: end_idx].to(device))[0]
                )
                state_value, value_embedding_output = value_network.forward(priv_state[start_idx: end_idx].to(self.device),
                                                                             last_priv_state[start_idx: end_idx].to(self.device),
                                                                             last_action[start_idx: end_idx].to(device),
                                                                             None,
                                                                             hidden_tmp,
                                                                             reward_input[start_idx: end_idx].to(device))[:2]
                state_value_list.append(
                    state_value
                )
                value_embedding_list.append(value_embedding_output)
                _, _, policy_embedding_output, _, _ = policy_network.get_mean_std(self.uncompress_state(state[start_idx: end_idx].to(device)),
                                         self.uncompress_state(last_state[start_idx: end_idx].to(device)),
                                         last_action[start_idx: end_idx].to(device), policy_hidden_tmp, reward_input[start_idx: end_idx].to(device), True)
                policy_embedding_list.append(policy_embedding_output)
            next_state_value = torch.cat(next_state_value_list, dim=0)
            state_value = torch.cat(state_value_list, dim=0)
            value_embedding = torch.cat(value_embedding_list, dim=0).to(cpu_device)
            policy_embedding = torch.cat(policy_embedding_list, dim=0).to(cpu_device)
            # next_state_value = self.values[0].forward(next_state, state, action, None, target_hiddens[0], reward)[0]
            # state_value = self.values[0].forward(state, last_state, last_action, None, value_hiddens[0], reward_input)[0]
            # print(f'state shape: {state.shape}, action shape: {action.shape}, last_action shape: {last_action.shape}, '
            #       f'next_state shape: {next_state_value.shape}, reward shape: {reward_input.shape},state_value shape: {state_value.shape}, done shape: {done.shape}, mask shape: {mask.shape}')
            advantages, returns = estimate_advantages(reward.to(device), (1 - done).to(device), state_value.to(device), mask.to(device), next_state_value.to(device), self.parameter.gamma, 0.97)
            # logp_old_2 = self.policy.logp(state, action, last_state, last_action, policy_hidden, reward_input)
            # self.logger.add_tabular_data(tb_prefix='train', full_kl=(((logp_old - logp_old_2) * mask).sum() / mask.sum()).item())
            # logp_old = logp_old_2
        del target_policy_hidden, target_hiddens, state_value, next_state_value
        policy_hidden.to_device(cpu_device)
        value_hiddens[0].to_device(cpu_device)
        advantages = advantages.to(cpu_device)
        returns = returns.to(cpu_device)
        # print(f'state shape: {state.shape}, action shape: {action.shape}, last_action shape: {last_action.shape}, '
        #       f'next_state shape: {next_state_value.shape}, advantages shape: {advantages.shape}, returns shape: {returns.shape}, logp_old: {logp_old.shape}')
        self.logger(f'training state shape: {state.shape}, device: {state.device}')
        torch.cuda.empty_cache()
        epoch_num = 10 if self.parameter.image_input else 15

        for epoch in tqdm(range(epoch_num)):
            # 打乱数据集
            permutation = torch.randperm(state.shape[0])
            # 计算batch_size的大小
            batch_size = int(np.ceil(state.shape[0] / 10)) if self.parameter.image_input else int(np.ceil(state.shape[0] / 10))
            # 打印一次日志
            # print(
            #     f'dataset size {dataset_size}, mini_batch_num: {self.parameter.ppo_minibatch_num}, batch_size: {batch_size}')
            # 将数据分为多个mini_batch，分批次经过神经网络
            for i in range(0, state.shape[0], batch_size):
                # 取出数据下标
                with (torch.no_grad()):
                    batch_obs_list, batch_priv_obs_list, batch_last_obs_list, batch_last_priv_obs_list, batch_last_act_list, batch_act_list, batch_adv_list, batch_reward_input_list, batch_ret_list, batch_old_logp_list, \
                    batch_mask_list, policy_hidden_batch_list, value_hidden_batch_list, batch_policy_embedding_list, batch_value_embedding_list = [], [], [], [], [], [], [], [], [], [], [], [], [],[], []
                    indices = permutation[i:i + batch_size]

                    total_size = indices.shape[0]
                    each_size = int(np.ceil(total_size / self.total_gpus))
                    sizes = [total_size - each_size * (self.total_gpus - 1)] + [each_size] * (self.total_gpus - 1)
                    max_main_gpu_batchsize = 5
                    if sizes[0] > max_main_gpu_batchsize and self.total_gpus > 1:
                        additional_data = sizes[0] - max_main_gpu_batchsize
                        sizes[0] = max_main_gpu_batchsize
                        each_add = int(np.ceil(additional_data / (self.total_gpus - 1)))
                        for j in range(1, self.total_gpus):
                            sizes[j] += min(each_add, additional_data)
                            additional_data -= each_add
                            if additional_data <= 0:
                                break
                    idx = i
                    for size_idx, size in enumerate(sizes):
                        indices = permutation[idx: idx+size]
                        device = self.training_devices[size_idx]
                        idx += size
                        value_hidden_batch = value_hiddens[0].hidden_state_sample(indices)
                        policy_hidden_batch = policy_hidden.hidden_state_sample(indices)
                        value_hidden_batch.to_device(device)
                        policy_hidden_batch.to_device(device)

                        # 将数据取出
                        batch_obs, batch_priv_obs, batch_act, batch_last_act, batch_last_obs, batch_last_priv_obs, batch_adv, batch_mask, \
                            batch_reward_input, batch_old_logp, batch_ret, batch_policy_embedding, batch_value_embedding = \
                        state[indices], priv_state[indices], \
                            action[indices], last_action[indices], last_state[indices], last_priv_state[indices], \
                            advantages[indices], mask[indices], reward_input[indices], logp_old[indices], returns[
                            indices], policy_embedding[indices], value_embedding[indices]
                        batch_obs, batch_priv_obs, batch_act, batch_last_act, batch_last_obs, batch_last_priv_obs, batch_adv, batch_mask, \
                            batch_reward_input, batch_old_logp, batch_ret, batch_policy_embedding, batch_value_embedding = map(
                            lambda x: x.to(device),
                            [batch_obs, batch_priv_obs, batch_act, batch_last_act, batch_last_obs, batch_last_priv_obs,
                             batch_adv, batch_mask, \
                             batch_reward_input, batch_old_logp, batch_ret, batch_policy_embedding, batch_value_embedding])
                        batch_obs, batch_last_obs = map(lambda x: self.uncompress_state(x), [batch_obs, batch_last_obs])
                        batch_obs_list.append(batch_obs)
                        batch_last_act_list.append(batch_last_act)
                        batch_last_obs_list.append(batch_last_obs)
                        batch_priv_obs_list.append(batch_priv_obs)
                        batch_last_priv_obs_list.append(batch_last_priv_obs)
                        batch_act_list.append(batch_act)
                        batch_adv_list.append(batch_adv)
                        batch_reward_input_list.append(batch_reward_input)
                        batch_ret_list.append(batch_ret)
                        batch_old_logp_list.append(batch_old_logp)
                        batch_mask_list.append(batch_mask)
                        policy_hidden_batch_list.append(policy_hidden_batch)
                        value_hidden_batch_list.append(value_hidden_batch)
                        batch_policy_embedding_list.append(batch_policy_embedding)
                        batch_value_embedding_list.append(batch_value_embedding)

                # if any([np.prod(list(item.shape)) == 0 for item in batch_obs_list]):
                #     continue
                # 进行一步PPO训练
                try:
                    logs = self._ppo_batch_train(batch_obs_list, batch_priv_obs_list, batch_last_obs_list, batch_last_priv_obs_list, batch_last_act_list, batch_act_list, batch_adv_list, batch_reward_input_list, batch_ret_list, batch_old_logp_list,
                                             batch_mask_list, policy_hidden_batch_list, value_hidden_batch_list, batch_policy_embedding_list, batch_value_embedding_list)
                    self.logger.add_tabular_data(tb_prefix='train', **logs)
                    if epoch == 0 and i == 0:
                        self.logger.add_tabular_data(tb_prefix='train', start_kl=logs['kl'])
                except Exception as _:
                    import traceback
                    traceback.print_exc()
        return {}
