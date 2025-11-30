from typing import Tuple

import numpy as np
import torch


def estimate_advantages(rewards: torch.Tensor, not_done: torch.Tensor,
                        values: torch.Tensor, mask: torch.Tensor, next_values:torch.Tensor,
                        gamma: float, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算GAE
    Args:
        rewards: 奖励
        not_done: 非done
        values: 状态值
        mask: 数据有效
        gamma: 衰减系数
        lam: GAE衰减系数

    Returns:
        advantages: GAE
        returns: 带gamma衰减的return
    """
    # reward: (L, 1)
    deltas = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    prev_advantage = 0
    prev_return = 0
    for i in reversed(range(rewards.shape[-2])):
        deltas[..., i, :] = (rewards[..., i, :] + gamma * next_values[..., i, :] * not_done[..., i, :] - values[..., i, :]) * mask[
                                                                                                                  ...,
                                                                                                                  i, :]
        advantages[..., i, :] = (deltas[..., i, :] + gamma * lam * prev_advantage * not_done[..., i, :]) * mask[..., i,
                                                                                                           :]
        returns[..., i, :] = (rewards[..., i, :] + gamma * not_done[..., i, :] * prev_return) * mask[..., i, :]
        prev_advantage = advantages[..., i, :] * mask[..., i, :]
        prev_return = returns[..., i, :] * mask[..., i, :]
    # returns = values + torch.clamp(advantages, -5, 5)
    adv_sum = (advantages * mask).sum()
    valid_sum = mask.sum()
    adv_mean = adv_sum / valid_sum
    adv_std_sum = ((advantages - adv_mean).pow(2) * mask).sum()
    adv_std = adv_std_sum / valid_sum
    adv_std = torch.sqrt(adv_std)
    advantages = advantages / (adv_std + 1e-9)
    advantages = torch.clamp(advantages, -5, 5)
    return advantages, returns, adv_mean, adv_std
