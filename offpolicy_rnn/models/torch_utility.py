import numpy as np
import torch

from .RNNHidden import RNNHidden
from .rnn_base import RNNBase
from typing import Optional
try:
    from offpolicy_rnn.models.smamba.mamba_ssm.ops.selective_scan_interface_new import selective_scan_fn
except Exception as e:
    selective_scan_fn = None
from einops import rearrange, repeat
import torch.nn.functional as F


def slice_tensor(x, slice_num):
    assert len(x.shape) == 3, 'slice operation should be added on 3-dim tensor'
    assert x.shape[1] % slice_num == 0, f'cannot reshape length with {x.shape[1]} to {slice_num} slices'
    s = x.shape
    x = x.reshape([s[0], s[1] // slice_num, slice_num, s[2]]).transpose(0, 1)
    return x


def merge_slice_tensor(data):
    s = data.shape
    data = data.reshape(s[0] * s[1], s[2], s[3])
    return data

def multi_batch_forward(network_input: torch.Tensor, network: RNNBase, hidden: Optional[RNNHidden], require_full_rnn_output: bool=False):
    # this function does not support require full hidden is True, which could result the full_rnn_output b

    assert len(network_input.shape) >= 3, f'at least a single batch data should be input'

    seq_len = network_input.shape[-2]
    dim = network_input.shape[-1]
    batch_info = list(network_input.shape[:-2])
    if len(network_input.shape) > 3:
        # does not match the input requirement of RNN
        network_input = network_input.reshape((-1, seq_len, dim))
    if hidden is not None:
        assert hidden.hidden_batch_size == network_input.shape[0]
    output, hidden, full_rnn_output = network.meta_forward(network_input, hidden, require_full_rnn_output)
    # only output is reshaped to original shape
    output = output.reshape(tuple([*batch_info, seq_len, output.shape[-1]]))
    return output, hidden, full_rnn_output


# def _forward_fix_length_onestep(self, embedding_input: torch.Tensor, uni_model_input: torch.Tensor, hidden):
#     assert len(embedding_input.shape) >= 3, '[traj_idx, time_idx, feature_idx] is required'
#     assert embedding_input.shape[-2] == 1, f'time step should be one!'
#     batch_size = embedding_input.shape[0]
#
#     hidden_length = RNNBase.get_hidden_length(hidden)
#     if hidden_length >= self.fixed_hist_length * batch_size:
#         hidden = RNNBase.pop_hidden_state(hidden, (hidden_length - self.fixed_hist_length * batch_size + batch_size) // batch_size * batch_size)
#     if hidden is None:
#         hidden = self.make_init_state(batch_size, embedding_input.device)
#     else:
#         hidden = RNNBase.append_hidden_state(hidden,
#                                         self.make_init_state(batch_size,
#                                                                                 embedding_input.device))
#     length = RNNBase.get_hidden_length(hidden) // batch_size
#     embedding_input = torch.cat([embedding_input] * length, dim=0)
#     uni_model_input = torch.cat([uni_model_input] * length, dim=0)
#     uni_model_output, hidden, embedding_output, full_memory = self.meta_forward(embedding_input,
#                                                                                 uni_model_input,
#                                                                                 hidden)
#     return uni_model_output, hidden, embedding_output, full_memory


def fixed_length_forward_one_step(network_input: torch.Tensor, network: RNNBase, hidden: Optional[RNNHidden],
                                  fixed_length: int):
    assert network.rnn_num > 0, f'rnn num must be larger than 0!!'
    network_input_shape = list(network_input.shape)
    data_batchsize = 1 if len(network_input_shape) == 2 else np.prod(network_input_shape[:-2])
    if len(network_input_shape) == 2:
        network_input = network_input.unsqueeze(0)
    assert network_input.shape[-2] == 1, f'only one step is acceptable!!'
    repeat_num = 1

    if hidden is not None:
        if hidden.hidden_batch_size >= data_batchsize * fixed_length:
            hidden.elementwise_pop(hidden.hidden_batch_size - data_batchsize * (fixed_length - 1))
        hidden_apd = network.make_init_state(data_batchsize, hidden.device)
        hidden.elementwise_append(hidden_apd)
        repeat_num = int(hidden.hidden_batch_size // data_batchsize)
    network_input = network_input.unsqueeze(0).repeat_interleave(repeat_num, dim=0)
    output, hidden, _ = multi_batch_forward(network_input, network, hidden, require_full_rnn_output=False)
    output = output[0]
    if len(network_input_shape) == 2:
        output = output.squeeze(0)
    return output, hidden


def fixed_length_forward(network_input: torch.Tensor, network: RNNBase, hidden: Optional[RNNHidden], fixed_length: int):
    assert network.rnn_num > 0, f'rnn num must be larger than 0!!'
    network_input_shape = list(network_input.shape)
    if len(network_input_shape) == 2:
        network_input = network_input.unsqueeze(0)
    seq_len = network_input_shape[-2]
    outputs = []
    for i in range(seq_len):
        output_i, hidden = fixed_length_forward_one_step(network_input[..., i:i+1,:], network, hidden, fixed_length)
        outputs.append(output_i)
    output = torch.cat(outputs, dim=-2)
    if len(network_input_shape) == 2:
        output = output.squeeze(0)
    return output, hidden

@torch.no_grad()
def get_gradient_stats(parameters):
    grad_min = float('inf')
    grad_max = float('-inf')
    total_norm_square = 0.0

    for param in parameters:
        if param.grad is not None:
            grad_min = min(grad_min, param.grad.min().item())
            grad_max = max(grad_max, param.grad.max().item())
            param_norm = torch.sum(param.grad.data ** 2)
            total_norm_square += param_norm.item()

    return grad_min, grad_max, total_norm_square


def selective_scan_ref(u, delta, A, B, C, start, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    start: r(B D L)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if start is not None:
        # deltaA = 0 if start = 1 (reset hidden start)
        deltaA = torch.einsum('bdln,bdl->bdln', deltaA, 1 - start)
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


def cumsum_fun_selective_scan(x, start, lst_cumsum):
    """
    利用 selective_scan_fn 实现带重置的累计和
    递推公式：
        y[0] = lst_cumsum * (1 - start[0]) + x[0]
        y[i] = (1 - start[i]) * y[i-1] + x[i]  for i>=1

    参数：
      x: (B, L, C)
      start: (B, L, 1)，其中值为0/1，1 表示在此处重置累计
      lst_cumsum: (B, 1, C)，初始累计状态

    返回：
      y: (B, L, C)，与原 for 循环等价的输出
    """
    B, L, C = x.shape

    # 为了符合 selective_scan 的接口，我们将通道维度作为状态维度
    # 将 x 从 (B, L, C) 转换为 (B, C, L)
    u = x.transpose(1, 2).contiguous()  # (B, C, L)

    # 构造 delta，全1张量，形状 (B, C, L)
    delta = torch.ones_like(u)

    # 设置 A 为全零，形状 (C, 1)（注意：此处 D = C，N = 1）
    A = torch.zeros(C, 1, device=x.device, dtype=x.dtype)

    # 设置 B 为全1，形状 (C, 1)
    B_param = torch.ones(C, 1, device=x.device, dtype=x.dtype)

    # 设置 C 为全1，形状 (C, 1)，这样最终 y = einsum(x, C) 得到的就是状态本身
    C_param = torch.ones(C, 1, device=x.device, dtype=x.dtype)

    # selective_scan 要求 start 的 shape 为 (B, D, L)，这里 D=C
    # 原来的 start 为 (B, L, 1)，转换为 (B, C, L)（对通道广播即可）
    start_expanded = start.transpose(1, 2).expand(B, C, L).contiguous()

    # 修改 u 的第一个时刻，使其包含初始状态 lst_cumsum 的贡献
    # 原始递推要求 y[0] = lst_cumsum*(1 - start[0]) + x[0]
    # u[...,0] 原本为 x[0]，因此更新 u[...,0]：
    u0 = u[..., 0]  # (B, C)
    # lst_cumsum: (B, 1, C) -> squeeze 成 (B, C)
    lst = lst_cumsum.squeeze(1)
    # start_expanded[...,0] 为 (B, C)
    u0_new = u0 + lst * (1 - start_expanded[..., 0])
    # 更新 u 的第一个时刻
    u = torch.cat([u0_new.unsqueeze(-1), u[..., 1:]], dim=-1)

    # 调用 selective_scan_fn，递推计算 s[0] = u[...,0], s[i] = (1-start)*s[i-1] + u[...,i]
    # 输出 y_out 的 shape 为 (B, C, L)
    # y_out = selective_scan_fn(u, delta, A, B_param, C_param, start_expanded,
    #                             D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False)
    if u.device == torch.device('cpu') or selective_scan_fn is None:
        y_out = selective_scan_ref(u, delta, A, B_param, C_param, start_expanded,
                                   D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False)
    else:
        # print(f'use cuda selective scan')
        y_out = selective_scan_fn(
            u, delta, A, B_param, C_param, start_expanded,
            D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False
        )


    # 将输出转换回 (B, L, C)
    y_final = y_out.transpose(1, 2)
    return y_final