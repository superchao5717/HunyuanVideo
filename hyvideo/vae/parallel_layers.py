import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.utils import _triple, _reverse_repeat_tuple
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_3_t
import torch_npu
from einops import rearrange
from xfuser.core.distributed import (
        get_sequence_parallel_world_size,
        get_sequence_parallel_rank,
        )
from .unet_causal_3d_blocks import prepare_causal_attention_mask
from diffusers.models.attention_processor import Attention

USE_FASCORE = True




def patchify(hidden_state, dim, is_overlap, world_size, rank):
    length = hidden_state.shape[dim]
    if is_overlap:
        overlap = rank % 2
        start_idx = (length + world_size - 1) // world_size * rank - overlap
        end_idx = min((length + world_size - 1) // world_size * (rank + 1) - overlap + 1, length)
    else:
        start_idx = (length + world_size - 1) // world_size * rank
        end_idx = min((length + world_size - 1) // world_size * (rank + 1), length)
    idx = torch.arange(start_idx, end_idx, device=f"npu:{rank}")
    return hidden_state.index_select(dim, idx).clone()


def depatchify(patch_hidden_state, dim, is_overlap, world_size, rank):
    if is_overlap:
        overlap = rank % 2
        start_idx = overlap
        end_idx = patch_hidden_state.shape[dim] + overlap - 1
        idx = torch.arange(start_idx, end_idx, device=f"npu:{rank}")
        patch_hidden_state = patch_hidden_state.index_select(dim, idx)
    
    patch_length_list = [torch.empty([1], dtype=torch.int64, device=f"npu:{rank}") 
                            for _ in range(world_size)]
    dist.all_gather(
        patch_length_list,
        torch.tensor(
            [patch_hidden_state.shape[dim]],
            dtype=torch.int64,
            device=f"npu:{rank}"
        )
    )
    patch_shape = list(patch_hidden_state.shape)
    patch_hidden_state_list = []
    for i in range(world_size):
        patch_shape[dim] = patch_length_list[i].item()
        patch_hidden_state_list.append(
            torch.empty(tuple(patch_shape), dtype=patch_hidden_state.dtype, device=f"npu:{rank}"))
    dist.all_gather(
        patch_hidden_state_list,
        patch_hidden_state.contiguous()
    )

    return torch.cat(patch_hidden_state_list, dim)


class BaseModule(nn.Module):
    def __init__(self, module: nn.Module, split_dim=-1):
        super(BaseModule, self).__init__()
        self.module = module
        self.split_dim = split_dim

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class P2PComm:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size
        self._reqs = []
    def send_recv(self, to_send, recv_tensor=None, send_rank=None, recv_rank=None):
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor
        
        if send_rank is None:
            send_rank = self.send_rank
        if recv_rank is None:
            recv_rank = self.recv_rank
    
        send_op = dist.P2POp(dist.isend, to_send, send_rank)
        recv_op = dist.P2POp(dist.irecv, res, recv_rank)
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        self._reqs.extend(reqs)
        return res

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = []


class PatchGroupNorm3d(BaseModule):
    def __init__(self, group_norm: nn.GroupNorm, split_dim, to_float32=False):
        super(PatchGroupNorm3d, self).__init__(group_norm, split_dim)
        self.to_float32 = to_float32
        if group_norm.affine:
            self.module.weight = group_norm.weight
            self.module.bias = group_norm.bias 

    def _slice(self, x, start_idx=None, end_idx=None):
        if self.split_dim == -1:
            if start_idx is None:
                return x[..., :end_idx]
            elif end_idx is None:
                return x[..., start_idx:]
            else:
                return x[..., start_idx:end_idx]
        elif self.split_dim == -2:
            if start_idx is None:
                return x[..., :end_idx, :]
            elif end_idx is None:
                return x[..., start_idx:, :]
            else:
                return x[..., start_idx:end_idx, :]
    
    def forward(self, x: Tensor) -> Tensor:
        # world_size = get_sequence_parallel_world_size()
        # rank = get_sequence_parallel_rank()
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        bs, channels, t, h, w = x.shape
        x = x.view(bs, self.module.num_groups, -1, t, h, w)

        dtype = x.dtype
        if self.to_float32:
            x = x.to(torch.float32)
        
        if rank % 2 == 0:
            x_unpad = self._slice(x, None, -1)
            group_mean = x_unpad.mean(dim=(2, 3, 4, 5), keepdim=True)
            group_var = torch.var(x_unpad, dim=(2, 3, 4, 5), unbiased=False, keepdim=True)
        else:
            x_unpad = self._slice(x, 1, None)
            group_mean = x[..., 1:, :].mean(dim=(2, 3, 4, 5), keepdim=True)
            group_var = torch.var(x[..., 1:, :], dim=(2, 3, 4, 5), unbiased=False, keepdim=True)
        size = torch.full((1, self.module.num_groups, 1, 1, 1, 1), x.shape[self.split_dim] - 1, dtype=dtype, device=x.device)
        group_mean_var_size = torch.cat((group_mean, group_var, size), dim=0)
        group_mean_var_size_list = [torch.empty_like(group_mean_var_size) for _ in range(world_size)]
        dist.all_gather(group_mean_var_size_list, group_mean_var_size)

        mean, var = 0, 0
        full_size = sum([tmp[-1, 0, 0, 0, 0, 0] for tmp in group_mean_var_size_list])
        for mean_var_size in group_mean_var_size_list:
            size = mean_var_size[-1, 0, 0, 0, 0, 0]
            mean += size / full_size * mean_var_size[:bs]
            var += size / full_size * mean_var_size[bs:-1]

        x = (x - mean) / torch.sqrt(var + self.module.eps)
        x = x.view(bs, -1, t, h, w).to(dtype)
        x = x * self.module.weight[None, :, None, None, None] + self.module.bias[None, :, None, None, None]
        return x
    
def get_normalization_helper(norm_type: str, norm_dim: int, eps: float = 1e-5):
    match norm_type:
        case None:
            return nn.Identity()
        case 'layer_norm':
            return nn.LayerNorm(norm_dim, eps=eps)
        case 'llama_rms_norm':
            return LlamaRMSNorm(norm_dim, eps=eps)
        case _:
            error_msg = "`norm_type` is not supported!"
            raise ValueError(error_msg)
        
class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, x, shift, scale):
        return torch_npu.npu_layer_norm_eval(
            x, normalized_shape=[self.hidden_size], weight=scale, bias=shift, eps=self.eps)

        
class GroupNorm3dAdapter(nn.Module):
    def __init__(self, group_norm: nn.GroupNorm):
        super().__init__()
        self.module = PatchGroupNorm3d(
            num_groups=group_norm.num_groups,
            num_channels=group_norm.num_channels,
            eps=group_norm.eps,
            affine=group_norm.affine
        )
        if group_norm.affine:
            self.module.weight = group_norm.weight
            self.module.bias = group_norm.bias

    def forward(self, x):
        return self.module(x)
    
class PatchConv3d(BaseModule):
    def __init__(self, module: nn.Conv3d, split_dim, num_blocks=2):
        super(PatchConv3d, self).__init__(module, split_dim)
        self.num_blocks = num_blocks
        # self.module.weight.data = conv2d.weight.data
        # self.module.bias.data = conv2d.bias.data

        if isinstance(self.module.dilation, int):
            assert self.module.dilation == 1, "dilation is not supported in PatchConv3d"
        else:
            for i in self.module.dilation:
                assert i == 1, "dilation is not supported in PatchConv3d"
        assert self.module.kernel_size[-2] == 1 
        assert self.module.stride == (1, 1, 1)
        assert self.num_blocks >= 2
        
    def forward(self, x):
        if (get_sequence_parallel_world_size() == 1) or (self.module.kernel_size[-2] == 1):
            return self.module(x)     
        


class PatchCausalConv3d(BaseModule):
    def __init__(self, module: nn.Conv3d, split_dim, num_blocks=2):
        super(PatchCausalConv3d, self).__init__(module, split_dim)
        self.num_blocks = num_blocks

        if isinstance(self.module.conv.dilation, int):
            assert self.module.conv.dilation == 1, "dilation is not supported in PatchCausalConv3d"
        else:
            for i in self.module.conv.dilation:
                assert i == 1, "dilation is not supported in PatchCausalConv3d"
        assert self.module.conv.kernel_size[-2] == 3 or self.module.conv.kernel_size[-2] == 1, "PatchCausalConv3d only support kernel_size (3, 3, 3) or (1, 1, 1)"
        assert self.module.conv.stride == (1, 1, 1), "PatchCausalConv3d only support stride (1, 1, 1)"
        assert self.num_blocks >= 1, "block_size should larger than 2 for pipeline computation" 

        self.module.time_causal_padding = list(self.module.time_causal_padding)
    
    def _slice(self, x, start_idx=None, end_idx=None, dim=-1):
        if dim == -1:
            if start_idx == -1:
                return x[..., -1:]
            elif end_idx == 1:
                return x[..., :1]
            else:
                return x[..., start_idx:end_idx]
        elif dim == -2:
            if start_idx == -1:
                return x[..., -1:, :]
            elif end_idx == 1:
                return x[..., :1, :]
            else:
                return x[..., start_idx:end_idx, :]
    
    def forward(self, x):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if (world_size == 1) or (self.module.conv.kernel_size[-2] == 1):
            return self.module(x)     
        else:         
            comm = P2PComm(rank, world_size)

            if self.split_dim == -1:
                self.module.time_causal_padding[0:2] = [0,0]
                dim_in, dim_out = -2, -1
                x = F.pad(x, self.module.time_causal_padding, mode=self.module.pad_mode)
                size_in, size_out = x.shape[-2:]
                bs, channels, t, h, w = x.shape
                overlap_shape = (bs, channels, t, h, 1)
            elif self.split_dim == -2:
                self.module.time_causal_padding[2:4] = [0,0]
                size_out, size_in = x.shape[-2:]
                dim_out, dim_in = -2, -1
                x = F.pad(x, self.module.time_causal_padding, mode=self.module.pad_mode)
                bs, channels, t, h, w = x.shape
                overlap_shape = (bs, channels, t, 1, w)
            else:
                raise ValueError(f"Cannot split video sequence into ulysses_degree x ring_degree ({world_size}) parts evenly")
            

            stride_out = size_out // 2
            stride_in = (size_in + self.num_blocks - 1) // self.num_blocks

            outputs = []
            for o in range(2):
                start_o = o * stride_out
                end_o = min((o + 1) * stride_out + 2, size_out)
                x_o = self._slice(x, size_out-end_o, size_out-start_o, dim_out) if rank % 2 == 0 else self._slice(x, start_o, end_o, dim_out)

                if o == 0:
                    if rank % 2 == 0 and rank != 0:
                        send = self._slice(x, None, 1, dim_out)
                        pedding_recv = comm.send_recv(send.contiguous(), send_rank=rank-1, recv_rank=rank-1)
                    elif rank % 2 != 0 and rank != world_size - 1:
                        send = self._slice(x, -1, None, dim_out)
                        pedding_recv = comm.send_recv(send.contiguous(), send_rank=rank+1, recv_rank=rank+1)
                else:
                    if rank % 2 == 0:
                        x_o = torch.cat([recv, x_o], dim=dim_out)
                        send = self._slice(outputs[0], -1, None, dim_out)
                        pedding_recv = comm.send_recv(send.contiguous(), send_rank=rank+1, recv_rank=rank+1)
                    else:
                        x_o = torch.cat([x_o, recv], dim=dim_out)
                        send = self._slice(outputs[0], None, 1, dim_out)
                        pedding_recv = comm.send_recv(send.contiguous(), send_rank=rank-1, recv_rank=rank-1)

                _outputs = []
                for i in range(self.num_blocks):
                    start_i = i * stride_in
                    end_i = min((i + 1) * stride_in + 2, size_in)
                    x_i = self._slice(x_o, start_i, end_i, dim_in)
                    _outputs.append(
                        F.conv3d(
                            x_i, 
                            self.module.conv.weight, 
                            self.module.conv.bias, 
                            self.module.conv.stride, 
                            # padding, 
                            dilation=self.module.conv.dilation, 
                            groups=self.module.conv.groups
                        )
                    )
                _outputs = torch.cat(_outputs, dim=dim_in)
                outputs.append(_outputs)

                if o == 0:
                    if rank == 0:
                        if self.module.pad_mode == "replicate":
                            recv = self._slice(x, None, 1, dim_out)
                        elif self.module.pad_mode == "constant":
                            recv = torch.zeros(
                                overlap_shape, dtype=x.dtype, device=x.device)
                        else:
                            raise NotImplementedError
                    elif rank == world_size - 1:
                        if self.module.pad_mode == "replicate":
                            recv = self._slice(x, -1, None, dim_out)
                        elif self.module.pad_mode == "constant":
                            recv = torch.zeros(
                                overlap_shape, dtype=x.dtype, device=x.device)
                        else:
                            raise NotImplementedError
                    else:
                        comm.wait()
                        recv = pedding_recv

                else:
                    comm.wait()
                    recv = pedding_recv
                    if rank % 2 == 0:
                        outputs.insert(0, recv)
                        outputs.reverse()
                    else:
                        outputs.insert(0, recv)

            # del x
            return torch.cat(outputs, dim=dim_out)
        

def register_upsample_forward(model):
    def _forward(self):
        def forward(
                hidden_states: torch.FloatTensor,
                output_size: Optional[int] = None,
                scale: float = 1.0,):
            rank = get_sequence_parallel_rank()
            assert hidden_states.shape[1] == self.channels

            # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
            dtype = hidden_states.dtype
            if dtype == torch.bfloat16:
                hidden_states = hidden_states.to(torch.float32)

            # if `output_size` is passed we force the interpolation output
            # size and do not make use of `scale_factor=2`
            if self.interpolate:
                B, C, T, H, W = hidden_states.shape
                first_h, other_h = hidden_states.split((1, T - 1), dim=2)
                if output_size is None:
                    if T > 1:
                        other_h = F.interpolate(other_h, scale_factor=self.upsample_factor, mode="nearest")

                    first_h = first_h.squeeze(2)
                    first_h = F.interpolate(first_h, scale_factor=self.upsample_factor[1:], mode="nearest")
                    first_h = first_h.unsqueeze(2)
                else:
                    raise NotImplementedError

                if T > 1:
                    hidden_states = torch.cat((first_h, other_h), dim=2)
                else:
                    hidden_states = first_h
                
                del first_h, other_h
                torch.cuda.empty_cache()
            


            # If the input is bfloat16, we cast back to bfloat16
            if dtype == torch.bfloat16:
                hidden_states = hidden_states.to(dtype)

            if self.conv.split_dim == -1:
                hidden_states = hidden_states[..., :-1] if rank % 2 == 0 else hidden_states[..., 1:]
            elif self.conv.split_dim == -2:
                hidden_states = hidden_states[..., :-1, :] if rank % 2 == 0 else hidden_states[..., 1:, :]
            else:
                raise NotImplementedError

            if self.use_conv:
                hidden_states = self.conv(hidden_states)
            return hidden_states
        return forward
    model.forward = _forward(model)


def register_vae_midblock_forward(model):
    def _forward(self):
        def forward(
                hidden_states: torch.FloatTensor, 
                temb: Optional[torch.FloatTensor] = None
                ) -> torch.FloatTensor:
            hidden_states = self.resnets[0](hidden_states, temb)
            for attn, resnet in zip(self.attentions, self.resnets[1:]):
                if attn is not None:
                    B, C, T, H, W = hidden_states.shape
                    hidden_states = attn(hidden_states, temb=temb)
                    hidden_states = rearrange(hidden_states, "b (f h w) c -> b c f h w", f=T, h=H, w=W)
                hidden_states = resnet(hidden_states, temb)
            return hidden_states
        return forward
    model.forward = _forward(model)

class AttnProcessor2_0_sp():
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, world_size, rank, split_dim):
        self.rank = rank
        self.world_size = world_size
        self.split_dim = split_dim

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        B, C, T, H, W = hidden_states.shape
        print(f"hidden_states shape: {hidden_states.shape}, hidden_states dtype: {hidden_states.dtype}\n")
        
        residual = hidden_states.clone() # b c f h w
        residual = rearrange(residual, "b c f h w -> b (f h w) c")

        hidden_states = attn.group_norm(hidden_states)  

        if self.rank % 2 == 0:
            if self.split_dim == -1:
                encoder_hidden_states = hidden_states[..., :-1].clone()
            elif self.split_dim == -2:
                encoder_hidden_states = hidden_states[..., :-1, :].clone()
        else:
            if self.split_dim == -1:
                encoder_hidden_states = hidden_states[..., 1:].clone()
            elif self.split_dim == -2:
                encoder_hidden_states = hidden_states[..., 1:, :].clone()

        hidden_states = rearrange(hidden_states, "b c f h w -> b (f h w) c")
        batch_size, q_sequence_length, _ = hidden_states.shape 
        query = attn.to_q(hidden_states)

        encoder_hidden_states = encoder_hidden_states.reshape(B, C, -1).permute(0, 2, 1).contiguous()

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        key = key.permute(0, 2, 1).reshape(B, C, T, H, -1).contiguous()
        value = value.permute(0, 2, 1).reshape(B, C, T, H, -1).contiguous()

        kv = torch.cat([key, value], dim=0)

        kv_B, kv_C, kv_T, kv_H, kv_W = kv.shape
        kv_full = torch.empty([self.world_size, kv_B, kv_C, kv_T, kv_H, kv_W], dtype=key.dtype, device=key.device)
        dist.all_gather_into_tensor(kv_full, kv)

        kv_full = kv_full.permute(1, 2, 3, 4, 0, 5).reshape(kv_B, kv_C, kv_T, kv_H, -1)
        key, value = kv_full.chunk(2, dim=0)        

        key = key.reshape(B, C, -1).permute(0, 2, 1).contiguous()
        value = value.reshape(B, C, -1).permute(0, 2, 1).contiguous()

 
        attention_mask = prepare_causal_attention_mask(
            T, H * W, hidden_states.dtype, hidden_states.device, batch_size=B
        ) # b, S_q, S_q

        # print(f"0 attention_mask.shape: {attention_mask.shape}, attention_mask.dtype: {attention_mask.dtype}\n")
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, q_sequence_length, batch_size, out_dim=4)

       
        attention_mask = attention_mask.reshape(batch_size, attn.heads, q_sequence_length, T, H, W)[..., 1:].contiguous()
        attention_mask_full = torch.empty([self.world_size, batch_size, attn.heads, q_sequence_length, T, H, W-1], dtype=attention_mask.dtype, device=attention_mask.device)
        dist.all_gather_into_tensor(attention_mask_full, attention_mask)
        attention_mask = attention_mask_full.permute(1, 2, 3, 4, 5, 0, 6).reshape(batch_size, attn.heads, q_sequence_length, -1)

        # print(f"1 attention_mask.shape: {attention_mask.shape}, attention_mask.dtype: {attention_mask.dtype}\n")

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.reshape(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        key = key.reshape(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        value = value.reshape(batch_size, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        
        if USE_FASCORE:
            attention_mask = attention_mask.to(torch.uint8)
            scale = 1.0 / math.sqrt(head_dim)
            hidden_states = torch_npu.npu_fusion_attention(
                                query, key, value,
                                head_num=attn.heads,
                                input_layout="BNSD",
                                scale=scale,
                                pse=None,
                                atten_mask=attention_mask,
                                pre_tockens=2147483647,
                                next_tockens=2147483647,
                                keep_prob=1.0,
                                sync=False,
                                inner_precise=0,
                                )[0]
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)


        hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AttnProcessor2_0_fa():
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, world_size, rank, split_dim):
        self.rank = rank
        self.world_size = world_size
        self.split_dim = split_dim

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        B, C, T, H, W = hidden_states.shape
        
        attention_mask = prepare_causal_attention_mask(
            T, H * W, hidden_states.dtype, hidden_states.device, batch_size=B
        )
        print(f"hidden_states shape: {hidden_states.shape}, hidden_states dtype: {hidden_states.dtype}\n")
        
        residual = hidden_states.clone()
        residual = rearrange(residual, "b c f h w -> b (f h w) c")

        
        hidden_states = attn.group_norm(hidden_states)
        hidden_states = rearrange(hidden_states, "b c f h w -> b (f h w) c")
        batch_size, q_sequence_length, _ = hidden_states.shape 

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        
        if USE_FASCORE:
            scale = 1.0 / math.sqrt(head_dim)
            hidden_states = torch_npu.npu_fusion_attention(
                                query, key, value,
                                head_num=attn.heads,
                                input_layout="BNSD",
                                scale=scale,
                                pse=None,
                                atten_mask=attention_mask,
                                pre_tockens=2147483647,
                                next_tockens=2147483647,
                                keep_prob=1.0,
                                sync=False,
                                inner_precise=0,
                                )[0]
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states