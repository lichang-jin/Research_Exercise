import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from xformers.ops import memory_efficient_attention

from einops import rearrange
from functools import partial


XFORMERS_AVAILABLE = False

class Attention(nn.Module):
    def __init__(
        self,
        dim : int,
        num_heads : int = 8,
        qkv_bias : bool = True,
        proj_bias : bool = True,
        attn_drop : float = 0.0,
        proj_drop : float = 0.0,
        norm_layer : nn.Module = nn.LayerNorm,
        qk_norm : bool = False,
        fused_attn : bool = True,   # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = torch.unbind(qkv, 0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q, k = self.rope(q, pos), self.rope(k, pos)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        def summarize_qkv_chunk(
                query: Tensor,  # BxHxIxD
                key: Tensor,  # BxHxJxD
                value: Tensor,  # BxHxJxD
                mask: Tensor,  # BxJ
                attn_bias_chunk: Tensor,  # BxHxIxJ
                causal: bool,
                q_start_index,
                k_start_index,
                dropout: float = 0.0,
        ):
            q_chunk_size, k_chunk_size = query.shape[-2], key.shape[-2]  # I, J
            device = query.device

            weight = torch.einsum('b h i d, b h j d -> b h i j', query, key)  # BxHxIxJ
            if attn_bias_chunk is not None:
                weight = weight + attn_bias_chunk

            mask_value = -torch.finfo(weight.dtype).max
            if mask is not None:
                mask = rearrange(mask, 'b j -> b 1 1 j')
                weight = weight.masked_fill(~mask, mask_value)

            if causal and q_start_index < (k_start_index + k_chunk_size - 1):
                causal_mask = torch.ones((q_chunk_size, k_chunk_size), dtype=torch.bool, device=device).triu(
                    q_start_index - k_start_index + 1)
                weight = weight.masked_fill(causal_mask, mask_value)

            weight_max = weight.amax(dim=-1, keepdim=True).detach()
            weight = weight - weight_max

            exp_weight = weight.exp()
            exp_weight = F.dropout(exp_weight, p=dropout)
            weight_value = torch.einsum('b h i j, b h j d -> b h i d', exp_weight, value)
            return exp_weight.sum(dim=-1), weight_value, rearrange(weight_max, '... 1 -> ...')

        def memory_efficient_attention_self(
                query: Tensor,
                key: Tensor,
                value: Tensor,
                mask: Tensor = None,
                causal: bool = False,
                attn_bias: Tensor = None,
                q_bucket_size=512,
                k_bucket_size=1024,
                eps=1e-8,
                dropout: float = 0.0,
                training: bool = False,
        ) -> Tensor:
            checkpointed_summarize_qkv_chunk = partial(checkpoint, summarize_qkv_chunk)  # 固定函数的一个参数

            scale = query.shape[-1] ** -0.5
            query = query * scale

            needs_backwards = query.requires_grad or key.requires_grad or value.requires_grad
            summarize_qkv_fn = checkpointed_summarize_qkv_chunk if needs_backwards else summarize_qkv_chunk

            # chunk all the inputs
            q_chunks = query.split(q_bucket_size, dim=-2)  # BxHxIxD
            k_chunks = key.split(k_bucket_size, dim=-2)  # BxHxJxD
            v_chunks = value.split(k_bucket_size, dim=-2)  # BxHxJxD
            mask_chunks = mask.split(k_bucket_size, dim=-1) if mask is not None else ((None,) * len(k_chunks))

            attn_bias_chunks = []
            if attn_bias is not None:
                attn_bias_chunks = attn_bias.split(q_bucket_size, dim=-2)  # BxHxIxD
                attn_bias_chunks = list(map(lambda t: t.split(k_bucket_size, dim=-1), attn_bias_chunks))  # BxHxIxJ

            # loop through all chunks
            out = []
            for q_index, q_chunk in enumerate(q_chunks):
                exp_weights = []
                weighted_values = []
                weight_maxes = []

                for k_index, (k_chunks, v_chunks, mask_chunk) in enumerate(zip(k_chunks, v_chunks, mask_chunks)):
                    q_start_index = q_index * q_bucket_size
                    k_start_index = k_index * k_bucket_size

                    if causal and k_start_index > (q_start_index + q_chunk.shape[-2] - 1):
                        # if chunk is to be all masked out causally, skip
                        continue

                    attn_bias_chunk = attn_bias_chunks[q_index][k_index] if attn_bias is not None else None

                    exp_weight_chunk, weighted_value_chunk, weight_max_chunk = summarize_qkv_fn(
                        q_chunk, k_chunks, v_chunks, mask_chunk, attn_bias_chunk, causal, q_start_index, k_start_index,
                        dropout if training else 0.
                    )
                    exp_weights.append(exp_weight_chunk)
                    weighted_values.append(weighted_value_chunk)
                    weight_maxes.append(weight_max_chunk)

                exp_weights = torch.stack(exp_weights, dim=-1)
                weighted_values = torch.stack(weighted_values, dim=-1)
                weight_maxes = torch.stack(weight_maxes, dim=-1)

                global_max = weight_maxes.amax(dim=-1, keepdim=True)
                renorm_factor = (weight_maxes - global_max).exp().detach()
                exp_weights = exp_weights * renorm_factor
                weighted_values = weighted_values * rearrange(renorm_factor, '... c -> ... 1 c')

                all_values = weighted_values.sum(dim=-1)
                all_weights = exp_weights.sum(dim=-1)
                normalized_values = all_values / (rearrange(all_weights, '... -> ... 1') + eps)
                out.append(normalized_values)

            return torch.cat(out, dim=-2)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)        # [B, N, nh, hd]
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
