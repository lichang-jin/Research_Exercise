from .attention import Attention, MemEffAttention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import MLP

import torch
from torch import Tensor, nn

from typing import Callable, List, Any, Tuple, Dict, Optional
from xformers.ops import fmha

XFORMERS_AVAILABLE = False

class Block(nn.Module):
    def __init__(
        self,
        dim : int,
        num_heads : int,
        mlp_ratio : float = 4.0,
        qkv_bias : bool = True,
        proj_bias : bool = True,
        ffn_bias : bool = True,
        drop : float = 0.0,
        attn_drop : float = 0.0,
        drop_path : float = 0.0,
        act_layer : Callable[..., nn.Module] = nn.GELU,
        norm_layer : Callable[..., nn.Module] = nn.LayerNorm,
        attn_class : Callable[..., nn.Module] = Attention,
        ffn_layer : Callable[..., nn.Module] = MLP,
        qk_norm : bool = False,
        fused_attn : bool = True,
        init_values=None,
        rope=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attention = attn_class(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )
        self.layer_scale1 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.layer_scale2 = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, pos=None) -> Tensor:
        def attn_residual_func(x: Tensor, pos=None) -> Tensor:
            return self.layer_scale1(self.attention(self.norm1(x), pos=pos))
        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.layer_scale2(self.mlp(self.norm2(x)))

        def drop_add_residual_stochastic_depth(x: Tensor, residual_func, sample_drop_ratio: float = 0.0, pos=None) -> Tensor:
            # 1) extract subset using permutation
            brange, residual_scale_factor = get_brange_scales(x, sample_drop_ratio)
            x_subset = x[brange]

            # 2) apply residual function
            if pos is not None:
                # if necessary, apply rope to the subset
                pos_subset = pos[brange]
                residual = residual_func(x_subset, pos_subset)
            else:
                residual = residual_func(x_subset)
            x_flat = x.flatten(1)
            residual_flat = residual.flatten(1)

            # 3) add the residual
            x_plus_residual = torch.index_add(x_flat, 0, brange, residual_flat.to(dtype=x.dtype), alpha=residual_scale_factor)
            return x_plus_residual.view_as(x)

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(x, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio, pos=pos)
            x = drop_add_residual_stochastic_depth(x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio)
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos))
            x = x + self.drop_path2(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x, pos=pos)
            x = x + ffn_residual_func(x)
        return x


class NestedTensorBlock(Block):
    def forward_nested(self, x_list: List[Tensor]) -> List[Tensor]:
        """x_list contains a list of tensors to nest together and run."""
        assert isinstance(self.attention, MemEffAttention)

        attn_bias_cache: Dict[Tuple, Any] = {}  # 存储已计算的掩码

        def attn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
            return self.attention(self.norm1(x), attn_bias=attn_bias)
        def ffn_residual_func(x: Tensor, attn_bias=None) -> Tensor:
            return self.mlp(self.norm2(x))

        def get_attn_bias_and_cat(x_list: List[Tensor], branges=None) -> Tensor:
            """This function will perform the index select, cat the tensors, and provide the attn_bias from cache."""
            batch_sizes = [b.shape[0] for b in branges] if branges is not None else [x.shape[0] for x in x_list]
            all_shapes = tuple((b, x.shape[1]) for b, x in zip(batch_sizes, x_list))
            if all_shapes not in attn_bias_cache.keys():
                seq_lens = []
                for b, x in zip(batch_sizes, x_list):
                    for _ in range(b):
                        seq_lens.append(x.shape[1])


        def drop_add_residual_stochastic_depth_list(x: List[Tensor], residual_func, sample_drop_ratio: float = 0.0, scaling_vector=None) -> Tensor:
            # 1) generate random set of indices for dropping samples in the batch
            brange_scales = [get_brange_scales(x, sample_drop_ratio) for x in x_list]
            branges = [s[0] for s in brange_scales]
            residual_scale_factors = [s[1] for s in brange_scales]

            # 2) get attention bias and index + concat the tensors
            attn_bias = self.attention.get_attn_bias(branges, scaling_vector=scaling_vector)


        if self.training and self.sample_drop_ratio > 0.0:
            x_list = drop_add_residual_stochastic_depth



def get_brange_scales(x: Tensor, sample_drop_ratio: float = 0.0):
    B, N, D = x.shape
    sample_subset_size = max(int(B * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(B, device=x.device))[:sample_subset_size]  # 生成一个从 0 到 B-1 的随机排列
    residual_scale_factor = B / sample_subset_size
    return brange, residual_scale_factor

