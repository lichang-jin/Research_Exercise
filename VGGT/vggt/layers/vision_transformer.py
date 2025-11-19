import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.init import trunc_normal_
from . import MLP, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block

from functools import partial
import math
import logging
from typing import Tuple, Sequence, Callable, Union, List


logger = logging.getLogger("DINOv2")

def named_apply(func: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        func(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(func=func, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        func(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for block in self:
            x = block(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size = 224,
        patch_size = 16,
        in_channels = 3,
        embed_dim = 768,
        depth = 12,
        num_heads = 12,
        mlp_ratio = 4.0,
        qkv_bias = True,
        ffn_bias = True,
        proj_bias = True,
        drop_path_rate = 0.0,
        drop_path_uniform = False,
        init_values = None,
        embed_layer = PatchEmbed,
        act_layer = nn.GELU,
        block_fn = Block,
        ffn_layer = "mlp",
        block_chunks = 1,
        num_register_tokens = 0,
        interpolate_antialias = False,
        interpolate_offset = 0.1,
        qk_norm = False,
    ) -> None:
        """
        :param img_size (int, tuple): input image size
        :param patch_size (int, tuple): patch size
        :param in_channels (int): number of input channels
        :param embed_dim (int): embedding dimension
        :param depth (int): depth of transformer
        :param num_heads (int): number of attention heads
        :param mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        :param qkv_bias (bool): enable bias for qkv if True
        :param ffn_bias (bool): enable bias for proj in attn if True
        :param proj_bias (bool): enable bias for ffn if True
        :param drop_path_rate (float): stochastic depth rate
        :param drop_path_uniform (bool): apply uniform drop rate across blocks
        :param init_values (float): layer-scale init values, None or 0 => no layer-scale
        :param embed_layer (nn.Module): patch embedding layer
        :param act_layer (nn.Module): MLP activation layer
        :param block_fn (nn.Module): transformer block class
        :param ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
        :param block_chunks (int): split block sequence into block_chunks units for FSDP wrap
        :param num_register_tokens (int): number of extra cls tokens (so-called "registers")
        :param interpolate_antialias (str): flag to apply 抗锯齿 when interpolating positional embeddings
        :param interpolate_offset (float)： work-around offset to apply when interpolating positional embeddings
        :param qk_norm (bool): apply layer norm on qk before softmax if True
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.num_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.use_reentrant = False

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert self.num_register_tokens >= 0
        self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, embed_dim)) if self.num_register_tokens > 0 else None

        if drop_path_uniform:
            dpr = [drop_path_rate] * self.num_blocks
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_blocks)]

        if ffn_layer == "mlp":
            logger.info("Using MLP layer as feed-forward network.")
            ffn_layer = MLP
        elif ffn_layer == "swiglu" or ffn_layer == "swiglufused":
            logger.info("Using SwiGLU layer as feed-forward network.")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("Using Identity layer as feed-forward network.")
            ffn_layer = lambda *args, **kwargs: nn.Identity()
        else:
            raise NotImplementedError(f"Unknown ffn_layer: {ffn_layer}.")

        block_list = [
            block_fn (
                dim = embed_dim,
                num_heads = num_heads,
                mlp_ratio = mlp_ratio,
                qkv_bias = qkv_bias,
                proj_bias = proj_bias,
                ffn_bias = ffn_bias,
                drop_path = dpr[i],
                norm_layer = norm_layer,
                act_layer = act_layer,
                ffn_layer = ffn_layer,
                init_values = init_values,
                qk_norm = qk_norm,
            )
            for i in range(self.num_blocks)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunk_size = self.num_blocks // block_chunks
            for i in range(0, self.num_blocks, chunk_size):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + block_list[i : i + chunk_size])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(block_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)

        def _init_weights_vit_timm(module: nn.Module, name: str = ""):
            """ViT weight initialization, original timm impl (for reproducibility)"""
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        named_apply(_init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        num_patch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if num_patch == N and w == h: # no interpolation
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]

        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert M * M == N, "The number of patches must be a square number."
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (w0, h0)
        patch_pos_embed = F.interpolate(patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2), mode="bicubic", antialias=self.interpolate_antialias, **kwargs)
        assert patch_pos_embed.shape[-2:] == (w0, h0)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks = None):
        B, nc, h, w = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
        return x

    def forward_features(self, x, masks = None):
        if isinstance(x, list):
            # forward_features_list
            x_list = [self.prepare_tokens_with_masks(x_, masks_) for x_, masks_ in zip(x, masks)]
            for blk in self.blocks:
                if self.training:
                    x_list = checkpoint(blk, x_list, use_reentrant=self.use_reentrant)
                else:
                    x_list = blk(x_list)
            output = []
            for x_, masks_ in zip(x_list, masks):
                x_norm = self.norm(x_)
                output.append({
                    "x_norm_cls-token": x_norm[:, 0],
                    "x_norm_register-tokens": x_norm[:, 1:self.num_register_tokens + 1],
                    "x_norm_tokens": x_norm[:, self.num_register_tokens + 1:],
                    "x": x_,
                    "masks": masks_,
                })
            return output

        else:
            x = self.prepare_tokens_with_masks(x, masks)
            for blk in self.blocks:
                if self.training:
                    x = checkpoint(blk, x, use_reentrant=self.use_reentrant)
                else:
                    x = blk(x)
            x_norm = self.norm(x)
            return {
                "x_norm_cls-token": x_norm[:, 0],
                "x_norm_register-tokens": x_norm[:, 1:self.num_register_tokens + 1],
                "x_norm_tokens": x_norm[:, self.num_register_tokens + 1:],
                "x": x,
                "masks": masks,
            }

    def forward(self, *args, is_training = True, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_cls-token"])


    def _get_intermediate_layers_not_chunked(self, x, n = 1):
        # If n: int, take the n last blocks. If n: list, take them.
        x = self.prepare_tokens_with_masks(x)
        total_blocks = self.num_blocks
        blocks_idx = range(total_blocks - n, total_blocks) if isinstance(n, int) else n
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_idx:
                output.append(x)
        assert len(output) == len(blocks_idx), f"Only {len(output)} blocks found, but {len(blocks_idx)} expected."
        return output

    def _get_intermediate_layers_chunked(self, x, n = 1):
        # If n: int, take the n last blocks. If n: list, take them.
        x = self.prepare_tokens_with_masks(x)
        total_blocks = self.num_blocks
        blocks_idx = range(total_blocks - n, total_blocks) if isinstance(n, int) else n
        output = []
        i = 0
        for blk_chunk in self.blocks:
            for blk in blk_chunk[i:]:   # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_idx:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_idx), f"Only {len(output)} blocks found, but {len(blocks_idx)} expected."
        return output

    def get_intermediate_layers(self, x, n = 1, reshape = False, return_class_token = False, norm = True):
        """Get intermediate layers output. If return_class_token => True, return Tuple[Tuple[Tensor]], else return Tuple[Tensor]."""
        if self.chunked_blocks:
            output = self._get_intermediate_layers_chunked(x, n)
        else:
            output = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            output = [self.norm(out) for out in output]
        class_tokens = [out[:, 0] for out in output]
        output = [out[:, 1 + self.num_register_tokens:] for out in output]
        if reshape:
            B, _, w, h = x.shape
            output = [out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous() for out in output]
        if return_class_token:
            return tuple(zip(output, class_tokens))
        else:
            return tuple(output)
