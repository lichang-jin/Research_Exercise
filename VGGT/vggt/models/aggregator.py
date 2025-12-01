import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from typing import Tuple, List
import logging

from ..layers.block import Block
from ..layers import PatchEmbed
from ..layers.rope import RotaryPositionEmbedding2D, PositionGetter
from ..layers.vision_transformer import ViT_small, ViT_base, ViT_large, ViT_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    def __init__(
        self,
        image_size : int = 518,
        patch_size : int = 14,
        embed_dim : int = 1024,
        depth : int = 24,
        num_heads : int = 16,
        mlp_ratio : float = 4.0,
        num_register_tokens : int = 4,
        block_fn : nn.Module = Block,
        qkv_bias : bool = True,
        proj_bias : bool = True,
        ffn_bias : bool = True,
        patch_embed : str = "dinov2_vitl14_reg",
        aa_order : List[str] = None,
        aa_block_size : int = 1,
        qk_norm : bool = True,
        rope_frequency : int = 100,
        init_values : float = 1e-2,
    ) -> None:
        """
        The Aggregator applies alternating-attention over input frames.
        :param image_size (int): Image size in pixels.
        :param patch_size (int): Size of each patch for PatchEmbed.
        :param embed_dim (int): Dimension of the token embeddings.
        :param depth (int): Number of blocks.
        :param num_heads (int): Number of attention heads.
        :param mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        :param num_register_tokens (int): Number of register tokens.
        :param block_fn (nn.Module): The block type used for attention.
        :param qkv_bias (bool): Whether to include bias in QKV projections.
        :param proj_bias (bool): Whether to include bias in the output projection.
        :param ffn_bias (bool): Whether to include bias in MLP layers.
        :param patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        :param aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        :param aa_block_size (int): How many blocks to group under each attention type before switching.
        :param qk_norm (bool): Whether to apply QK normalization.
        :param rope_frequency (int): Base frequency for rotary embedding. -1 to disable.
        :param init_values (float): Init scale for layer scale.
        """
        super().__init__()
        if aa_order is None:
            aa_order = ["frame", "global"]

        self.__build_patch_embed__(patch_embed, image_size, patch_size, embed_dim, num_register_tokens)
        self.rope = RotaryPositionEmbedding2D(frequency=rope_frequency) if rope_frequency > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                rope=self.rope,
            ) for _ in range(depth)
        ])
        self.global_blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                rope=self.rope,
            ) for _ in range(depth)
        ])

        self.aa_order = aa_order
        self.aa_block_size = aa_block_size
        self.depth = depth
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({self.depth}) must be divisible by aa_block_size ({self.aa_block_size})")
        self.aa_block_num = self.depth // self.aa_block_size
        self.patch_size = patch_size
        # Note: We have two camera tokens, one for the first frame and one for the rest.
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        self.patch_start_idx = 1 + num_register_tokens
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)
        self.use_reentrant = False # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed: str = "dinov2_vitl14_reg",
        image_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        num_register_tokens: int = 4,
        block_chunks: int = 0,
        init_values: float = 1.0,
        interpolate_antialias: bool = True,
        interpolate_offset: float = 0.0,
    ) -> None:
        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(
                img_size=image_size,
                patch_size=patch_size,
                in_channels=3,
                embed_dim=embed_dim,
            )
        else:
            vit_models = {
                "dinov2_vitl14_reg": ViT_large,
                "dinov2_vitb14_reg": ViT_base,
                "dinov2_vits14_reg": ViT_small,
                "dinov2_vitg14_reg": ViT_giant2,
            }
            self.patch_embed = vit_models[patch_embed](
                img_size=image_size,
                patch_size=patch_size,
                in_channels=3,
                num_register_tokens=num_register_tokens,
                block_chunks=block_chunks,
                init_values=init_values,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
            )
            if hasattr(self.patch_embed, "mask_token"):
                # Disable gradient updates for mask token
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Tuple[List[Tensor], int]:
        """
        :param images: Input images with shape [B, S, 3, H, W].
        :return: The list of outputs from the attention blocks.
        """
        B, S, C, H, W = images.shape
        if C != 3:
            raise ValueError(f"Expected input images to have 3 channels, but got {C} channels")
        images = images.view(B * S, C, H, W)
        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_tokens"]

        def slice_expand_and_flatten(token_tensor, B_, S_) -> Tensor:
            """
            Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
            1) Uses the first position (index=0) for the first frame only
            2) Uses the second position (index=1) for all remaining frames (S-1 frames)
            3) Expands both to match batch size B
            4) Concatenates to form (B, S, X, C)
            5) Flattens to (B*S, X, C) for processing
            :return: Processed tokens with shape (B*S, X, C)
            """
            query = token_tensor[:, 0:1, ...].expand(B_, 1, *token_tensor.shape[2:])
            others = token_tensor[:, 1:, ...].view(B_, S_ - 1, *token_tensor.shape[2:])
            combined = torch.cat([query, others], dim=1)
            return combined.view(B_ * S_, *combined.shape[2:])

        camera_tokens = slice_expand_and_flatten(self.camera_token, B, S)
        register_tokens = slice_expand_and_flatten(self.register_token, B, S)
        tokens = torch.cat([camera_tokens, register_tokens, patch_tokens], dim=1)

        pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device) if self.rope is not None else None
        if self.patch_start_idx > 0:
            # Do not use position embedding for special tokens (camera and register tokens), set pos 0 for them
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        _, P, C = tokens.shape
        frame_idx = 0
        global_idx = 0
        output_list = []
        for _ in range(self.aa_block_num):
            frame_intermediates, global_intermediates = None, None
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(tokens, B, S, P, C, frame_idx, pos)
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(tokens, B, S, P, C, global_idx, pos)
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
            for i in range(self.aa_block_size):
                output_list.append(torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1))
        del frame_intermediates, global_intermediates
        return output_list, self.patch_start_idx


    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """Process frame attention blocks. We keep tokens in shape [B*S, P, C]."""
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        if pos is not None:
            if pos.shape != (B * S, P, 2):
                pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """Process global attention blocks. We keep tokens in shape [B, S*P, C]."""
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)
        if pos is not None:
            if pos.shape != (B, S * P, 2):
                pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        return tokens, global_idx, intermediates