import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.quantized import FloatFunctional

from typing import List, Tuple, Union

from .head_act import activate_head


class DPTHead(nn.Module):
    def __init__(
        self,
        dim_in : int,
        patch_size : int = 14,
        dim_out : int = 4,
        point_act : str = "inv_log",
        confidence_act : str = "exp1",
        features : int = 256,
        out_channels : List[int] = None,
        intermediate_layer_idx : List[int] = None,
        pos_embed : bool = True,
        feature_only : bool = False,
        down_ratio : int = 1,
    ) -> None:
        """
        :param dim_in (int): Input dimension (channels).
        :param patch_size (int, optional): Patch size.
        :param dim_out (int, optional): Number of output channels.
        :param point_act (str, optional): Activation type.
        :param confidence_act (str, optional): Confidence activation type.
        :param features (int, optional): Feature channels for intermediate representations.
        :param out_channels (List[int], optional): Output channels for each intermediate layer.
        :param intermediate_layer_idx (List[int], optional): Indices of layers from aggregated tokens used for DPT.
        :param pos_embed (bool, optional): Whether to use positional embedding.
        :param feature_only (bool, optional): If True, return features only without the last several layers and activation head.
        :param down_ratio (int, optional): Downscaling factor for the output resolution.
        """
        if out_channels is None:
            out_channels = [256, 512, 1024, 1024]
        if intermediate_layer_idx is None:
            intermediate_layer_idx = [4, 11, 17, 23]

        super().__init__()
        self.patch_size = patch_size
        self.point_act = point_act
        self.confidence_act = confidence_act
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.intermediate_layer_idx = intermediate_layer_idx
        self.down_ratio = down_ratio

        self.norm = nn.LayerNorm(dim_in)
        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList([nn.Conv2d(dim_in, out_channel, kernel_size=1, stride=1, padding=0) for out_channel in out_channels])

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
            assert len(in_shape) >= 3
            out_shape1 = out_shape
            out_shape2 = out_shape * 2 if expand else out_shape
            out_shape3 = out_shape * 4 if expand else out_shape
            out_shape4 = out_shape * 8 if expand else out_shape

            scratch = nn.Module()
            scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
            scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
            scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
            if len(in_shape) >= 4:
                scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
            return scratch

        self.scratch = _make_scratch(out_channels, features, expand=False)

        # Attach additional modules to scratch.
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet4 = FeatureFusionBlock(features, has_residual=False)

        head_features1 = features
        head_features2 = 32
        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(head_features1, head_features1, kernel_size=3, stride=1, padding=1)
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features1, head_features1 // 2, kernel_size=3, stride=1, padding=1)
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features1 // 2, head_features2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features2, dim_out, kernel_size=1, stride=1, padding=0),
            )

    def forward(
        self,
        aggregated_tokens_list: List[Tensor],
        images: Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        :param aggregated_tokens_list: List of token tensors from different transformer layers.
        :param images: Input images with shape [B, S, 3, H, W], in range [0, 1].
        :param patch_start_idx: Starting index for patch tokens in the token sequence. Used to separate patch tokens from other tokens (e.g., camera or register tokens).
        :param frames_chunk_size: Number of frames to process in each chunk. If None or larger than S, all frames are processed at once.
        :return: Tensor or Tuple[Tensor, Tensor]:
            - If feature_only=True: Feature maps with shape [B, S, C, H, W]
            - Otherwise: Tuple of (predictions, confidence) both with shape [B, S, 1, H, W]
        """
        B, S, _, H, W = images.shape
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_frame(aggregated_tokens_list, images, patch_start_idx)

        assert frames_chunk_size > 0
        all_points, all_confidence = [], []
        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            # Process batch of frames
            if self.feature_only:
                chunk_points = self._forward_frame(aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx)
                all_points.append(chunk_points)
            else:
                chunk_points, chunk_confidence = self._forward_frame(aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx)
                all_points.append(chunk_points)
                all_confidence.append(chunk_confidence)

        if self.feature_only:
            return torch.cat(all_points, dim=1)
        else:
            return torch.cat(all_points, dim=1), torch.cat(all_confidence, dim=1)

    def _forward_frame(
        self,
        aggregated_tokens_list: List[Tensor],
        images: Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Implementation of the forward pass through the DPT head. This method processes a specific chunk of frames from the sequence."""
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()
        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        out = []
        dpt_idx = 0

        def apply_pos_embed(x: Tensor, H: int, W: int, ratio: float = 0.1) -> Tensor:
            """Apply positional embedding to tensor x."""
            patch_h, patch_w = x.shape[-2], x.shape[-1]
            pos_embed = create_uv_grid(patch_h, patch_w, aspect_ratio=float(W) / float(H), dtype=x.dtype, device=x.device)
            pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
            pos_embed = pos_embed * ratio
            pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
            return x + pos_embed

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]
            x = x.reshape(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.premute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = apply_pos_embed(x, H, W)
            x = self.resize_layers[dpt_idx](x)
            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        out = self._forward_scratch(out)
        # Interpolate fused output to match target image resolution.
        size = (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio))
        out = custom_interpolate(out, size=size, mode="bilinear", align_corners=True)
        if self.pos_embed:
            out = apply_pos_embed(out, H, W)

        if self.feature_only:
            return out.view(B, S, *out.shape[1:])

        out = self.scratch.output_conv2(out)
        points, confidence = activate_head(out, self.point_act, self.confidence_act)
        return points.view(B, S, *points.shape[1:]), confidence.view(B, S, *confidence.shape[1:])

    def _forward_scratch(self, features: List[Tensor]) -> Tensor:
        """Forward pass through the fusion blocks."""
        layer1, layer2, layer3, layer4 = features
        layer1_rn = self.scratch.layer1_rn(layer1)
        layer2_rn = self.scratch.layer2_rn(layer2)
        layer3_rn = self.scratch.layer3_rn(layer3)
        layer4_rn = self.scratch.layer4_rn(layer4)
        out = self.scratch.refinenet4(layer4_rn, size=layer3_rn.shape[2:])
        del layer4_rn, layer4
        out = self.scratch.refinenet3(out, layer3_rn, size=layer2_rn.shape[2:])
        del layer3_rn, layer3
        out = self.scratch.refinenet2(out, layer2_rn, size=layer1_rn.shape[2:])
        del layer2_rn, layer2
        out = self.scratch.refinenet1(out, layer1_rn)
        del layer1_rn, layer1
        return self.scratch.output_conv1(out)


class ResidualConvUnit(nn.Module):
    def __init__(
        self,
        features : int,
        batch_norm : bool = False,
        act_function : nn.Module = nn.ReLU(inplace=True),
        groups : int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=groups)
        self.norm1 = nn.BatchNorm2d(features, eps=1e-5) if batch_norm else None
        self.norm2 = nn.BatchNorm2d(features, eps=1e-5) if batch_norm else None
        self.act_function = act_function
        self.skip_add = FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        r = self.act_function(x)
        r = self.conv1(r)
        if self.norm1 is not None:
            r = self.norm1(r)
        r = self.act_function(r)
        r = self.conv2(r)
        if self.norm2 is not None:
            r = self.norm2(r)
        return self.skip_add.add(r, x)


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        features : int,
        act_function : nn.Module = nn.ReLU(inplace=True),
        deconv : bool = False,
        batch_norm : bool = False,
        expand: bool = False,
        align_corners : bool = True,
        size : int = None,
        has_residual : bool = True,
        groups : int = 1,
    ) -> None:
        super().__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        self.size = size
        out_features = features // 2 if expand else features

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=groups)
        self.resConfUnit1 = ResidualConvUnit(features, batch_norm, act_function, groups) if has_residual else None
        self.resConfUnit2 = ResidualConvUnit(features, batch_norm, act_function, groups)
        self.skip_add = FloatFunctional()

    def forward(self, *xs, size=None) -> Tensor:
        x = xs[0]
        if self.resConfUnit1 is not None:
            r = self.resConfUnit1(xs[1])
            x = self.skip_add.add(x, r)
        x = self.resConfUnit2(x)

        if size is None and self.size is None:
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        x = custom_interpolate(x, **modifier, align_corners=self.align_corners)
        x = self.out_conv(x)
        return x


def custom_interpolate(
        x: Tensor,
        size: Tuple[int, int] = None,
        scale_factor: float = None,
        mode: str = "bilinear",
        align_corners: bool = True
) -> Tensor:
    """Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate."""
    INT_MAX = 1610612736
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [F.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)


def create_uv_grid(H: int, W: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> Tensor:
    """
    Create a normalized UV grid of shape (width, height, 2).
    :param H: Number of points vertically.
    :param W: Number of points horizontally.
    :param aspect_ratio: Width-to-height ratio. Defaults to width/height.
    :param dtype: Data type of the resulting tensor.
    :param device: Device on which the tensor is created.
    :return: A (width, height, 2) tensor of UV coordinates.
    """
    if aspect_ratio is None:
        aspect_ratio = float(W) / float(H)

    # Compute normalized spans for X and Y
    diag_factor = (aspect_ratio ** 2 + 1) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    # Establish the linespace boundaries
    left_x = -span_x * (W - 1) / W
    right_x = span_x * (W - 1) / W
    top_y = -span_y * (H - 1) / H
    bottom_y = span_y * (H - 1) / H

    # Generate 1D coordinates
    x = torch.linspace(left_x, right_x, steps=W, dtype=dtype, device=device)
    y = torch.linspace(top_y, bottom_y, steps=H, dtype=dtype, device=device)

    # Create 2D meshgrid (width x height) and stack into UV
    u, v = torch.meshgrid(x, y, indexing="xy")
    uv_grid = torch.stack((u, v), dim=-1)
    return uv_grid


def position_grid_to_embed(pos_grid: Tensor, embed_dim: int, omega: float = 100.0) -> Tensor:
    """Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)"""
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2, "Position grid must have 2 dimensions (x, y)"
    pos_flat = pos_grid.reshape(-1, grid_dim)  # (H*W)x2

    def make_sincos_pos_embed(pos: Tensor, embed_dim: int, omega: float = 100.0) -> Tensor:
        """This function generates a 1D positional embedding from a given grid using sine and cosine functions."""
        assert embed_dim % 2 == 0, "Embedding dimension must be even"
        device = pos.device
        Omega = torch.arange(embed_dim // 2, dtype=torch.float32 if device.type == "mps" else torch.double, device=device)
        Omega = Omega / (embed_dim / 2.0)
        Omega = 1.0 / (omega ** Omega)

        pos = pos.reshape(-1)
        out = torch.einsum("m,d->md", pos, Omega)
        embed_sin = torch.sin(out)
        embed_cos = torch.cos(out)
        embed = torch.cat((embed_sin, embed_cos), dim=1)
        return embed.float()

    embed_x = make_sincos_pos_embed(pos_flat[:, 0], embed_dim // 2, omega)
    embed_y = make_sincos_pos_embed(pos_flat[:, 1], embed_dim // 2, omega)
    embed = torch.cat((embed_x, embed_y), dim=1)
    return embed.view(H, W, embed_dim)
