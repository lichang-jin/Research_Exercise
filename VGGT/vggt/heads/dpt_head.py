import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.quantized import FloatFunctional

from typing import List, Tuple

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

        def custom_interpolate(
            x: Tensor,
            size: Tuple[int, int] = None,
            scale_factor: float = None,
            mode: str = "bilinear",
            align_corners: bool = True
        ) -> Tensor:
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

        x = custom_interpolate(x, **modifier, align_corners=self.align_corners)
        x = self.out_conv(x)
        return x

