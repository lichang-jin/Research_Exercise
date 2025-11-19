from torch import Tensor, nn

from typing import Callable, Optional, Tuple, Union


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size : Union[int, Tuple[int, int]] = 224,
        patch_size : Union[int, Tuple[int, int]] = 16,
        in_channels : int = 3,
        embed_dim : int = 768,
        norm_layer : Optional[Callable] = None,
        flatten_embedding : bool = True
    ) -> None:
        super().__init__()

        def make_2tuple(x):
            if isinstance(x, Tuple):
                assert len(x) == 2
                assert isinstance(x[0], int)
                assert isinstance(x[1], int)
                return x

            assert isinstance(x, int)
            return Tuple([x, x])

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1])

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patch_grid_size = patch_grid_size
        self.num_patches: int = patch_grid_size[0] * patch_grid_size[1]
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x : Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size
        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # (B, C, H, W)
        H, W = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # (B, H, W, C)
        return x

    def flops(self) -> float:
        flops = self.num_patches * (self.embed_dim * self.in_channels * self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += self.num_patches * self.embed_dim
        return flops

