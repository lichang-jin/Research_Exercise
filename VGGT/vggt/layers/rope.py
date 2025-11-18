import torch
from torch import Tensor, nn
from torch.nn import functional as F

from typing import Dict, Tuple


class PositionGetter:
    def __init__(self):
        self.position_cache : Dict[Tuple[int, int], Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> Tensor:
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[(height, width)] = positions

        cached_positions = self.position_cache[(height, width)]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[Tensor, Tensor]] = {}

    def _compute_frequency_components(self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # Compute frequency bands
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_frequency = 1.0 / (self.base_frequency ** exponents)

            # Generate position-dependent frequencies
            positions = torch.arange(seq_len, device=device, dtype=inv_frequency.dtype)
            angles = torch.einsum("i, j -> i j", positions, inv_frequency).to(dtype)

            # Compute and cache frequency components
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = torch.cos(angles).to(dtype)
            sin_components = torch.sin(angles).to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: Tensor) -> Tensor:
        dim = x.shape[-1]
        x1, x2 = x[..., :dim // 2], x[..., dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(self, tokens: Tensor, positions: Tensor, cos_components: Tensor, sin_components: Tensor) -> Tensor:
        cos = F.embedding(positions, cos_components)[:, None, :, :]
        sin = F.embedding(positions, sin_components)[:, None, :, :]
        return tokens * cos + self._rotate_features(tokens) * sin

    def forward(self, tokens: Tensor, positions: Tensor) -> Tensor:
        assert tokens.shape[-1] % 2 == 0, "Feature dimension must be even."
        assert positions.ndim == 3 and positions.shape[-1] == 2, "Positions must have shape (batch_size, n_tokens, 2)."

        feature_dim = tokens.shape[-1] // 2
        max_position = int(positions.max()) + 1
        cos_components, sin_components = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)

        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_components, sin_components)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_components, sin_components)
        return torch.cat((vertical_features, horizontal_features), dim=-1)
