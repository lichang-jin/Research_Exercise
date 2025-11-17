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
