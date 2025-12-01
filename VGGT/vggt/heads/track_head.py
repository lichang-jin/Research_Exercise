import torch
from torch import nn, Tensor
from dpt_head import DPTHead


class TrackHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x