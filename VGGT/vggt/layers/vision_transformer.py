import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torch.nn.init import trunc_normal_
from . import MLP, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block

from functools import partial
import math
import logging
from typing import Tuple, Sequence, Callable, Union


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
