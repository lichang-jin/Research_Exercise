import torch
from torch import Tensor, nn

from ..layers.block import Block
from ..layers import MLP
from ..heads.head_act import activate_pose


class CameraHead(nn.Module):
    def __init__(
        self,
        dim : int = 2048,
        trunk_depth : int = 4,
        pose_encoding_type : str = "absT_quaR_FoV",
        num_heads : int = 16,
        mlp_ratio : float = 4.0,
        init_values : float = 1e-2,
        translation_act : str = "linear",
        quaternion_act : str = "linear",
        focal_length_act : str = "relu",
    ) -> None:
        super().__init__()
        self.translation_act = translation_act
        self.quaternion_act = quaternion_act
        self.focal_length_act = focal_length_act
        self.trunk_depth = trunk_depth
        self.trunk = nn.Sequential(*[Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values) for _ in range(trunk_depth)])

        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.token_norm = nn.LayerNorm(dim)
        self.trunk_norm = nn.LayerNorm(dim)

        # Learnable empty camera pose token.
        self.empty_pose_token = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, dim)
        # Module for producing modulation parameters: shift, scale, and a gate.
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim, bias=True))
        # Adaptive layer normalization without affine parameters.
        self.adaptive_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.pose_branch = MLP(in_features=dim, hidden_features=dim // 2, out_features=self.target_dim, drop=0.0)

    def forward(self, aggregated_token_list: list, num_iterations: int = 4) -> list:
        tokens = aggregated_token_list[-1]  # Use tokens from the last block for camera prediction.
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)
        pred_pose_list = self.trunk_fn(pose_tokens, num_iterations)
        return pred_pose_list

    def trunk_fn(self, pose_tokens: Tensor, num_iterations: int) -> list:
        B, S, C = pose_tokens.shape
        pred_pose = None
        pred_pose_list = []

        for _ in range(num_iterations):
            if pred_pose is None:
                module_input = self.embed_pose(self.empty_pose_token.expand(B, S, -1))
            else:
                pred_pose = pred_pose.detach()
                module_input = self.embed_pose(pred_pose)

            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)
            pose_tokens_modulated = gate_msa * (self.adaptive_norm(pose_tokens) * (1 + scale_msa) + shift_msa) + pose_tokens
            pose_tokens_modulated = self.trunk(pose_tokens_modulated)
            pred_pose_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))
            if pred_pose is None:
                pred_pose = pred_pose_delta
            else:
                pred_pose = pred_pose + pred_pose_delta

            activated_pose = activate_pose(pred_pose, self.translation_act, self.quaternion_act, self.focal_length_act)
            pred_pose_list.append(activated_pose)

        return pred_pose_list




