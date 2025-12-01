import torch.cuda.amp
from torch import nn, Tensor
from huggingface_hub import PyTorchModelHubMixin

from ..heads import CameraHead, DPTHead, TrackHead
from aggregator import Aggregator


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        image_size : int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        enable_camera : bool = True,
        enable_point : bool = True,
        enable_depth : bool = True,
        enable_track : bool = True,
    ) -> None:
        super().__init__()

        self.aggregator = Aggregator(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        self.camera_head = CameraHead(
            dim=2 * embed_dim,
        ) if enable_camera else None

        self.point_head = DPTHead(
            dim_in=2 * embed_dim,
            dim_out=4,
            point_act="inv_log",
            confidence_act="exp1",
        ) if enable_point else None

        self.depth_head = DPTHead(
            dim_in=2 * embed_dim,
            dim_out=2,
            point_act="exp",
            confidence_act="exp1",
        ) if enable_depth else None

        self.track_head = TrackHead(

        ) if enable_track else None

    def forward(self, images: Tensor, query_points: Tensor = None):
        """
        Forward pass of the VGGT model.
        :param images: Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
        :param query_points: Query points for tracking, in pixel coordinates. Shape: [N, 2] or [B, N, 2], where N is the number of query points.
        :return:
            dict: A dictionary containing the following predictions:
                - pose_enc: Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth: Predicted depth maps with shape [B, S, H, W, 1].
                - depth_confidence: Confidence scores for depth predictions with shape [B, S, H, W].
                - world_points: 3D world coordinates for each pixel with shape [B, S, H, W, 3].
                - world_points_confidence: Confidence scores for world points with shape [B, S, H, W].
                - images: Original input images, preserved for visualization.
                If query_points is provided, also includes:
                - track: Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates.
                - visibility: Visibility scores for tracked points with shape [B, S, N].
                - track_confidence: Confidence scores for tracked points with shape [B, S, N].
        """
        if len(images.shape) == 4:
            # If without batch dimension, add it.
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            # If without batch dimension, add it.
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        predictions = {}
        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_confidence = self.depth_head(aggregated_tokens_list, images, patch_start_idx)
                predictions["depth"] = depth
                predictions["depth_confidence"] = depth_confidence

            if self.point_head is not None:
                points, points_confidence = self.point_head(aggregated_tokens_list, images, patch_start_idx)
                predictions["world_points"] = points
                predictions["world_points_confidence"] = points_confidence

        if self.track_head is not None and query_points is not None:
            track_list, visibility, track_confidence = self.track_head(aggregated_tokens_list, images, patch_start_idx, query_points)
            predictions["track"] = track_list[-1]
            predictions["visibility"] = visibility
            predictions["track_confidence"] = track_confidence

        if not self.training:
            predictions["images"] = images

        return predictions