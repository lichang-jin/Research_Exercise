import torch
from torch import Tensor
import numpy as np

from distortion import apply_distortion, signal_undistortion, iterative_undistortion


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray,
    camera_extrinsic: np.ndarray,
    camera_intrinsic: np.ndarray,
) -> np.ndarray:
    """
    Unproject a batch of depth maps to 3D world coordinates.
    :param depth_map: Batch of depth maps of shape (S, H, W, 1) or (S, H, W).
    :param camera_extrinsic: Batch of camera extrinsic matrices of shape (S, 3, 4).
    :param camera_intrinsic: Batch of camera intrinsic matrices of shape (S, 3, 3).
    :return: Batch of 3D world coordinates of shape (S, H, W, 3).
    """
    if isinstance(depth_map, Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(camera_extrinsic, Tensor):
        camera_extrinsic = camera_extrinsic.cpu().numpy()
    if isinstance(camera_intrinsic, Tensor):
        camera_intrinsic = camera_intrinsic.cpu().numpy()

    def depth_to_camera_coords_points(depth_map_, camera_intrinsic_) -> np.ndarray:
        """
        Convert a depth map to camera coordinates.
        :param depth_map_: Depth map of shape (H, W).
        :param camera_intrinsic_: Camera intrinsic matrix of shape (3, 3).
        :return: Camera coordinates (H, W, 3).
        """
        H, W = depth_map_.shape
        assert camera_intrinsic_.shape == (3, 3), f"Intrinsic matrix must be shape (3, 3), got {camera_intrinsic_.shape}"
        assert camera_intrinsic_[0, 1] == 0 and camera_intrinsic_[1, 0] == 0, "Intrinsic matrix must have zero skew"
        fx, fy = camera_intrinsic_[0, 0], camera_intrinsic_[1, 1]
        u0, v0 = camera_intrinsic_[0, 2], camera_intrinsic_[1, 2]
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        x = (u - u0) * depth_map_ / fx
        y = (v - v0) * depth_map_ / fy
        z = depth_map_
        return np.stack((x, y, z), axis=-1).astype(np.float32)

    def depth_to_world_coords_points(depth_map_, camera_extrinsic_, camera_intrinsic_, eps=1e-8):
        """
        Convert a depth map to world coordinates.
        :param depth_map_: Depth map of shape (H, W).
        :param camera_extrinsic_: Camera intrinsic matrix of shape (3, 3).
        :param camera_intrinsic_: Camera extrinsic matrix of shape (3, 4).
        :param eps: Default 1e-8.
        :return: World coordinates (H, W, 3), camera coordinates (H, W, 3) and valid depth mask (H, W).
        """
        if depth_map_ is None:
            return None, None, None
        point_mask = depth_map_ > eps
        camera_coords_points = depth_to_camera_coords_points(depth_map_, camera_intrinsic_)
        camera_to_world_extrinsic = closed_form_inverse_se3(camera_extrinsic_[None])[0]
        R_camera_to_world = camera_to_world_extrinsic[:3, :3]
        T_camera_to_world = camera_to_world_extrinsic[:3, 3:]
        world_coords_points = np.dot(camera_coords_points, R_camera_to_world.T) + T_camera_to_world
        return world_coords_points, camera_coords_points, point_mask

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        world_points, _, _ = depth_to_world_coords_points(depth_map[frame_idx].squeeze(-1), camera_extrinsic[frame_idx], camera_intrinsic[frame_idx])
        world_points_list.append(world_points)
    return np.stack(world_points_list, axis=0)


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.
    If `R` and `T` are provided, they must correspond to the rotation and translation components of `se3`.
    Otherwise, they will be extracted from `se3`.
    :param se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
    :param R: Nx3x3 array or tensor of rotation matrices.
    :param T: Nx3x1 array or tensor of translation vectors.
    :return: Inverted SE3 matrices with the same type and device as `se3`.
    """
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be shape (N, 4, 4) or (N, 3, 4), got {se3.shape}")

    R = se3[..., :3, :3] if R is None else R
    T = se3[..., :3, 3:] if T is None else T

    if isinstance(se3, np.ndarray):
        R_T = np.transpose(R, axes=(0, 2, 1))
        top_right = np.matmul(-R_T, T)
        inv_se3 = np.tile(np.eye(4), (len(R), 1, 1))
    elif isinstance(se3, Tensor):
        R_T = R.transpose(1, 2)
        top_right = torch.bmm(-R_T, T)
        inv_se3 = (torch.eye(4, 4)[None].repeat(len(R), 1, 1)).to(R.dtype).to(R.device)
    else:
        raise TypeError(f"se3 must be a numpy array or a torch tensor, got {type(se3)}")

    inv_se3[..., :3, :3] = R_T
    inv_se3[..., :3, 3:] = top_right
    return inv_se3


def project_world_points_to_camera(
    world_points: Tensor,
    camera_extrinsic: Tensor,
    camera_intrinsic: Tensor = None,
    distortion_params: Tensor = None,
    default: float = 0,
    only_points_camera: bool = False,
):
    """
    Transforms 3D points to 2D using extrinsic and intrinsic parameters.
    :param world_points: 3D points of shape Px3.
    :param camera_extrinsic: Extrinsic parameters of shape Bx3x4.
    :param camera_intrinsic: Intrinsic parameters of shape Bx3x3.
    :param distortion_params: Extra parameters of shape BxN, which is used for radial distortion.
    :param default: Default value to replace NaNs in the output.
    :param only_points_camera: Default False.
    :return: Transformed 2D points of shape BxNx2.
    """
    device = world_points.device
    with torch.autocast(device.type, enabled=False):
        B = camera_extrinsic.shape[0]
        world_points = torch.cat([world_points, torch.ones_like(world_points[:, :1])], dim=1)
        world_points = world_points.unsqueeze(0).expand(B, -1, -1)
        world_points_T = world_points.transpose(-1, -2)
        camera_points = torch.bmm(camera_extrinsic, world_points_T)

        if only_points_camera:
            image_points = None
        else:
            image_points = image_points_from_camera_points(camera_points, camera_intrinsic, distortion_params, default)
        return image_points, camera_points


def image_points_from_camera_points(camera_points: Tensor, camera_intrinsic: Tensor, distortion_params: Tensor = None, default: float = 0.0) -> Tensor:
    """
    Applies intrinsic parameters and optional distortion to the given 3D points.
    :param camera_points: 3D points in camera coordinates of shape Bx3xN.
    :param camera_intrinsic: Intrinsic camera parameters of shape Bx3x3.
    :param distortion_params: Distortion parameters of shape BxN, where N can be 1, 2, or 4.
    :param default: Default value to replace NaNs in the output.
    :return: pixel_coords: 2D points in pixel coordinates of shape BxNx2.
    """
    camera_points = camera_points / camera_points[:, 2:3, :]
    ndc_xy = camera_points[:, :2, :]

    if distortion_params is not None:
        x_distorted, y_distorted = apply_distortion(distortion_params, ndc_xy[:, 0], ndc_xy[:, 1])
        xy_distorted = torch.stack([x_distorted, y_distorted], dim=1)
    else:
        xy_distorted = ndc_xy

    camera_coords = torch.cat((xy_distorted, torch.ones_like(xy_distorted[:, :1, :])), dim=1)
    pixel_coords = torch.bmm(camera_intrinsic, camera_coords)[:, :2, :]
    pixel_coords = torch.nan_to_num(pixel_coords, nan=default)
    return pixel_coords.transpose(1, 2)


def camera_points_from_image_points(pred_tracks: Tensor, camera_intrinsic: Tensor, extra_params: Tensor = None) -> Tensor:
    """
    Normalize predicted tracks based on camera intrinsics.
    :param pred_tracks: The predicted tracks tensor of shape [batch_size, num_tracks, 2].
    :param camera_intrinsic: The camera intrinsics tensor of shape [batch_size, 3, 3].
    :param extra_params: Distortion parameters of shape BxN, where N can be 1, 2, or 4.
    :return: Normalized tracks tensor.
    """
    principal_point = camera_intrinsic[:, [0, 1], [2, 2]].unsqueeze(-2) # (u, v)
    focal_length = camera_intrinsic[:, [0, 1], [0, 1]].unsqueeze(-2)    # (fx, fy)
    normalized_tracks = (pred_tracks - principal_point) / focal_length

    if extra_params is not None:
        try:
            normalized_tracks = iterative_undistortion(extra_params, normalized_tracks)
        except:
            normalized_tracks = signal_undistortion(extra_params, normalized_tracks)

    return normalized_tracks
