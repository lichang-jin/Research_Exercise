import numpy as np
import torch
from torch import Tensor
from typing import Union, Tuple

ArrayLike = Union[np.ndarray, Tensor]


def _ensure_tensor(x: ArrayLike) -> Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, Tensor):
        return x
    else:
        return torch.tensor(x)


def signal_undistortion(params: ArrayLike, track_normalized: ArrayLike) -> Tensor:
    """
    Apply undistortion to the normalized tracks using the given distortion parameters once.
    :param params: Distortion parameters of shape BxN.
    :param track_normalized: Normalized tracks tensor of shape [batch_size, num_tracks, 2].
    :return: Undistorted normalized tracks tensor.
    """
    params, track_normalized = _ensure_tensor(params), _ensure_tensor(track_normalized)
    u, v = track_normalized[..., 0].clone(), track_normalized[..., 1].clone()
    u_undistorted, v_undistorted = apply_distortion(params, u, v)
    return torch.stack([u_undistorted, v_undistorted], dim=-1)


def iterative_undistortion(params: ArrayLike, track_normalized: ArrayLike, max_iterations: int = 100, max_step_norm: float = 1e-10, rel_step_size: float = 1e-6) -> Tensor:
    """
    Iteratively undistort the normalized tracks using the given distortion parameters.
    :param params: Distortion parameters of shape BxN.
    :param track_normalized: Normalized tracks tensor of shape [batch_size, num_tracks, 2].
    :param max_iterations: Maximum number of iterations for the undistortion process.
    :param max_step_norm: Maximum step norm for convergence.
    :param rel_step_size: Relative step size for numerical differentiation.
    :return: Undistorted normalized tracks tensor.
    """
    params, track_normalized = _ensure_tensor(params), _ensure_tensor(track_normalized)
    B, N = track_normalized.shape[:2]
    u, v = track_normalized[..., 0].clone(), track_normalized[..., 1].clone()
    origin_u, origin_v = u.clone(), v.clone()

    eps = torch.finfo(u.dtype).eps
    for idx in range(max_iterations):
        u_undistorted, v_undistorted = apply_distortion(params, u, v)
        du, dv = origin_u - u_undistorted, origin_v - v_undistorted
        step_u, step_v = torch.clamp(torch.abs(u) * rel_step_size, min=eps), torch.clamp(torch.abs(v) * rel_step_size, min=eps)
        J_00 = (apply_distortion(params, u + step_u, v)[0] - apply_distortion(params, u - step_u, v)[0]) / (2 * step_u)
        J_01 = (apply_distortion(params, u, v + step_v)[0] - apply_distortion(params, u, v - step_v)[0]) / (2 * step_v)
        J_10 = (apply_distortion(params, u + step_u, v)[1] - apply_distortion(params, u - step_u, v)[1]) / (2 * step_u)
        J_11 = (apply_distortion(params, u, v + step_v)[1] - apply_distortion(params, u, v - step_v)[1]) / (2 * step_v)
        J = torch.stack([torch.stack([J_00 + 1, J_01], dim=-1), torch.stack([J_10, J_11 + 1], dim=-1)], dim=-2)
        delta = torch.linalg.solve(J, torch.stack([du, dv], dim=-1))
        u, v = u + delta[..., 0], v + delta[..., 1]

        if torch.max((delta ** 2).sum(dim=-1)) < max_step_norm:
            break
    return torch.stack([u, v], dim=-1)


def apply_distortion(extra_params, u, v) -> Tuple[Tensor, Tensor]:
    """
    Applies radial or OpenCV distortion to the given 2D points.
    :param extra_params: Distortion parameters of shape BxN, where N can be 1, 2, or 4.
    :param u: Normalized x coordinates of shape Bxn(num_tracks).
    :param v: Normalized y coordinates of shape Bxn(num_tracks).
    :return: Distorted 2D points of shape BxNx2.
    """
    extra_params, u, v = _ensure_tensor(extra_params), _ensure_tensor(u), _ensure_tensor(v)
    u2, v2, uv = u * u, v * v, u * v
    r2 = u2 + v2

    num_params = extra_params.shape[1]
    if num_params == 1:
        # Simple radial distortion
        k = extra_params[:, 0]
        radial = k[:, None] * r2
        du, dv = u * radial, v * radial
    elif num_params == 2:
        # RadialCameraModel distortion
        k1, k2 = extra_params[:, 0], extra_params[:, 1]
        radial = k1[:, None] * r2 + k2[:, None] * r2 * r2
        du, dv = u * radial, v * radial
    elif num_params == 4:
        # OpenCVCameraModel distortion
        k1, k2, p1, p2 = extra_params[:, 0], extra_params[:, 1], extra_params[:, 2], extra_params[:, 3]
        radial = k1[:, None] * r2 + k2[:, None] * r2 * r2
        du = u * radial + 2 * p1[:, None] * uv + p2[:, None] * (r2 + 2 * u2)
        dv = v * radial + 2 * p2[:, None] * uv + p1[:, None] * (r2 + 2 * v2)
    else:
        raise ValueError(f"Invalid number of distortion parameters: {num_params}")

    return u.clone() + du, v.clone() + dv
