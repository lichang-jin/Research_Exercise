import torch
from torch import Tensor
import torch.nn.functional as F


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    :param quaternions: Quaternions with real part last, as tensor of shape (..., 4) [i, j, k, r].
    :return: Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    matrix = torch.stack((
        1 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1 - two_s * (i * i + j * j),
    ), -1)
    matrix = matrix.reshape(quaternions.shape[:-1] + (3, 3))
    return matrix


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.
    :param matrix: Rotation matrices as tensor of shape (..., 3, 3).
    :return: Quaternions with real part last, as tensor of shape (..., 4) [i, j, k, r].
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape f{matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    def _sqrt_position_part(x: Tensor) -> Tensor:
        """Returns torch.sqrt(torch.max(0, x)), but with a zero subgradient where x is 0."""
        ret = torch.zeros_like(x)
        position_mask = x > 0
        if torch.is_grad_enabled():
            ret[position_mask] = torch.sqrt(x[position_mask])
        else:
            ret = torch.where(position_mask, torch.sqrt(x), ret)
        return ret

    q_abs = _sqrt_position_part(torch.stack( # [4r^2, 4i^2, 4j^2, 4k^2]
        [1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1))
    q = torch.stack([ # 假设不同的分量 [|r|, |i|, |j|, |k|] 最大
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m21 + m12], dim=-1),
        torch.stack([m10 - m01, m02 + m20, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)  # r, i, j, k
    # We floor here at 0.1 but the exact level is not important; if q_abs is small, the candidate won't be picked.
    floor = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    q_candidates = q / (2.0 * q_abs[..., None].max(floor))
    # if not for numerical problems, q_candidates[i] should be same (up to a sign), forall i; we pick the best-conditioned one (with the largest denominator)
    quaternions = q_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))
    quaternions = quaternions[..., [1, 2, 3, 0]]  # r, i, j, k -> i, j, k, r

    def _standardize_quaternion(quaternion: Tensor) -> Tensor:
        """Convert a unit quaternion to a standard form: one in which the real part is non-negative."""
        return torch.where(quaternion[..., 3:4] < 0, -quaternion, quaternion)

    quaternions = _standardize_quaternion(quaternions)
    return quaternions



