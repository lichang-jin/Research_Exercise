import torch
from torch import Tensor
import torch.nn.functional as F


def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    :param quaternions: Quaternions with real part last, as tensor of shape (..., 4).
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
    :return: Quaternions with real part last, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape f{matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = torch.stack((