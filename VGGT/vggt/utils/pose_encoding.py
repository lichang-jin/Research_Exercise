import torch
from rotation import quaternion_to_matrix, matrix_to_quaternion


def camera_param_to_pose_encoding(extrinsic, intrinsic, image_size=None, pose_encoding_type = "absT_quaR_FoV"):
    """
    Convert camera extrinsic and intrinsic to a compact pose encoding.
    :param extrinsic: Camera extrinsic matrix of shape (B, S, 3, 4).
    :param intrinsic: Camera intrinsic matrix of shape (B, S, 3, 3).
    :param image_size: Tuple of image size (H, W).
    :param pose_encoding_type: Type of pose encoding, "absT_quaR_FoV".
    :return: Encoded camera pose parameters with shape (B, S, 9).
        - [:3] = absolute translation vector T (3D)
        - [3:7] = rotation as quaternion quat (4D)
        - [7:] = field of view (2D)
    """
    if pose_encoding_type == "absT_quaR_FoV":
        R = extrinsic[:, :, :3, :3]
        T = extrinsic[:, :, :3, 3]
        quaternion = matrix_to_quaternion(R)
        H, W = image_size
        FoV_H = 2 * torch.atan((H / 2) / intrinsic[..., 1, 1])
        FoV_W = 2 * torch.atan((W / 2) / intrinsic[..., 0, 0])
        pose_encoding = torch.cat([T, quaternion, FoV_H[..., None], FoV_W[..., None]], dim=-1).float()
    else:
        raise NotImplementedError

    return pose_encoding


def pose_encoding_to_camera_param(pose_encoding, image_size=None, pose_encoding_type="absT_quaR_FoV", build_intrinsics=True):
    """
    Convert a pose encoding back to camera extrinsic and intrinsic.
    :param pose_encoding: Encoded camera pose parameters with shape (B, S, 9).
        - [:3] = absolute translation vector T (3D)
        - [3:7] = rotation as quaternion quat (4D)
        - [7:] = field of view (2D)
    :param image_size: Tuple of image size (H, W).
    :param pose_encoding_type: Type of pose encoding, "absT_quaR_FoV".
    :param build_intrinsics: Whether to reconstruct the intrinsics matrix, if False, the intrinsics matrix will be None.
    :return: Tuple: (extrinsic, intrinsic)
    """
    intrinsic = None

    if pose_encoding_type == "absT_quaR_FoV":
        T = pose_encoding[..., :3]
        quaternion = pose_encoding[..., 3:7]
        FoV_H = pose_encoding[..., 7]
        FoV_W = pose_encoding[..., 8]
        R = quaternion_to_matrix(quaternion)
        extrinsic = torch.cat([R, T[..., None]], dim=-1)

        if build_intrinsics:
            H, W = image_size
            fx = (W / 2.0) / torch.tan(FoV_W / 2.0)
            fy = (H / 2.0) / torch.tan(FoV_H / 2.0)
            intrinsic = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
            intrinsic[..., 0, 0] = fx
            intrinsic[..., 1, 1] = fy
            intrinsic[..., 0, 2] = W / 2.0
            intrinsic[..., 1, 2] = H / 2.0
            intrinsic[..., 2, 2] = 1.0
    else:
        raise NotImplementedError

    return extrinsic, intrinsic