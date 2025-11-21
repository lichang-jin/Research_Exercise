import torch
from torch import Tensor
import torch.nn.functional as F


def activate_pose(pred_pose, translation_act = "linear", quaternion_act = "linear", focal_length_act = "linear"):
    """
    :param pred_pose: Tensor containing encoded pose parameters [translation, quaternion, focal length]
    :param translation_act: Activation type for translation component
    :param quaternion_act: Activation type for quaternion component
    :param focal_length_act: Activation type for focal length component
    :return: Activated pose parameters tensor
    """
    translation_component = pred_pose[:, :3]
    quaternion_component = pred_pose[:, 3:7]
    focal_length_component = pred_pose[:, 7:]

    translation = base_pose_act(translation_component, translation_act)
    quaternion = base_pose_act(quaternion_component, quaternion_act)
    focal_length = base_pose_act(focal_length_component, focal_length_act)
    return torch.cat([translation, quaternion, focal_length], dim=-1)


def base_pose_act(pose, act_type = "linear"):
    """
    Apply basic activation function to pose parameters.
    :param pose: Tensor containing encoded pose parameters
    :param act_type: Activation type ("linear", "inv_log", "exp", "relu")
    :return: Activated pose parameters
    """
    if act_type == "linear":
        return pose
    elif act_type == "inv_log":
        return _Math_inverse_log_transform(pose)
    elif act_type == "exp":
        return torch.exp(pose)
    elif act_type == "relu":
        return F.relu(pose)
    else:
        raise ValueError(f"Invalid activation type: {act_type}")


def activate_head(output: Tensor, point_act = "norm_exp", confidence_act = "exp1"):
    """
    Process network output to extract 3D points and confidence values.
    :param output: Network output tensor (B, C, H, W)
    :param point_act: Activation type for 3D points ("norm_exp", "norm", "exp", "relu", "inv_log", "xy_inv_log", "sigmoid", "linear")
    :param confidence_act: Activation type for confidence values ("exp1", "exp0", "sigmoid", "linear")
    :return: 3D points tensor, confidence tensor
    """
    fmap = output.permute(0, 2, 3, 1)      # (B, H, W, C)
    xyz = fmap[:, :, :, :-1]               # (B, H, W, C-1)
    confidence = fmap[:, :, :, -1]         # (B, H, W, 1)

    if point_act == "norm_exp":
        d = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        points = xyz / d * torch.expm1(d)
    elif point_act == "norm":
        points = xyz / xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    elif point_act == "exp":
        points = torch.exp(xyz)
    elif point_act == "relu":
        points = F.relu(xyz)
    elif point_act == "inv_log":
        points = _Math_inverse_log_transform(xyz)
    elif point_act == "xy_inv_log":
        xy, z = xyz.split([2, 1], dim=-1)
        z = _Math_inverse_log_transform(z)
        points = torch.cat([xy * z, z], dim=-1)
    elif point_act == "sigmoid":
        points = torch.sigmoid(xyz)
    elif point_act == "linear":
        points = xyz
    else:
        raise ValueError(f"Invalid point activation type: {point_act}")

    if confidence_act == "exp1":
        confidence = 1 + torch.exp(confidence)
    elif confidence_act == "exp0":
        confidence = torch.exp(confidence)
    elif confidence_act == "sigmoid":
        confidence = torch.sigmoid(confidence)
    elif confidence_act == "linear":
        confidence = confidence
    else:
        raise ValueError(f"Invalid confidence activation type: {confidence_act}")

    return points, confidence


def _Math_inverse_log_transform(x):
    """f(x) = sign(x) * (exp(|x|) - 1)"""
    return torch.sign(x) * (torch.expm1(torch.abs(x)))