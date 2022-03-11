""" 
@Date: 2021/11/06
@description:
"""
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.conversion import depth2xyz


def convert_img(value, h, need_nor=True, cmap=None):
    value = value.clone().detach().cpu().numpy()[None]
    if need_nor:
        value -= value.min()
        value /= value.max() - value.min()
    grad_img = value.repeat(int(h), axis=0)

    if cmap is None:
        grad_img = grad_img[..., np.newaxis].repeat(3, axis=-1)
    elif cmap == cv2.COLORMAP_PLASMA:
        grad_img = cv2.applyColorMap((grad_img * 255).astype(np.uint8), colormap=cmap)
        grad_img = grad_img[..., ::-1]
        grad_img = grad_img.astype(np.float) / 255.0
    elif cmap == 'HSV':
        grad_img = np.round(grad_img * 1000) / 1000.0
        grad_img = grad_img[..., np.newaxis].repeat(3, axis=-1)
        grad_img[..., 0] = grad_img[..., 0] * 180
        grad_img[..., 1] = 255
        grad_img[..., 2] = 255
        grad_img = grad_img.astype(np.uint8)
        grad_img = cv2.cvtColor(grad_img, cv2.COLOR_HSV2RGB)
        grad_img = grad_img.astype(np.float) / 255.0
    return grad_img


def show_grad(depth, grad_conv, h=5, show=False):
    """
    :param h:
    :param depth: [patch_num]
    :param grad_conv:
    :param show:
    :return:
    """

    direction, angle, grad = get_all(depth[None], grad_conv)

    # depth_img = convert_img(depth, h)
    # angle_img = convert_img(angle[0], h)
    # grad_img = convert_img(grad[0], depth.shape[-1] // 4 - h * 2)
    depth_img = convert_img(depth, h, cmap=cv2.COLORMAP_PLASMA)
    angle_img = convert_img(angle[0], h, cmap='HSV')

    # vis_grad = grad[0] / grad[0].max() / 2 + 0.5
    grad_img = convert_img(grad[0], h)
    img = np.concatenate([depth_img, angle_img, grad_img], axis=0)
    if show:
        plt.imshow(img)
        plt.show()
    return img


def get_grad(direction):
    """
    :param direction: [b patch_num]
    :return:[b patch_num]
    """
    a = torch.roll(direction, -1, dims=1)  # xz[i+1]
    b = torch.roll(direction, 1, dims=1)  # xz[i-1]
    grad = torch.acos(torch.clip(a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1], -1+1e-6, 1-1e-6))
    return grad


def get_grad2(angle, grad_conv):
    """
    :param angle: [b patch_num]
    :param grad_conv:
    :return:[b patch_num]
    """
    angle = torch.sin(angle)
    angle = angle + 1

    angle = torch.cat([angle[..., -1:], angle, angle[..., :1]], dim=-1)
    grad = grad_conv(angle[:, None])  # [b, patch_num] -> [b, 1, patch_num]
    # grad = torch.abs(grad)
    return grad.reshape(angle.shape[0], -1)


def get_edge_angle(direction):
    """
    :param direction: [b patch_num 2]
    :return:
    """
    angle = torch.atan2(direction[..., 1], direction[..., 0])
    return angle


def get_edge_direction(depth):
    xz = depth2xyz(depth)[..., ::2]
    direction = torch.roll(xz, -1, dims=1) - xz  # direct[i] = xz[i+1] - xz[i]
    direction = direction / direction.norm(p=2, dim=-1)[..., None]
    return direction


def get_all(depth, grad_conv):
    """

    :param grad_conv:
    :param depth: [b patch_num]
    :return:
    """
    direction = get_edge_direction(depth)
    angle = get_edge_angle(direction)
    # angle_grad = get_grad(direction)
    angle_grad = get_grad2(angle, grad_conv)  # signed gradient
    return direction, angle, angle_grad
