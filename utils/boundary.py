""" 
@date: 2021/06/19
@description:
"""
import math
import functools

from scipy import stats
from scipy.ndimage.filters import maximum_filter
import numpy as np
from typing import List
from utils.conversion import uv2xyz, xyz2uv, depth2xyz, uv2pixel, depth2uv, pixel2uv, xyz2pixel, uv2lonlat
from utils.visibility_polygon import calc_visible_polygon


def connect_corners_uv(uv1: np.ndarray, uv2: np.ndarray, length=256) -> np.ndarray:
    """
    :param uv1: [u, v]
    :param uv2: [u, v]
    :param length: Fix the total length in pixel coordinates
    :return:
    """
    # why -0.5? Check out the uv2Pixel function
    p_u1 = uv1[0] * length - 0.5
    p_u2 = uv2[0] * length - 0.5

    if abs(p_u1 - p_u2) < length / 2:
        start = np.ceil(min(p_u1, p_u2))
        p = max(p_u1, p_u2)
        end = np.floor(p)
        if end == np.ceil(p):
            end = end - 1
    else:
        start = np.ceil(max(p_u1, p_u2))
        p = min(p_u1, p_u2) + length
        end = np.floor(p)
        if end == np.ceil(p):
            end = end - 1
    p_us = (np.arange(start, end + 1) % length).astype(np.float64)
    if len(p_us) == 0:
        return None
    us = (p_us + 0.5) / length  # why +0.5? Check out the uv2Pixel function

    plan_y = boundary_type(np.array([uv1, uv2]))
    xyz1 = uv2xyz(np.array(uv1), plan_y)
    xyz2 = uv2xyz(np.array(uv2), plan_y)
    x1 = xyz1[0]
    z1 = xyz1[2]
    x2 = xyz2[0]
    z2 = xyz2[2]

    d_x = x2 - x1
    d_z = z2 - z1

    lon_s = (us - 0.5) * 2 * np.pi
    k = np.tan(lon_s)
    ps = (k * z1 - x1) / (d_x - k * d_z)
    cs = np.sqrt((z1 + ps * d_z) ** 2 + (x1 + ps * d_x) ** 2)

    lats = np.arctan2(plan_y, cs)
    vs = lats / np.pi + 0.5
    uv = np.stack([us, vs], axis=-1)

    if start == end:
        return uv[0:1]
    return uv


def connect_corners_xyz(uv1: np.ndarray, uv2: np.ndarray, step=0.01) -> np.ndarray:
    """
    :param uv1: [u, v]
    :param uv2: [u, v]
    :param step: Fixed step size in xyz coordinates
    :return:
    """
    plan_y = boundary_type(np.array([uv1, uv2]))
    xyz1 = uv2xyz(np.array(uv1), plan_y)
    xyz2 = uv2xyz(np.array(uv2), plan_y)

    vec = xyz2 - xyz1
    norm = np.linalg.norm(vec, ord=2)
    direct = vec / norm
    xyz = np.array([xyz1 + direct * dis for dis in np.linspace(0, norm, int(norm / step))])
    if len(xyz) == 0:
        xyz = np.array([xyz2])
    uv = xyz2uv(xyz)
    return uv


def connect_corners(uv1: np.ndarray, uv2: np.ndarray, step=0.01, length=None) -> np.ndarray:
    """
    :param uv1: [u, v]
    :param uv2: [u, v]
    :param step:
    :param length:
    :return: [[u1, v1], [u2, v2]....] if length!=None，length of return result = length
    """
    if length is not None:
        uv = connect_corners_uv(uv1, uv2, length)
    elif step is not None:
        uv = connect_corners_xyz(uv1, uv2, step)
    else:
        uv = np.array([uv1])
    return uv


def visibility_corners(corners):
    plan_y = boundary_type(corners)
    xyz = uv2xyz(corners, plan_y)
    xz = xyz[:, ::2]
    xz = calc_visible_polygon(center=np.array([0, 0]), polygon=xz, show=False)
    xyz = np.insert(xz, 1, plan_y, axis=1)
    output = xyz2uv(xyz).astype(np.float32)
    return output


def corners2boundary(corners: np.ndarray, step=0.01, length=None, visible=True) -> np.ndarray:
    """
    When there is occlusion, even if the length is fixed, the final output length may be greater than the given length,
     which is more defined as the fixed step size under UV
    :param length:
    :param step:
    :param corners: [[u1, v1], [u2, v2]....]
    :param visible:
    :return:  [[u1, v1], [u2, v2]....] if length!=None，length of return result = length
    """
    assert step is not None or length is not None, "the step and length parameters cannot be null at the same time"
    if len(corners) < 3:
        return corners

    if visible:
        corners = visibility_corners(corners)

    n_con = len(corners)
    boundary = None
    for j in range(n_con):
        uv = connect_corners(corners[j], corners[(j + 1) % n_con], step, length)
        if uv is None:
            continue
        if boundary is None:
            boundary = uv
        else:
            boundary = np.concatenate((boundary, uv))
    boundary = np.roll(boundary, -boundary.argmin(axis=0)[0], axis=0)

    output_polygon = []
    for i, p in enumerate(boundary):
        q = boundary[(i + 1) % len(boundary)]
        if int(p[0] * 10000) == int(q[0] * 10000):
            continue
        output_polygon.append(p)
    output_polygon = np.array(output_polygon, dtype=np.float32)
    return output_polygon


def corners2boundaries(ratio: float, corners_xyz: np.ndarray = None, corners_uv: np.ndarray = None, step=0.01,
                       length=None, visible=True):
    """
    When both step and length are None, corners are also returned
    :param ratio:
    :param corners_xyz:
    :param corners_uv:
    :param step:
    :param length:
    :param visible:
    :return: floor_boundary, ceil_boundary
    """
    if corners_xyz is None:
        plan_y = boundary_type(corners_uv)
        xyz = uv2xyz(corners_uv, plan_y)
        floor_xyz = xyz.copy()
        ceil_xyz = xyz.copy()
        if plan_y > 0:
            ceil_xyz[:, 1] *= -ratio
        else:
            floor_xyz[:, 1] /= -ratio
    else:
        floor_xyz = corners_xyz.copy()
        ceil_xyz = corners_xyz.copy()
        if corners_xyz[0][1] > 0:
            ceil_xyz[:, 1] *= -ratio
        else:
            floor_xyz[:, 1] /= -ratio

    floor_uv = xyz2uv(floor_xyz)
    ceil_uv = xyz2uv(ceil_xyz)
    if step is None and length is None:
        return floor_uv, ceil_uv

    floor_boundary = corners2boundary(floor_uv, step, length, visible)
    ceil_boundary = corners2boundary(ceil_uv, step, length, visible)
    return floor_boundary, ceil_boundary


def depth2boundary(depth: np.array, step=0.01, length=None,):
    xyz = depth2xyz(depth)
    uv = xyz2uv(xyz)
    return corners2boundary(uv, step, length, visible=False)


def depth2boundaries(ratio: float, depth: np.array, step=0.01, length=None,):
    """

    :param ratio:
    :param depth:
    :param step:
    :param length:
    :return: floor_boundary, ceil_boundary
    """
    xyz = depth2xyz(depth)
    return corners2boundaries(ratio, corners_xyz=xyz, step=step, length=length, visible=False)


def boundary_type(corners: np.ndarray) -> int:
    """
    Returns the boundary type that also represents the projection plane
    :param corners:
    :return:
    """
    if is_ceil_boundary(corners):
        plan_y = -1
    elif is_floor_boundary(corners):
        plan_y = 1
    else:
        # An intersection occurs and an exception is considered
        assert False, 'corners error!'
    return plan_y


def is_normal_layout(boundaries: List[np.array]):
    if len(boundaries) != 2:
        print("boundaries length must be 2!")
        return False

    if boundary_type(boundaries[0]) != -1:
        print("ceil boundary error!")
        return False

    if boundary_type(boundaries[1]) != 1:
        print("floor boundary error!")
        return False
    return True


def is_ceil_boundary(corners: np.ndarray) -> bool:
    m = corners[..., 1].max()
    return m < 0.5


def is_floor_boundary(corners: np.ndarray) -> bool:
    m = corners[..., 1].min()
    return m > 0.5


@functools.lru_cache()
def get_gauss_map(sigma=1.5, width=5):
    x = np.arange(width*2 + 1) - width
    y = stats.norm(0, sigma).pdf(x)
    y = y / y.max()
    return y


def get_heat_map(u_s, patch_num=256, sigma=2, window_width=15, show=False):
    """
    :param window_width:
    :param sigma:
    :param u_s: [u1, u2, u3, ...]
    :param patch_num
    :param show
    :return:
    """
    pixel_us = uv2pixel(u_s, w=patch_num, axis=0)
    gauss_map = get_gauss_map(sigma, window_width)
    heat_map_all = []
    for u in pixel_us:
        heat_map = np.zeros(patch_num, dtype=np.float)
        left = u-window_width
        right = u+window_width+1

        offset = 0
        if left < 0:
            offset = left
        elif right > patch_num:
            offset = right - patch_num

        left = left - offset
        right = right - offset
        heat_map[left:right] = gauss_map
        if offset != 0:
            heat_map = np.roll(heat_map, offset)
        heat_map_all.append(heat_map)

    heat_map_all = np.array(heat_map_all).max(axis=0)
    if show:
        import matplotlib.pyplot as plt
        plt.imshow(heat_map_all[None].repeat(50, axis=0))
        plt.show()
    return heat_map_all


def find_peaks(signal, size=15*2+1, min_v=0.05, N=None):
    # code from HorizonNet: https://github.com/sunset1995/HorizonNet/blob/master/inference.py
    max_v = maximum_filter(signal, size=size, mode='wrap')
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > min_v]
    if N is not None:
        order = np.argsort(-signal[pk_loc])
        pk_loc = pk_loc[order[:N]]
        pk_loc = pk_loc[np.argsort(pk_loc)]
    return pk_loc, signal[pk_loc]


def get_object_cor(depth, size, center_u, patch_num=256):
    width_u = size[0, center_u]
    height_v = size[1, center_u]
    boundary_v = size[2, center_u]

    center_boundary_v = depth2uv(depth[center_u:center_u + 1])[0, 1]
    center_bottom_v = center_boundary_v - boundary_v
    center_top_v = center_bottom_v - height_v

    base_v = center_boundary_v - 0.5
    assert base_v > 0

    center_u = pixel2uv(np.array([center_u]), w=patch_num, h=patch_num // 2, axis=0)[0]

    center_boundary_uv = np.array([center_u, center_boundary_v])
    center_bottom_uv = np.array([center_u, center_bottom_v])
    center_top_uv = np.array([center_u, center_top_v])

    left_u = center_u - width_u / 2
    right_u = center_u + width_u / 2

    left_u = 1 + left_u if left_u < 0 else left_u
    right_u = right_u - 1 if right_u > 1 else right_u

    pixel_u = uv2pixel(np.array([left_u, right_u]), w=patch_num, h=patch_num // 2, axis=0)
    left_pixel_u = pixel_u[0]
    right_pixel_u = pixel_u[1]

    left_boundary_v = depth2uv(depth[left_pixel_u:left_pixel_u + 1])[0, 1]
    right_boundary_v = depth2uv(depth[right_pixel_u:right_pixel_u + 1])[0, 1]

    left_boundary_uv = np.array([left_u, left_boundary_v])
    right_boundary_uv = np.array([right_u, right_boundary_v])

    xyz = uv2xyz(np.array([left_boundary_uv, right_boundary_uv, center_boundary_uv]))
    left_boundary_xyz = xyz[0]
    right_boundary_xyz = xyz[1]

    # need align
    center_boundary_xyz = xyz[2]
    center_bottom_xyz = uv2xyz(np.array([center_bottom_uv]))[0]
    center_top_xyz = uv2xyz(np.array([center_top_uv]))[0]
    center_boundary_norm = np.linalg.norm(center_boundary_xyz[::2])
    center_bottom_norm = np.linalg.norm(center_bottom_xyz[::2])
    center_top_norm = np.linalg.norm(center_top_xyz[::2])
    center_bottom_xyz = center_bottom_xyz * center_boundary_norm / center_bottom_norm
    center_top_xyz = center_top_xyz * center_boundary_norm / center_top_norm

    left_bottom_xyz = left_boundary_xyz.copy()
    left_bottom_xyz[1] = center_bottom_xyz[1]
    right_bottom_xyz = right_boundary_xyz.copy()
    right_bottom_xyz[1] = center_bottom_xyz[1]

    left_top_xyz = left_boundary_xyz.copy()
    left_top_xyz[1] = center_top_xyz[1]
    right_top_xyz = right_boundary_xyz.copy()
    right_top_xyz[1] = center_top_xyz[1]

    uv = xyz2uv(np.array([left_bottom_xyz, right_bottom_xyz, left_top_xyz, right_top_xyz]))
    left_bottom_uv = uv[0]
    right_bottom_uv = uv[1]
    left_top_uv = uv[2]
    right_top_uv = uv[3]

    return [left_bottom_uv, right_bottom_uv, left_top_uv, right_top_uv], \
           [left_bottom_xyz, right_bottom_xyz, left_top_xyz, right_top_xyz]


def layout2depth(boundaries: List[np.array], return_mask=False, show=False, camera_height=1.6):
    """

    :param camera_height:
    :param boundaries: [[[u_f1, v_f2], [u_f2, v_f2],...], [[u_c1, v_c2], [u_c2, v_c2]]]
    :param return_mask:
    :param show:
    :return:
    """
    # code from HorizonNet: https://github.com/sunset1995/HorizonNet/blob/master/eval_general.py

    w = len(boundaries[0])
    h = w//2
    # Convert corners to per-column boundary first
    # Up -pi/2,  Down pi/2
    vf = uv2lonlat(boundaries[0])
    vc = uv2lonlat(boundaries[1])
    vc = vc[None, :, 1]  # [1, w]
    vf = vf[None, :, 1]  # [1, w]
    assert (vc > 0).sum() == 0
    assert (vf < 0).sum() == 0

    # Per-pixel v coordinate (vertical angle)
    vs = ((np.arange(h) + 0.5) / h - 0.5) * np.pi
    vs = np.repeat(vs[:, None], w, axis=1)  # [h, w]

    # Floor-plane to depth
    floor_h = camera_height
    floor_d = np.abs(floor_h / np.sin(vs))

    # wall to camera distance on horizontal plane at cross camera center
    cs = floor_h / np.tan(vf)

    # Ceiling-plane to depth
    ceil_h = np.abs(cs * np.tan(vc))  # [1, w]
    ceil_d = np.abs(ceil_h / np.sin(vs))  # [h, w]

    # Wall to depth
    wall_d = np.abs(cs / np.cos(vs))  # [h, w]

    # Recover layout depth
    floor_mask = (vs > vf)
    ceil_mask = (vs < vc)
    wall_mask = (~floor_mask) & (~ceil_mask)
    depth = np.zeros([h, w], np.float32)  # [h, w]
    depth[floor_mask] = floor_d[floor_mask]
    depth[ceil_mask] = ceil_d[ceil_mask]
    depth[wall_mask] = wall_d[wall_mask]

    assert (depth == 0).sum() == 0
    if return_mask:
        return depth, floor_mask, ceil_mask, wall_mask
    if show:
        import matplotlib.pyplot as plt
        plt.imshow(depth)
        plt.show()
    return depth


def calc_rotation(corners: np.ndarray):
    xz = uv2xyz(corners)[..., 0::2]
    max_norm = -1
    max_v = None
    for i in range(len(xz)):
        p_c = xz[i]
        p_n = xz[(i + 1) % len(xz)]
        v_cn = p_n - p_c
        v_norm = np.linalg.norm(v_cn)
        if v_norm > max_norm:
            max_norm = v_norm
            max_v = v_cn

    # v<-----------|o
    # |     |      |
    # | ----|----z |
    # |     |      |
    # |     x     \|/
    # |------------u
    # It is required that the vector be aligned on the x-axis, z equals y, and x is still x.
    # In floorplan, x is displayed as the x-coordinate and z as the y-coordinate
    rotation = np.arctan2(max_v[1], max_v[0])
    return rotation


if __name__ == '__main__':
    corners = np.array([[0.2, 0.7],
                        [0.4, 0.7],
                        [0.3, 0.6],
                        [0.6, 0.6],
                        [0.8, 0.7]])
    get_heat_map(u=corners[..., 0], show=True, sigma=2, width=15)
    pass

