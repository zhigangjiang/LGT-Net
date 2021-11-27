"""
@date: 2021/6/30
@description:
"""
import numpy as np
from typing import List

from utils.boundary import *
from scipy.optimize import least_squares
from functools import partial


def lsq_fit(ceil_norm, floor_norm):
    """
    Least Squares
    :param ceil_norm:
    :param floor_norm:
    :return:
    """

    def error_fun(ratio, ceil_norm, floor_norm):
        error = np.abs(ratio * ceil_norm - floor_norm)
        return error

    init_ratio = np.mean(floor_norm / ceil_norm, axis=-1)
    error_func = partial(error_fun, ceil_norm=ceil_norm, floor_norm=floor_norm)
    ret = least_squares(error_func, init_ratio, verbose=0)
    ratio = ret.x[0]
    return ratio


def mean_percentile_fit(ceil_norm, floor_norm, p1=25, p2=75):
    """
    :param ceil_norm:
    :param floor_norm:
    :param p1:
    :param p2:
    :return:
    """
    ratio = floor_norm / ceil_norm
    r_min = np.percentile(ratio, p1)
    r_max = np.percentile(ratio, p2)
    return ratio[(r_min <= ratio) & (ratio <= r_max)].mean()


def calc_ceil_ratio(boundaries: List[np.array], mode='lsq'):
    """
    :param boundaries: [ [[cu1, cv1], [cu2, cv2], ...], [[fu1, fv1], [fu2, fv2], ...] ]
    :param mode: 'lsq' or 'mean'
    :return:
    """
    assert len(boundaries[0].shape) < 4 and len(boundaries[1].shape) < 4, 'error shape'
    if not is_normal_layout(boundaries):
        return 0

    ceil_boundary = boundaries[0]
    floor_boundary = boundaries[1]
    assert ceil_boundary.shape[-2] == floor_boundary.shape[-2], "boundary need same length"

    ceil_xyz = uv2xyz(ceil_boundary, -1)
    floor_xyz = uv2xyz(floor_boundary, 1)

    ceil_xz = ceil_xyz[..., ::2]
    floor_xz = floor_xyz[..., ::2]

    ceil_norm = np.linalg.norm(ceil_xz, axis=-1)
    floor_norm = np.linalg.norm(floor_xz, axis=-1)

    if mode == "lsq":
        if len(ceil_norm.shape) == 2:
            ratio = np.array([lsq_fit(ceil_norm[i], floor_norm[i]) for i in range(ceil_norm.shape[0])])
        else:
            ratio = lsq_fit(ceil_norm, floor_norm)
    else:
        if len(ceil_norm.shape) == 2:
            ratio = np.array([mean_percentile_fit(ceil_norm[i], floor_norm[i]) for i in range(ceil_norm.shape[0])])
        else:
            ratio = mean_percentile_fit(ceil_norm, floor_norm)

    return ratio


def calc_ceil_height(boundaries: List[np.array], camera_height=1.6, mode='lsq') -> float:
    """
    :param boundaries: [ [[cu1, cv1], [cu2, cv2], ...], [[fu1, fv1], [fu2, fv2], ...] ]
    :param camera_height:
    :param mode:
    :return:
    """
    ratio = calc_ceil_ratio(boundaries, mode)
    ceil_height = camera_height * ratio
    return ceil_height


def calc_room_height(boundaries: List[np.array], camera_height=1.6, mode='lsq') -> float:
    """
    :param boundaries: also can corners，format: [ [[cu1, cv1], [cu2, cv2], ...], [[fu1, fv1], [fu2, fv2], ...] ],
    0 denotes ceil, 1 denotes floor
    :param camera_height: actual camera height determines the scale
    :param mode: fitting method lsq or mean
    :return:
    """
    ceil_height = calc_ceil_height(boundaries, camera_height, mode)
    room_height = camera_height + ceil_height
    return room_height


def height2ratio(height, camera_height=1.6):
    ceil_height = height - camera_height
    ratio = ceil_height / camera_height
    return ratio


def ratio2height(ratio, camera_height=1.6):
    ceil_height = camera_height * ratio
    room_height = camera_height + ceil_height
    return room_height


if __name__ == '__main__':
    from dataset.mp3d_dataset import MP3DDataset

    dataset = MP3DDataset(root_dir="../src/dataset/mp3d", mode="train")
    for data in dataset:
        ceil_corners = data['corners'][::2]
        floor_corners = data['corners'][1::2]
        # ceil_boundary = corners2boundary(ceil_corners, length=1024)
        # floor_boundary = corners2boundary(floor_corners, length=1024)
        room_height1 = calc_room_height([ceil_corners, floor_corners], camera_height=1.6, mode='mean')
        room_height2 = calc_room_height([ceil_corners, floor_corners], camera_height=1.6, mode='lsq')
        print(room_height1, room_height2, data['cameraCeilingHeight'] + 1.6)
