""" 
@date: 2021/06/19
@description:
Specification of 4 coordinate systems:
Pixel coordinates (used in panoramic images), the range is related to the image size,
generally converted to UV coordinates first, the first is horizontal coordinates,
increasing to the right, the second is column coordinates, increasing down

Uv coordinates (used in panoramic images), the range is [0~1], the upper left corner is the origin,
u is the abscissa and increases to the right, V is the column coordinate and increases to the right

Longitude and latitude coordinates (spherical), the range of longitude lon is [-pi~ PI],
and the range of dimension is [-pi/2~ PI /2]. The center of the panorama is the origin,
and the longitude increases to the right and the dimension increases to the down

Xyz coordinate (used in 3-dimensional space, of course,
it can also represent longitude and latitude coordinates on the sphere).
If on the sphere, the coordinate mode length is 1, when y is projected to the height of the camera,
the real position information of space points will be obtained

Correspondence between longitude and latitude coordinates and xyz coordinates:
                     | -pi/2
                     |
     lef   _ _ _ _ _ |_ _ _ _ _
     -pi /           |          \
     pi |            - - - - -  -\ - z 0 mid
 right  \_ _ _ _ _ /_|_ _ _ _ _ _/
                /    |
               /     |
            x/       | y pi/2
"""

import numpy as np
import torch
import functools


@functools.lru_cache()
def get_u(w, is_np, b=None):
    u = pixel2uv(np.array(range(w)) if is_np else torch.arange(0, w), w=w, axis=0)
    if b is not None:
        u = u[np.newaxis].repeat(b) if is_np else u.repeat(b, 1)
    return u


@functools.lru_cache()
def get_lon(w, is_np, b=None):
    lon = pixel2lonlat(np.array(range(w)) if is_np else torch.arange(0, w), w=w, axis=0)
    if b is not None:
        lon = lon[np.newaxis].repeat(b, axis=0) if is_np else lon.repeat(b, 1)
    return lon


def pixel2uv(pixel, w=1024, h=512, axis=None):
    pixel = pixel.astype(np.float) if isinstance(pixel, np.ndarray) else pixel.float()
    # +0.5 will make left/right and up/down coordinates symmetric
    if axis is None:
        u = (pixel[..., 0:1] + 0.5) / w
        v = (pixel[..., 1:] + 0.5) / h
    elif axis == 0:
        u = (pixel + 0.5) / (w * 1.0)
        return u
    elif axis == 1:
        v = (pixel + 0.5) / (h * 1.0)
        return v
    else:
        assert False, "axis error"

    lst = [u, v]
    uv = np.concatenate(lst, axis=-1) if isinstance(pixel, np.ndarray) else torch.cat(lst, dim=-1)
    return uv


def pixel2lonlat(pixel, w=1024, h=512, axis=None):
    uv = pixel2uv(pixel, w, h, axis)
    lonlat = uv2lonlat(uv, axis)
    return lonlat


def pixel2xyz(pixel, w=1024, h=512):
    lonlat = pixel2lonlat(pixel, w, h)
    xyz = lonlat2xyz(lonlat)
    return xyz


def uv2lonlat(uv, axis=None):
    if axis is None:
        lon = (uv[..., 0:1] - 0.5) * 2 * np.pi
        lat = (uv[..., 1:] - 0.5) * np.pi
    elif axis == 0:
        lon = (uv - 0.5) * 2 * np.pi
        return lon
    elif axis == 1:
        lat = (uv - 0.5) * np.pi
        return lat
    else:
        assert False, "axis error"

    lst = [lon, lat]
    lonlat = np.concatenate(lst, axis=-1) if isinstance(uv, np.ndarray) else torch.cat(lst, dim=-1)
    return lonlat


def uv2xyz(uv, plan_y=None, spherical=False):
    lonlat = uv2lonlat(uv)
    xyz = lonlat2xyz(lonlat)
    if spherical:
        # Projection onto the sphere
        return xyz

    if plan_y is None:
        from utils.boundary import boundary_type
        plan_y = boundary_type(uv)
    # Projection onto the specified plane
    xyz = xyz * (plan_y / xyz[..., 1])[..., None]

    return xyz


def lonlat2xyz(lonlat, plan_y=None):
    lon = lonlat[..., 0:1]
    lat = lonlat[..., 1:]
    cos = np.cos if isinstance(lonlat, np.ndarray) else torch.cos
    sin = np.sin if isinstance(lonlat, np.ndarray) else torch.sin
    x = cos(lat) * sin(lon)
    y = sin(lat)
    z = cos(lat) * cos(lon)
    lst = [x, y, z]
    xyz = np.concatenate(lst, axis=-1) if isinstance(lonlat, np.ndarray) else torch.cat(lst, dim=-1)

    if plan_y is not None:
        xyz = xyz * (plan_y / xyz[..., 1])[..., None]

    return xyz


#####################


def xyz2lonlat(xyz):
    atan2 = np.arctan2 if isinstance(xyz, np.ndarray) else torch.atan2
    asin = np.arcsin if isinstance(xyz, np.ndarray) else torch.asin
    norm = np.linalg.norm(xyz, axis=-1) if isinstance(xyz, np.ndarray) else torch.norm(xyz, p=2, dim=-1)
    xyz_norm = xyz / norm[..., None]
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]
    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]
    lonlat = np.concatenate(lst, axis=-1) if isinstance(xyz, np.ndarray) else torch.cat(lst, dim=-1)
    return lonlat


def xyz2uv(xyz):
    lonlat = xyz2lonlat(xyz)
    uv = lonlat2uv(lonlat)
    return uv


def xyz2pixel(xyz, w=1024, h=512):
    uv = xyz2uv(xyz)
    pixel = uv2pixel(uv, w, h)
    return pixel


def lonlat2uv(lonlat, axis=None):
    if axis is None:
        u = lonlat[..., 0:1] / (2 * np.pi) + 0.5
        v = lonlat[..., 1:] / np.pi + 0.5
    elif axis == 0:
        u = lonlat / (2 * np.pi) + 0.5
        return u
    elif axis == 1:
        v = lonlat / np.pi + 0.5
        return v
    else:
        assert False, "axis error"

    lst = [u, v]
    uv = np.concatenate(lst, axis=-1) if isinstance(lonlat, np.ndarray) else torch.cat(lst, dim=-1)
    return uv


def lonlat2pixel(lonlat, w=1024, h=512, axis=None, need_round=True):
    uv = lonlat2uv(lonlat, axis)
    pixel = uv2pixel(uv, w, h, axis, need_round)
    return pixel


def uv2pixel(uv, w=1024, h=512, axis=None, need_round=True):
    """
    :param uv: [[u1, v1], [u2, v2] ...]
    :param w: width of panorama image
    :param h: height of panorama image
    :param axis: sometimes the input data is only u(axis =0) or only v(axis=1)
    :param need_round:
    :return:
    """
    if axis is None:
        pu = uv[..., 0:1] * w - 0.5
        pv = uv[..., 1:] * h - 0.5
    elif axis == 0:
        pu = uv * w - 0.5
        if need_round:
            pu = pu.round().astype(np.int) if isinstance(uv, np.ndarray) else pu.round().int()
        return pu
    elif axis == 1:
        pv = uv * h - 0.5
        if need_round:
            pv = pv.round().astype(np.int) if isinstance(uv, np.ndarray) else pv.round().int()
        return pv
    else:
        assert False, "axis error"

    lst = [pu, pv]
    if need_round:
        pixel = np.concatenate(lst, axis=-1).round().astype(np.int) if isinstance(uv, np.ndarray) else torch.cat(lst,
                                                                                                                 dim=-1).round().int()
    else:
        pixel = np.concatenate(lst, axis=-1) if isinstance(uv, np.ndarray) else torch.cat(lst, dim=-1)
    pixel[..., 0] = pixel[..., 0] % w
    pixel[..., 1] = pixel[..., 1] % h

    return pixel


#####################


def xyz2depth(xyz, plan_y=1):
    """
    :param xyz:
    :param plan_y:
    :return:
    """
    xyz = xyz * (plan_y / xyz[..., 1])[..., None]
    xz = xyz[..., ::2]
    depth = np.linalg.norm(xz, axis=-1) if isinstance(xz, np.ndarray) else torch.norm(xz, dim=-1)
    return depth


def uv2depth(uv, plan_y=None):
    if plan_y is None:
        from utils.boundary import boundary_type
        plan_y = boundary_type(uv)

    xyz = uv2xyz(uv, plan_y)
    depth = xyz2depth(xyz, plan_y)
    return depth


def lonlat2depth(lonlat, plan_y=None):
    if plan_y is None:
        from utils.boundary import boundary_type
        plan_y = boundary_type(lonlat2uv(lonlat))

    xyz = lonlat2xyz(lonlat, plan_y)
    depth = xyz2depth(xyz, plan_y)
    return depth


def depth2xyz(depth, plan_y=1):
    """
    :param depth: [patch_num] or [b, patch_num]
    :param plan_y:
    :return:
    """
    is_np = isinstance(depth, np.ndarray)
    w = depth.shape[-1]

    lon = get_lon(w, is_np, b=depth.shape[0] if len(depth.shape) == 2 else None)
    if not is_np:
        lon = lon.to(depth.device)

    cos = np.cos if is_np else torch.cos
    sin = np.sin if is_np else torch.sin
    # polar covert to cartesian
    if len(depth.shape) == 2:
        b = depth.shape[0]
        xyz = np.zeros((b, w, 3)) if is_np else torch.zeros((b, w, 3))
    else:
        xyz = np.zeros((w, 3)) if is_np else torch.zeros((w, 3))

    if not is_np:
        xyz = xyz.to(depth.device)

    xyz[..., 0] = depth * sin(lon)
    xyz[..., 1] = plan_y
    xyz[..., 2] = depth * cos(lon)
    return xyz


def depth2uv(depth, plan_y=1):
    xyz = depth2xyz(depth, plan_y)
    uv = xyz2uv(xyz)
    return uv


def depth2pixel(depth, w=1024, h=512, need_round=True, plan_y=1):
    uv = depth2uv(depth, plan_y)
    pixel = uv2pixel(uv, w, h, need_round=need_round)
    return pixel


if __name__ == '__main__':
    a = np.array([[0.5, 1, 0.5]])
    a = xyz2pixel(a)
    print(a)


if __name__ == '__main__1':
    np.set_printoptions(suppress=True)

    a = np.array([[0, 0], [1023, 511]])
    a = pixel2xyz(a)
    a = xyz2pixel(a)
    print(a)

    ###########
    a = torch.tensor([[0, 0], [1023, 511]])
    a = pixel2xyz(a)
    a = xyz2pixel(a)
    print(a)

    ###########
    u = np.array([0, 256, 512, 1023])
    lon = pixel2lonlat(u, axis=0)
    u = lonlat2pixel(lon, axis=0)
    print(u)

    u = torch.tensor([0, 256, 512, 1023])
    lon = pixel2lonlat(u, axis=0)
    u = lonlat2pixel(lon, axis=0)
    print(u)

    ###########
    v = np.array([0, 256, 511])
    lat = pixel2lonlat(v, axis=1)
    v = lonlat2pixel(lat, axis=1)
    print(v)

    v = torch.tensor([0, 256, 511])
    lat = pixel2lonlat(v, axis=1)
    v = lonlat2pixel(lat, axis=1)
    print(v)
