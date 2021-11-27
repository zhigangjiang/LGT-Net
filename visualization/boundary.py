"""
@date: 2021/06/19
@description:
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
from utils.conversion import uv2pixel
from utils.boundary import corners2boundary, corners2boundaries, find_peaks, connect_corners_uv, get_object_cor, \
    visibility_corners


def draw_boundary(pano_img, corners: np.ndarray = None, boundary: np.ndarray = None, draw_corners=True, show=False,
                  step=0.01, length=None, boundary_color=None, marker_color=None, title=None, visible=True):
    if marker_color is None:
        marker_color = [0, 0, 1]
    if boundary_color is None:
        boundary_color = [0, 1, 0]

    assert corners is not None or boundary is not None, "corners or boundary error"

    shape = sorted(pano_img.shape)
    assert len(shape) > 1, "pano_img shape error"
    w = shape[-1]
    h = shape[-2]

    pano_img = pano_img.copy()
    if (corners is not None and len(corners) > 2) or \
            (boundary is not None and len(boundary) > 2):
        if isinstance(boundary_color, list) or isinstance(boundary_color, np.array):
            if boundary is None:
                boundary = corners2boundary(corners, step, length, visible)

            boundary = uv2pixel(boundary, w, h)
            pano_img[boundary[:, 1], boundary[:, 0]] = boundary_color
            pano_img[np.clip(boundary[:, 1] + 1, 0, h - 1), boundary[:, 0]] = boundary_color
            pano_img[np.clip(boundary[:, 1] - 1, 0, h - 1), boundary[:, 0]] = boundary_color

            if pano_img.shape[1] > 512:
                pano_img[np.clip(boundary[:, 1] + 1, 0, h - 1), np.clip(boundary[:, 0] + 1, 0, w - 1)] = boundary_color
                pano_img[np.clip(boundary[:, 1] + 1, 0, h - 1), np.clip(boundary[:, 0] - 1, 0, w - 1)] = boundary_color
                pano_img[np.clip(boundary[:, 1] - 1, 0, h - 1), np.clip(boundary[:, 0] + 1, 0, w - 1)] = boundary_color
                pano_img[np.clip(boundary[:, 1] - 1, 0, h - 1), np.clip(boundary[:, 0] - 1, 0, w - 1)] = boundary_color

            pano_img[boundary[:, 1], np.clip(boundary[:, 0] + 1, 0, w - 1)] = boundary_color
            pano_img[boundary[:, 1], np.clip(boundary[:, 0] - 1, 0, w - 1)] = boundary_color

        if corners is not None and draw_corners:
            if visible:
                corners = visibility_corners(corners)
            corners = uv2pixel(corners, w, h)
            for corner in corners:
                cv2.drawMarker(pano_img, tuple(corner), marker_color, markerType=0, markerSize=10, thickness=2)

    if show:
        plt.figure(figsize=(10, 5))
        if title is not None:
            plt.title(title)

        plt.axis('off')
        plt.imshow(pano_img)
        plt.show()

    return pano_img


def draw_boundaries(pano_img, corners_list: list = None, boundary_list: list = None, draw_corners=True, show=False,
                    step=0.01, length=None, boundary_color=None, marker_color=None, title=None, ratio=None, visible=True):
    """

    :param visible:
    :param pano_img:
    :param corners_list:
    :param boundary_list:
    :param draw_corners:
    :param show:
    :param step:
    :param length:
    :param boundary_color: RGB color
    :param marker_color: RGB color
    :param title:
    :param ratio: ceil_height/camera_height
    :return:
    """
    assert corners_list is not None or boundary_list is not None, "corners_list or boundary_list error"

    if corners_list is not None:
        if ratio is not None and len(corners_list) == 1:
            corners_list = corners2boundaries(ratio, corners_uv=corners_list[0], step=None, visible=visible)

        for i, corners in enumerate(corners_list):
            pano_img = draw_boundary(pano_img, corners=corners, draw_corners=draw_corners,
                                     show=show if i == len(corners_list) - 1 else False,
                                     step=step, length=length, boundary_color=boundary_color, marker_color=marker_color,
                                     title=title, visible=visible)
    elif boundary_list is not None:
        if ratio is not None and len(boundary_list) == 1:
            boundary_list = corners2boundaries(ratio, corners_uv=boundary_list[0], step=None, visible=visible)

        for i, boundary in enumerate(boundary_list):
            pano_img = draw_boundary(pano_img, boundary=boundary, draw_corners=draw_corners,
                                     show=show if i == len(boundary_list) - 1 else False,
                                     step=step, length=length, boundary_color=boundary_color, marker_color=marker_color,
                                     title=title, visible=visible)

    return pano_img


def draw_object(pano_img, heat_maps, size, depth, window_width=15, show=False):
    # window, door, opening
    colors = [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
    for i, heat_map in enumerate(heat_maps):
        pk_u_s, _ = find_peaks(heat_map, size=window_width*2+1)
        for pk_u in pk_u_s:
            uv, xyz = get_object_cor(depth, size, center_u=pk_u, patch_num=len(heat_map))

            bottom_poly = connect_corners_uv(uv[0], uv[1], length=pano_img.shape[1])
            top_poly = connect_corners_uv(uv[2], uv[3], length=pano_img.shape[1])[::-1]

            bottom_max_index = bottom_poly[..., 0].argmax()
            if bottom_max_index != len(bottom_poly)-1:
                top_max_index = top_poly[..., 0].argmax()
                poly1 = np.concatenate([bottom_poly[:bottom_max_index+1], top_poly[top_max_index:]])
                poly1 = uv2pixel(poly1, w=pano_img.shape[1], h=pano_img.shape[0])
                poly1 = poly1[:, None, :]

                poly2 = np.concatenate([bottom_poly[bottom_max_index+1:], top_poly[:top_max_index]])
                poly2 = uv2pixel(poly2, w=pano_img.shape[1], h=pano_img.shape[0])
                poly2 = poly2[:, None, :]

                poly = [poly1, poly2]
            else:
                poly = np.concatenate([bottom_poly, top_poly])
                poly = uv2pixel(poly, w=pano_img.shape[1], h=pano_img.shape[0])
                poly = poly[:, None, :]
                poly = [poly]

            cv2.drawContours(pano_img, poly, -1, colors[i], 1)
            #
            # boundary_center_xyz = uv2xyz(np.array([pk_u, pk_v]))
            #
            # l_b_xyz =
    if show:
        plt.imshow(pano_img)
        plt.show()


if __name__ == '__main__':
    from visualization.floorplan import draw_floorplan
    from utils.conversion import uv2xyz

    pano_img = np.zeros([512, 1024, 3])
    corners = np.array([[0.2, 0.7],
                        [0.4, 0.7],
                        [0.3, 0.6],
                        [0.6, 0.6],
                        [0.8, 0.7]])
    # draw_boundary(pano_img, corners, show=True)
    draw_boundaries(pano_img, corners_list=[corners], show=True, length=1024, ratio=1.2)
    draw_floorplan(uv2xyz(corners)[..., ::2], show=True, marker_color=None, center_color=0.8)