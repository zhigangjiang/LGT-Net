""" 
@Date: 2021/10/06
@description: Use the approach proposed by DuLa-Net
"""
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

from visualization.floorplan import draw_floorplan


def merge_near(lst, diag):
    group = [[0, ]]
    for i in range(1, len(lst)):
        if lst[i] - np.mean(group[-1]) < diag * 0.02:
            group[-1].append(lst[i])
        else:
            group.append([lst[i], ])
    if len(group) == 1:
        group = [lst[0], lst[-1]]
    else:
        group = [int(np.mean(x)) for x in group]
    return group


def fit_layout_old(floor_xz, need_cube=False, show=False, block_eps=0.05):
    show_radius = np.linalg.norm(floor_xz, axis=-1).max()
    side_l = 512
    floorplan = draw_floorplan(xz=floor_xz, show_radius=show_radius, show=show, scale=1, side_l=side_l).astype(np.uint8)
    center = np.array([side_l / 2, side_l / 2])
    polys = cv2.findContours(floorplan, 1, 2)
    if isinstance(polys, tuple):
        if len(polys) == 3:
            # opencv 3
            polys = list(polys[1])
        else:
            polys = list(polys[0])
    polys.sort(key=lambda x: cv2.contourArea(x), reverse=True)
    poly = polys[0]
    sub_x, sub_y, w, h = cv2.boundingRect(poly)
    floorplan_sub = floorplan[sub_y:sub_y + h, sub_x:sub_x + w]
    sub_center = center - np.array([sub_x, sub_y])
    polys = cv2.findContours(floorplan_sub, 1, 2)
    if isinstance(polys, tuple):
        if len(polys) == 3:
            polys = polys[1]
        else:
            polys = polys[0]
    poly = polys[0]
    epsilon = 0.005 * cv2.arcLength(poly, True)
    poly = cv2.approxPolyDP(poly, epsilon, True)

    x_lst = [0, ]
    y_lst = [0, ]
    for i in range(len(poly)):
        p1 = poly[i][0]
        p2 = poly[(i + 1) % len(poly)][0]

        if (p2[0] - p1[0]) == 0:
            slope = 10
        else:
            slope = abs((p2[1] - p1[1]) / (p2[0] - p1[0]))

        if slope <= 1:
            s = int((p1[1] + p2[1]) / 2)
            y_lst.append(s)
        elif slope > 1:
            s = int((p1[0] + p2[0]) / 2)
            x_lst.append(s)

    x_lst.append(floorplan_sub.shape[1])
    y_lst.append(floorplan_sub.shape[0])
    x_lst.sort()
    y_lst.sort()

    diag = math.sqrt(math.pow(floorplan_sub.shape[1], 2) + math.pow(floorplan_sub.shape[0], 2))
    x_lst = merge_near(x_lst, diag)
    y_lst = merge_near(y_lst, diag)
    if need_cube and len(x_lst) > 2:
        x_lst = [x_lst[0], x_lst[-1]]
    if need_cube and len(y_lst) > 2:
        y_lst = [y_lst[0], y_lst[-1]]

    ans = np.zeros((floorplan_sub.shape[0], floorplan_sub.shape[1]))
    for i in range(len(x_lst) - 1):
        for j in range(len(y_lst) - 1):
            sample = floorplan_sub[y_lst[j]:y_lst[j + 1], x_lst[i]:x_lst[i + 1]]
            score = 0 if sample.size == 0 else sample.mean()
            if score >= 0.3:
                ans[y_lst[j]:y_lst[j + 1], x_lst[i]:x_lst[i + 1]] = 1

    pred = np.uint8(ans)
    pred_polys = cv2.findContours(pred, 1, 3)
    if isinstance(pred_polys, tuple):
        if len(pred_polys) == 3:
            pred_polys = pred_polys[1]
        else:
            pred_polys = pred_polys[0]

    polygon = [(p[0][1], p[0][0]) for p in pred_polys[0][::-1]]

    v = np.array([p[0] + sub_y for p in polygon])
    u = np.array([p[1] + sub_x for p in polygon])
    #     side_l
    # v<-----------|o
    # |     |      |
    # | ----|----z |   side_l
    # |     |      |
    # |     x     \|/
    # |------------u
    side_l = floorplan.shape[0]
    pred_xz = np.concatenate((u[:, np.newaxis] - side_l // 2, side_l // 2 - v[:, np.newaxis]), axis=1)

    pred_xz = pred_xz * show_radius / (side_l // 2)
    if show:
        draw_floorplan(pred_xz, show_radius=show_radius, show=show)
    return pred_xz


if __name__ == '__main__':
    from utils.conversion import uv2xyz

    pano_img = np.zeros([512, 1024, 3])
    corners = np.array([[0.1, 0.7],
                        [0.4, 0.7],
                        [0.3, 0.6],
                        [0.6, 0.6],
                        [0.8, 0.7]])
    xz = uv2xyz(corners)[..., ::2]
    draw_floorplan(xz, show=True, marker_color=None, center_color=0.8)

    xz = fit_layout_old(xz)
    draw_floorplan(xz, show=True, marker_color=None, center_color=0.8)
