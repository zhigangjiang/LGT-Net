""" 
@Date: 2021/10/06
@description: Use the approach proposed by DuLa-Net
"""
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, '../../../')))

from visualization.floorplan import draw_floorplan

def calc_angle(v1: np.array, v2: np.array):
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    theta = np.arccos(np.dot(v1, v2) / norm)
    return theta
    
def merge_near(lst, diag, min):
    group = [[min, ]]
    for i in range(1, len(lst)):
        if lst[i][1] == 0 and lst[i][0] - np.mean(group[-1]) < diag * 0.02:
            group[-1].append(lst[i][0])
        else:
            group.append([lst[i][0], ])
    if len(group) == 1:
        group = [lst[0][0], lst[-1][0]]
    else:
        group = [int(np.mean(x)) for x in group]
    return group


def fit_layout(floor_xz, need_cube=False, show=False, block_eps=5):
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

    x_lst = [[poly[:, 0, 0].min(), 0], ]
    y_lst = [[poly[:, 0, 1].min(), 0], ]

    ans = np.zeros((floorplan_sub.shape[0], floorplan_sub.shape[1]))

    for i in range(len(poly)):
        p1 = poly[i][0]
        p2 = poly[(i + 1) % len(poly)][0]
        # We added occlusion detection
        cp1 = p1 - sub_center
        cp2 = p2 - sub_center
        p12 = p2 - p1
        l1 = np.linalg.norm(cp1)
        l2 = np.linalg.norm(cp2)
        l3 = np.linalg.norm(p12)
        # We added occlusion detection
        is_block1 = np.rad2deg(calc_angle(cp1, cp2)) < block_eps
        is_block2 = np.rad2deg(calc_angle(cp2, p12)) < block_eps*2
        is_block3 = np.rad2deg(calc_angle(cp2, -p12)) < block_eps*2
        is_block = is_block1 and (is_block2 or is_block3)

        if (p2[0] - p1[0]) == 0:
            slope = 10
        else:
            slope = abs((p2[1] - p1[1]) / (p2[0] - p1[0]))

        if is_block:
            s = p1[1] if l1 < l2 else p2[1]
            y_lst.append([s, 1])
            s = p1[0] if l1 < l2 else p2[0]
            x_lst.append([s, 1])

            left = p1[0] if p1[0] < p2[0] else p2[0]
            right = p1[0] if p1[0] > p2[0] else p2[0]
            top = p1[1] if p1[1] < p2[1] else p2[1]
            bottom = p1[1] if p1[1] > p2[1] else p2[1]
            sample = floorplan_sub[top:bottom, left:right]
            score = 0 if sample.size == 0 else sample.mean()
            if score >= 0.3:
                ans[top:bottom, left:right] = 1

        else:
            if slope <= 1:
                s = int((p1[1] + p2[1]) / 2)
                y_lst.append([s, 0])
            elif slope > 1:
                s = int((p1[0] + p2[0]) / 2)
                x_lst.append([s, 0])

    debug_show = False
    if debug_show:
        plt.figure(dpi=300)
        plt.axis('off')
        a = cv2.drawMarker(floorplan_sub.copy()*0.5, tuple(sub_center.astype(int)), [1], markerType=0, markerSize=10, thickness=2)
        plt.imshow(cv2.drawContours(a, [poly], 0, 1, 1))
        plt.savefig('src/1.png', bbox_inches='tight', transparent=True, pad_inches=0)
        plt.show()

        plt.figure(dpi=300)
        plt.axis('off')
        a = cv2.drawMarker(ans.copy()*0.5, tuple(sub_center.astype(int)), [1], markerType=0, markerSize=10, thickness=2)
        plt.imshow(cv2.drawContours(a, [poly], 0, 1, 1))
        # plt.show()
        plt.savefig('src/2.png', bbox_inches='tight', transparent=True, pad_inches=0)
        plt.show()

    x_lst.append([poly[:, 0, 0].max(), 0])
    y_lst.append([poly[:, 0, 1].max(), 0])
    x_lst.sort(key=lambda x: x[0])
    y_lst.sort(key=lambda x: x[0])

    diag = math.sqrt(math.pow(floorplan_sub.shape[1], 2) + math.pow(floorplan_sub.shape[0], 2))
    x_lst = merge_near(x_lst, diag, poly[:, 0, 0].min())
    y_lst = merge_near(y_lst, diag, poly[:, 0, 1].min())
    if need_cube and len(x_lst) > 2:
        x_lst = [x_lst[0], x_lst[-1]]
    if need_cube and len(y_lst) > 2:
        y_lst = [y_lst[0], y_lst[-1]]

    for i in range(len(x_lst) - 1):
        for j in range(len(y_lst) - 1):
            sample = floorplan_sub[y_lst[j]:y_lst[j + 1], x_lst[i]:x_lst[i + 1]]
            score = 0 if sample.size == 0 else sample.mean()
            if score >= 0.3:
                ans[y_lst[j]:y_lst[j + 1], x_lst[i]:x_lst[i + 1]] = 1

    if debug_show:
        plt.figure(dpi=300)
        plt.axis('off')
        a = cv2.drawMarker(ans.copy() * 0.5, tuple(sub_center.astype(int)), [1],
                           markerType=0, markerSize=10, thickness=2)
        plt.imshow(cv2.drawContours(a, [poly], 0, 1, 1))
        # plt.show()
        plt.savefig('src/3.png', bbox_inches='tight', transparent=True, pad_inches=0)
        plt.show()

    pred = np.uint8(ans)
    pred_polys = cv2.findContours(pred, 1, 3)
    if isinstance(pred_polys, tuple):
        if len(pred_polys) == 3:
            pred_polys = pred_polys[1]
        else:
            pred_polys = pred_polys[0]

    pred_polys.sort(key=lambda x: cv2.contourArea(x), reverse=True)
    pred_poly = pred_polys[0]
    # findContours may produce errors, which are enforced here 
    for i in range(len(pred_poly)):
        p1 = pred_poly[i][0]
        p2 = pred_poly[(i+1)%len(pred_poly)][0]
        if abs(p1[0] - p2[0]) < abs(p1[1] - p2[1]):
            p1[0] = p2[0]
        else:
            p1[1] = p2[1]

    if debug_show:
        plt.figure(dpi=300)
        plt.axis('off')
        a = cv2.drawMarker(ans.copy() * 0.5, tuple(sub_center.astype(int)), [1],
                           markerType=0, markerSize=10, thickness=2)
        a = cv2.drawContours(a, [poly], 0, 0.8, 1)
        a = cv2.drawContours(a, [pred_poly], 0, 1, 1)
        plt.imshow(a)
        # plt.show()
        plt.savefig('src/4.png', bbox_inches='tight', transparent=True, pad_inches=0)
        plt.show()

    polygon = [(p[0][1], p[0][0]) for p in pred_poly[::-1]]

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

    show_process = False
    if show_process:
        img = np.zeros((floorplan_sub.shape[0], floorplan_sub.shape[1], 3))
        for x in x_lst:
            cv2.line(img, (x, 0), (x, floorplan_sub.shape[0]), (0, 255, 0), 1)
        for y in y_lst:
            cv2.line(img, (0, y), (floorplan_sub.shape[1], y), (255, 0, 0), 1)

        fig = plt.figure()
        plt.axis('off')
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(floorplan)
        ax3 = fig.add_subplot(2, 2, 2)
        ax3.imshow(floorplan_sub)
        ax4 = fig.add_subplot(2, 2, 3)
        ax4.imshow(img)
        ax5 = fig.add_subplot(2, 2, 4)
        ax5.imshow(ans)
        plt.show()

    return pred_xz


if __name__ == '__main__':
    # processed_xz = fit_layout(floor_xz=np.load('/room_layout_estimation/lgt_net/floor_xz.npy'), need_cube=False, show=False)

    from utils.conversion import uv2xyz

    pano_img = np.zeros([512, 1024, 3])
    corners = np.array([[0.1, 0.7],
                        [0.4, 0.7],
                        [0.3, 0.6],
                        [0.6, 0.6],
                        [0.8, 0.7]])
    xz = uv2xyz(corners)[..., ::2]
    draw_floorplan(xz, show=True, marker_color=None, center_color=0.8)

    xz = fit_layout(xz)
    draw_floorplan(xz, show=True, marker_color=None, center_color=0.8)
