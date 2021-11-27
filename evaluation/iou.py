"""
@date: 2021/6/29
@description:
The method with "_floorplan" suffix is only for comparison, which is used for calculation in LED2-net.
However, the floorplan is affected by show_radius. Setting too large will result in the decrease of accuracy,
and setting too small will result in the failure of calculation beyond the range.
"""
import numpy as np
from shapely.geometry import Polygon


def calc_inter_area(dt_xz, gt_xz):
    """
    :param dt_xz: Prediction boundaries can also be corners, format: [[x1, z1], [x2, z2], ...]
    :param gt_xz: Ground truth boundaries can also be corners, format: [[x1, z1], [x2, z2], ...]
    :return:
    """
    dt_polygon = Polygon(dt_xz)
    gt_polygon = Polygon(gt_xz)

    dt_area = dt_polygon.area
    gt_area = gt_polygon.area
    inter_area = dt_polygon.intersection(gt_polygon).area
    return dt_area, gt_area, inter_area


def calc_IoU_2D(dt_xz, gt_xz):
    """
    :param dt_xz: Prediction boundaries can also be corners, format: [[x1, z1], [x2, z2], ...]
    :param gt_xz: Ground truth boundaries can also be corners, format: [[x1, z1], [x2, z2], ...]
    :return:
    """
    dt_area, gt_area, inter_area = calc_inter_area(dt_xz, gt_xz)
    iou_2d = inter_area / (gt_area + dt_area - inter_area)
    return iou_2d


def calc_IoU_3D(dt_xz, gt_xz, dt_height, gt_height):
    """
    :param dt_xz: Prediction boundaries can also be corners, format: [[x1, z1], [x2, z2], ...]
    :param gt_xz: Ground truth boundaries can also be corners, format: [[x1, z1], [x2, z2], ...]
    :param dt_height:
    :param gt_height:
    :return:
    """
    dt_area, gt_area, inter_area = calc_inter_area(dt_xz, gt_xz)
    dt_volume = dt_area * dt_height
    gt_volume = gt_area * gt_height
    inter_volume = inter_area * min(dt_height, gt_height)
    iou_3d = inter_volume / (dt_volume + gt_volume - inter_volume)
    return iou_3d


def calc_IoU(dt_xz, gt_xz, dt_height, gt_height):
    """
    :param dt_xz: Prediction boundaries can also be corners, format: [[x1, z1], [x2, z2], ...]
    :param gt_xz: Ground truth boundaries can also be corners, format: [[x1, z1], [x2, z2], ...]
    :param dt_height:
    :param gt_height:
    :return:
    """
    dt_area, gt_area, inter_area = calc_inter_area(dt_xz, gt_xz)
    iou_2d = inter_area / (gt_area + dt_area - inter_area)

    dt_volume = dt_area * dt_height
    gt_volume = gt_area * gt_height
    inter_volume = inter_area * min(dt_height, gt_height)
    iou_3d = inter_volume / (dt_volume + gt_volume - inter_volume)

    return iou_2d, iou_3d


def calc_Iou_height(dt_height, gt_height):
    return min(dt_height, gt_height) / max(dt_height, gt_height)


# the following is for testing only
def calc_inter_area_floorplan(dt_floorplan, gt_floorplan):
    intersect = np.sum(np.logical_and(dt_floorplan, gt_floorplan))
    dt_area = np.sum(dt_floorplan)
    gt_area = np.sum(gt_floorplan)
    return dt_area, gt_area, intersect


def calc_IoU_2D_floorplan(dt_floorplan, gt_floorplan):
    dt_area, gt_area, inter_area = calc_inter_area_floorplan(dt_floorplan, gt_floorplan)
    iou_2d = inter_area / (gt_area + dt_area - inter_area)
    return iou_2d


def calc_IoU_3D_floorplan(dt_floorplan, gt_floorplan, dt_height, gt_height):
    dt_area, gt_area, inter_area = calc_inter_area_floorplan(dt_floorplan, gt_floorplan)
    dt_volume = dt_area * dt_height
    gt_volume = gt_area * gt_height
    inter_volume = inter_area * min(dt_height, gt_height)
    iou_3d = inter_volume / (dt_volume + gt_volume - inter_volume)
    return iou_3d


def calc_IoU_floorplan(dt_floorplan, gt_floorplan, dt_height, gt_height):
    dt_area, gt_area, inter_area = calc_inter_area_floorplan(dt_floorplan, gt_floorplan)
    iou_2d = inter_area / (gt_area + dt_area - inter_area)

    dt_volume = dt_area * dt_height
    gt_volume = gt_area * gt_height
    inter_volume = inter_area * min(dt_height, gt_height)
    iou_3d = inter_volume / (dt_volume + gt_volume - inter_volume)
    return iou_2d, iou_3d


if __name__ == '__main__':
    from visualization.floorplan import draw_floorplan, draw_iou_floorplan
    from visualization.boundary import draw_boundaries, corners2boundaries
    from utils.conversion import uv2xyz
    from utils.height import height2ratio

    # dummy data
    dt_floor_corners = np.array([[0.2, 0.7],
                                 [0.4, 0.7],
                                 [0.6, 0.7],
                                 [0.8, 0.7]])
    dt_height = 2.8

    gt_floor_corners = np.array([[0.3, 0.7],
                                 [0.5, 0.7],
                                 [0.7, 0.7],
                                 [0.9, 0.7]])
    gt_height = 3.2

    dt_xz = uv2xyz(dt_floor_corners)[..., ::2]
    gt_xz = uv2xyz(gt_floor_corners)[..., ::2]

    dt_floorplan = draw_floorplan(dt_xz, show=False, show_radius=1)
    gt_floorplan = draw_floorplan(gt_xz, show=False, show_radius=1)
    # dt_floorplan = draw_floorplan(dt_xz, show=False, show_radius=2)
    # gt_floorplan = draw_floorplan(gt_xz, show=False, show_radius=2)

    iou_2d, iou_3d = calc_IoU_floorplan(dt_floorplan, gt_floorplan, dt_height, gt_height)
    print('use floor plan image:', iou_2d, iou_3d)

    iou_2d, iou_3d = calc_IoU(dt_xz, gt_xz, dt_height, gt_height)
    print('use floor plan polygon:', iou_2d, iou_3d)

    draw_iou_floorplan(dt_xz, gt_xz, show=True, iou_2d=iou_2d, iou_3d=iou_3d)
    pano_bd = draw_boundaries(np.zeros([512, 1024, 3]), corners_list=[dt_floor_corners],
                              boundary_color=[0, 0, 1], ratio=height2ratio(dt_height), draw_corners=False)
    pano_bd = draw_boundaries(pano_bd, corners_list=[gt_floor_corners],
                              boundary_color=[0, 1, 0], ratio=height2ratio(gt_height), show=True, draw_corners=False)
