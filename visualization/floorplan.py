"""
@date: 2021/6/29
@description:
"""
import cv2


import matplotlib.pyplot as plt

from PIL import Image
from utils.boundary import *


def draw_floorplan(xz, fill_color=None, border_color=None, side_l=512, show_radius=None, show=False, marker_color=None,
                   center_color=None, scale=1.5):
    """
    :param scale:
    :param center_color:
    :param marker_color: for corners marking
    :param fill_color:
    :param border_color: boundary color
    :param xz: [[x1, z1], [x2, z2], ....]
    :param side_l: side length (pixel) of the output result
    :param show_radius: The displayed maximum radius m (proportional to the projection plane plan_y of xz),
    such as set to 1, means that the pixel value of side_l/2 is expressed as 1m, if not set this value to display all
    :param show:
    :return:
    """
    if fill_color is None:
        fill_color = [1]

    board = np.zeros([side_l, side_l, len(fill_color)], dtype=np.float)

    if show_radius is None:
        show_radius = np.linalg.norm(xz, axis=-1).max()

    xz = xz * side_l / (2*scale) / show_radius
    # v<-----------|o
    # |     |      |
    # | ----|----z |
    # |     |      |
    # |     x     \|/
    # |------------u
    xz[:, 1] = -xz[:, 1]
    xz += side_l // 2  # moving to center
    xz = xz.astype(np.int)
    cv2.fillPoly(board, [xz], fill_color)
    if border_color:
        cv2.drawContours(board, [xz], 0, border_color, 2)

    if marker_color is not None:
        for p in xz:
            cv2.drawMarker(board, tuple(p), marker_color, markerType=0, markerSize=10, thickness=2)
    if center_color is not None:
        cv2.drawMarker(board, tuple([side_l // 2, side_l // 2]), center_color, markerType=0, markerSize=10, thickness=2)

    if show:
        # plt.rcParams['figure.dpi'] = 300
        plt.axis('off')
        plt.imshow(board[..., 0] if board.shape[-1] == 1 else board)
        plt.show()

    return board


def draw_iou_floorplan(dt_xz, gt_xz, show_radius=None, show=False, side_l=512,
                       iou_2d=None, iou_3d=None, dt_board_color=None, gt_board_color=None):
    """
    :param gt_board_color:
    :param dt_board_color:
    :param dt_xz: [[x1, z1], [x2, z2], ....]
    :param gt_xz: [[x1, z1], [x2, z2], ....]
    :param show:
    :param side_l: side length (pixel) of the output result
    :param show_radius: The displayed maximum radius m (proportional to the projection plane plan_y of xz),
    such as set to 1, means that the pixel value of side_l/2 is expressed as 1m, if not set this value to display all
    :param iou_2d:
    :param iou_3d:
    :return:
    """
    if dt_board_color is None:
        dt_board_color = [0, 1, 0, 1]
    if gt_board_color is None:
        gt_board_color = [0, 0, 1, 1]
    center_color = [1, 0, 0, 1]
    fill_color = [0.2, 0.2, 0.2, 0.2]

    if show_radius is None:
        # niform scale
        gt_radius = np.linalg.norm(gt_xz, axis=-1).max()
        dt_radius = np.linalg.norm(dt_xz, axis=-1).max()
        show_radius = gt_radius if gt_radius > dt_radius else dt_radius

    dt_floorplan = draw_floorplan(dt_xz, show_radius=show_radius, fill_color=fill_color,
                                  border_color=dt_board_color, side_l=side_l, show=False)
    gt_floorplan = draw_floorplan(gt_xz, show_radius=show_radius, fill_color=fill_color,
                                  border_color=gt_board_color, side_l=side_l, show=False,
                                  center_color=[1, 0, 0, 1])

    dt_floorplan = Image.fromarray((dt_floorplan * 255).astype(np.uint8), mode='RGBA')
    gt_floorplan = Image.fromarray((gt_floorplan * 255).astype(np.uint8), mode='RGBA')
    iou_floorplan = Image.alpha_composite(gt_floorplan, dt_floorplan)

    back = np.zeros([side_l, side_l, len(fill_color)], dtype=np.float)
    back[..., :] = [0.8, 0.8, 0.8, 1]
    back = Image.fromarray((back * 255).astype(np.uint8), mode='RGBA')

    iou_floorplan = Image.alpha_composite(back, iou_floorplan).convert("RGB")
    iou_floorplan = np.array(iou_floorplan) / 255.0

    if iou_2d is not None:
        cv2.putText(iou_floorplan, f'2d:{iou_2d * 100:.2f}', (10, 30), 2, 1, (0, 0, 0), 1)
    if iou_3d is not None:
        cv2.putText(iou_floorplan, f'3d:{iou_3d * 100:.2f}', (10, 60), 2, 1, (0, 0, 0), 1)

    if show:
        plt.axis('off')
        plt.imshow(iou_floorplan)
        plt.show()
    return iou_floorplan


if __name__ == '__main__':
    import numpy as np
    from dataset.mp3d_dataset import MP3DDataset
    from utils.boundary import depth2boundaries
    from utils.conversion import uv2xyz
    from visualization.boundary import draw_boundaries

    mp3d_dataset = MP3DDataset(root_dir='../src/dataset/mp3d', mode='train')
    gt = mp3d_dataset.__getitem__(0)

    # boundary_list = depth2boundaries(gt['ratio'], gt['depth'], step=None)
    # pano_img = draw_boundaries(gt['image'].transpose(1, 2, 0), boundary_list=boundary_list, show=True)
    # draw_floorplan(uv2xyz(boundary_list[0])[..., ::2], show=True, marker_color=None, center_color=0.8)
    # draw_floorplan(depth2xyz(gt['depth'])[..., ::2], show=True, marker_color=None, center_color=0.8)

    corners = gt['corners'][gt['corners'][..., 0] + gt['corners'][..., 1] != 0]
    dt_corners = corners + 0.1
    # img = draw_floorplan(uv2xyz(corners)[..., ::2], show=True, fill_color=[0.8, 0.8, 0.8, 0.2],
    #                      marker_color=None, center_color=[1, 0, 0, 1], border_color=[0, 0, 1, 1])
    # cv2.imwrite('../src/fig/flp.png', (img*255).astype(np.uint8))

    img = draw_iou_floorplan(uv2xyz(dt_corners)[..., ::2], uv2xyz(corners)[..., ::2],  side_l=512, show=True)
    img[..., 0:3] = img[..., 0:3][..., ::-1]
    # cv2.imwrite('../src/fig/flp.png', (img*255).astype(np.uint8))

