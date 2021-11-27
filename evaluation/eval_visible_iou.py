""" 
@Date: 2021/08/02
@description:
The 2DIoU for calculating the visible and full boundaries, such as the MP3D dataset,
has the following data: {'train': 0.9775843958583535, 'test': 0.9828616219607289, 'val': 0.9883810438132491},
indicating that our best performance is limited to below 98.29% 2DIoU using our approach.
"""
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from evaluation.iou import calc_IoU_2D
from visualization.floorplan import draw_iou_floorplan
from utils.conversion import depth2xyz, uv2xyz


def eval_dataset_visible_IoU(dataset, show=False):
    bar = tqdm(dataset, total=len(dataset), ncols=100)
    iou2ds = []
    for data in bar:
        bar.set_description(f"Processing {data['id']}")
        corners = data['corners']
        corners = corners[corners[..., 0] + corners[..., 1] != 0]  # Take effective corners
        all_xz = uv2xyz(corners)[..., ::2]
        visible_xz = depth2xyz(data['depth'])[..., ::2]
        iou2d = calc_IoU_2D(all_xz, visible_xz)
        iou2ds.append(iou2d)
        if show:
            layout_floorplan = draw_iou_floorplan(all_xz, visible_xz, iou2d=iou2d)
            plt.imshow(layout_floorplan)
            plt.show()

    mean_iou2d = np.array(iou2ds).mean()
    return mean_iou2d


def execute_eval_dataset_visible_IoU(root_dir, dataset, modes=None):
    if modes is None:
        modes = ["train", "test", "valid"]

    iou2d_d = {}
    for mode in modes:
        print("mode: {}".format(mode))
        iou2d = eval_dataset_visible_IoU(dataset(root_dir, mode, patch_num=1024,
                                                 keys=['depth', 'visible_corners', 'corners', 'id']), show=False)
        iou2d_d[mode] = iou2d
    return iou2d_d


if __name__ == '__main__':
    from dataset.mp3d_dataset import MP3DDataset

    iou2d_d = execute_eval_dataset_visible_IoU(root_dir='../src/dataset/mp3d',
                                               dataset=MP3DDataset,
                                               modes=['train', 'test', 'val'])
    print(iou2d_d)
