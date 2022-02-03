""" 
@Date: 2022/01/31
@description:
ZInd:
{'test': {'mw': 2789, 'aw': 381}, 'train': {'mw': 21228, 'aw': 3654}, 'val': {'mw': 2647, 'aw': 433}}

"""
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from evaluation.iou import calc_IoU_2D
from visualization.floorplan import draw_floorplan
from visualization.boundary import draw_boundaries
from utils.conversion import depth2xyz, uv2xyz


def analyse_layout_type(dataset, show=False):
    bar = tqdm(dataset, total=len(dataset), ncols=100)
    mw = 0
    aw = 0
    for data in bar:
        bar.set_description(f"Processing {data['id']}")
        corners = data['corners']
        corners = corners[corners[..., 0] + corners[..., 1] != 0]  # Take effective corners
        all_xz = uv2xyz(corners)[..., ::2]

        c = len(all_xz)
        flag = False
        for i in range(c - 1):
            l1 = all_xz[i + 1] - all_xz[i]
            l2 = all_xz[(i + 2) % c] - all_xz[i + 1]
            dot = np.dot(l1, l2)/(np.linalg.norm(l1)*np.linalg.norm(l2))
            if 0.9 > abs(dot) > 0.1:
                flag = True
                break
        if flag:
            aw += 1
        else:
            mw += 1

        if flag and show:
            draw_floorplan(all_xz, show=True)
            draw_boundaries(data['image'].transpose(1, 2, 0), [corners], ratio=data['ratio'], show=True)

    return {'mw': mw, "aw": aw}


def execute_analyse_layout_type(root_dir, dataset, modes=None):
    if modes is None:
        modes = ["test", "train", "val"]

    iou2d_d = {}
    for mode in modes:
        print("mode: {}".format(mode))
        types = analyse_layout_type(dataset(root_dir, mode), show=False)
        iou2d_d[mode] = types
        print(types)
    return iou2d_d


if __name__ == '__main__':
    from dataset.zind_dataset import ZindDataset
    from dataset.mp3d_dataset import MP3DDataset

    iou2d_d = execute_analyse_layout_type(root_dir='../src/dataset/mp3d',
                                          dataset=MP3DDataset)
    print(iou2d_d)
