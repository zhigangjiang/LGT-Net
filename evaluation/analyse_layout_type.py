""" 
@Date: 2022/01/31
@description:
ZInd:
{'test': {'mw': 2789, 'aw': 381}, 'train': {'mw': 21228, 'aw': 3654}, 'val': {'mw': 2647, 'aw': 433}}

"""
import numpy as np
import matplotlib.pyplot as plt
import json

from tqdm import tqdm
from evaluation.iou import calc_IoU_2D
from visualization.floorplan import draw_floorplan
from visualization.boundary import draw_boundaries
from utils.conversion import depth2xyz, uv2xyz


def analyse_layout_type(dataset, show=False):
    bar = tqdm(dataset, total=len(dataset), ncols=100)
    manhattan = 0
    atlanta = 0
    corner_type = {}
    for data in bar:
        bar.set_description(f"Processing {data['id']}")
        corners = data['corners']
        corners = corners[corners[..., 0] + corners[..., 1] != 0]  # Take effective corners
        corners_count = str(len(corners)) if len(corners) < 10 else "10"
        if corners_count not in corner_type:
            corner_type[corners_count] = 0
        corner_type[corners_count] += 1

        all_xz = uv2xyz(corners)[..., ::2]

        c = len(all_xz)
        flag = False
        for i in range(c - 1):
            l1 = all_xz[i + 1] - all_xz[i]
            l2 = all_xz[(i + 2) % c] - all_xz[i + 1]
            a = (np.linalg.norm(l1)*np.linalg.norm(l2))
            if a == 0:
                continue
            dot = np.dot(l1, l2)/a
            if 0.9 > abs(dot) > 0.1:
                # cos-1(0.1)=84.26 > angle > cos-1(0.9)=25.84 or
                # cos-1(-0.9)=154.16 > angle > cos-1(-0.1)=95.74
                flag = True
                break
        if flag:
            atlanta += 1
        else:
            manhattan += 1

        if flag and show:
            draw_floorplan(all_xz, show=True)
            draw_boundaries(data['image'].transpose(1, 2, 0), [corners], ratio=data['ratio'], show=True)

    corner_type = dict(sorted(corner_type.items(), key=lambda item: int(item[0])))
    return {'manhattan': manhattan, "atlanta": atlanta, "corner_type": corner_type}


def execute_analyse_layout_type(root_dir, dataset, modes=None):
    if modes is None:
        modes = ["train", "val", "test"]

    iou2d_d = {}
    for mode in modes:
        print("mode: {}".format(mode))
        types = analyse_layout_type(dataset(root_dir, mode), show=False)
        iou2d_d[mode] = types
        print(json.dumps(types, indent=4))
    return iou2d_d


if __name__ == '__main__':
    from dataset.zind_dataset import ZindDataset
    from dataset.mp3d_dataset import MP3DDataset

    iou2d_d = execute_analyse_layout_type(root_dir='../src/dataset/mp3d',
                                          dataset=MP3DDataset)
    # iou2d_d = execute_analyse_layout_type(root_dir='../src/dataset/zind',
    #                                       dataset=ZindDataset)
    print(json.dumps(iou2d_d, indent=4))
