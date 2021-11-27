"""
@date: 2021/7/5
@description:
"""
import json
import math
import shutil

import numpy as np
from utils.boundary import *
import dataset
import os
from tqdm import tqdm
from PIL import Image
from visualization.boundary import *
from visualization.floorplan import *
from shapely.geometry import Polygon, Point


def filter_center(ceil_corners):
    xyz = uv2xyz(ceil_corners, plan_y=1.6)
    xz = xyz[:, ::2]
    poly = Polygon(xz).buffer(-0.01)
    return poly.contains(Point(0, 0))


def filter_boundary(corners):
    if is_ceil_boundary(corners):
        return True
    elif is_floor_boundary(corners):
        return True
    else:
        # An intersection occurs and an exception is considered
        return False


def filter_self_intersection(corners):
    xz = uv2xyz(corners)[:, ::2]
    poly = Polygon(xz)
    return poly.is_valid


def filter_dataset(dataset, show=False, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(dataset.root_dir, dataset.mode)
        output_img_dir = os.path.join(output_dir, 'img_align')
        output_label_dir = os.path.join(output_dir, 'label_cor_align')
    else:
        output_dir = os.path.join(output_dir, dataset.mode)
        output_img_dir = os.path.join(output_dir, 'img')
        output_label_dir = os.path.join(output_dir, 'label_cor')

    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)

    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    bar = tqdm(dataset, total=len(dataset))
    for data in bar:
        name = data['name']
        bar.set_description(f"Processing {name}")
        img = data['img']
        corners = data['corners']

        if not filter_center(corners[1::2]):
            if show:
                draw_boundaries(img, corners_list=[corners[0::2], corners[1::2]], show=True)
            if not os.path.exists(data['img_path']):
                print("already remove")
            else:
                print(f"move {name}")
                shutil.move(data['img_path'], os.path.join(output_img_dir, os.path.basename(data['img_path'])))
                shutil.move(data['label_path'], os.path.join(output_label_dir, os.path.basename(data['label_path'])))


def execute_filter_dataset(root_dir, dataset_name="PanoS2D3DDataset", modes=None, output_dir=None):
    if modes is None:
        modes = ["train", "test", "valid"]

    for mode in modes:
        print("mode: {}".format(mode))

        filter_dataset(getattr(dataset, dataset_name)(root_dir, mode), show=False, output_dir=output_dir)


if __name__ == '__main__':
    execute_filter_dataset(root_dir='/root/data/hd/hnet_dataset',
                           dataset_name="PanoS2D3DDataset", modes=['train', "test", "valid"],
                           output_dir='/root/data/hd/hnet_dataset_close')
