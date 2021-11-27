"""
@date: 2021/6/16
@description:
"""
import math
import os
import numpy as np

from dataset.communal.read import read_image, read_label
from dataset.communal.base_dataset import BaseDataset
from utils.logger import get_logger


class PanoS2D3DDataset(BaseDataset):
    def __init__(self, root_dir, mode, shape=None, max_wall_num=0, aug=None, camera_height=1.6, logger=None,
                 split_list=None, patch_num=256, keys=None, for_test_index=None, subset=None):
        super().__init__(mode, shape, max_wall_num, aug, camera_height, patch_num, keys)

        if logger is None:
            logger = get_logger()
        self.root_dir = root_dir

        if mode is None:
            return
        label_dir = os.path.join(root_dir, 'valid' if mode == 'val' else mode, 'label_cor')
        img_dir = os.path.join(root_dir, 'valid' if mode == 'val' else mode, 'img')

        if split_list is None:
            split_list = [name.split('.')[0] for name in os.listdir(label_dir) if
                          not name.startswith('.') and name.endswith('txt')]

        split_list.sort()

        assert subset == 'pano' or subset == 's2d3d' or subset is None, 'error subset'
        if subset == 'pano':
            split_list = [name for name in split_list if 'pano_' in name]
            logger.info(f"Use PanoContext Dataset")
        elif subset == 's2d3d':
            split_list = [name for name in split_list if 'camera_' in name]
            logger.info(f"Use Stanford2D3D Dataset")

        if for_test_index is not None:
            split_list = split_list[:for_test_index]

        self.data = []
        invalid_num = 0
        for name in split_list:
            img_path = os.path.join(img_dir, f"{name}.png")
            label_path = os.path.join(label_dir, f"{name}.txt")

            if not os.path.exists(img_path):
                logger.warning(f"{img_path} not exists")
                invalid_num += 1
                continue
            if not os.path.exists(label_path):
                logger.warning(f"{label_path} not exists")
                invalid_num += 1
                continue

            with open(label_path, 'r') as f:
                lines = [line for line in f.readlines() if
                         len([c for c in line.split(' ') if c[0].isnumeric()]) > 1]
                if len(lines) % 2 != 0:
                    invalid_num += 1
                    continue
            self.data.append([img_path, label_path])

        logger.info(
            f"Build dataset mode: {self.mode} valid: {len(self.data)} invalid: {invalid_num}")

    def __getitem__(self, idx):
        rgb_path, label_path = self.data[idx]
        label = read_label(label_path, data_type='Pano_S2D3D')
        image = read_image(rgb_path, self.shape)
        output = self.process_data(label, image, self.patch_num)
        return output


if __name__ == '__main__':

    modes = ['test', 'val', 'train']
    for i in range(1):
        for mode in modes:
            print(mode)
            mp3d_dataset = PanoS2D3DDataset(root_dir='../src/dataset/pano_s2d3d', mode=mode, aug={
                # 'STRETCH': True,
                # 'ROTATE': True,
                # 'FLIP': True,
                # 'GAMMA': True
            })
            continue
            save_dir = f'../src/dataset/pano_s2d3d/visualization/{mode}'
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            bar = tqdm(mp3d_dataset, ncols=100)
            for data in bar:
                bar.set_description(f"Processing {data['id']}")
                boundary_list = depth2boundaries(data['ratio'], data['depth'], step=None)
                pano_img = draw_boundaries(data['image'].transpose(1, 2, 0), boundary_list=boundary_list, show=False)
                Image.fromarray((pano_img * 255).astype(np.uint8)).save(
                    os.path.join(save_dir, f"{data['id']}_boundary.png"))

                floorplan = draw_floorplan(uv2xyz(boundary_list[0])[..., ::2], show=False,
                                           marker_color=None, center_color=0.8, show_radius=None)
                Image.fromarray((floorplan.squeeze() * 255).astype(np.uint8)).save(
                    os.path.join(save_dir, f"{data['id']}_floorplan.png"))
