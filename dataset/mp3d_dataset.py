"""
@date: 2021/6/25
@description:
"""
import os
import json

from dataset.communal.read import read_image, read_label
from dataset.communal.base_dataset import BaseDataset
from utils.logger import get_logger


class MP3DDataset(BaseDataset):
    def __init__(self, root_dir, mode, shape=None, max_wall_num=0, aug=None, camera_height=1.6, logger=None,
                 split_list=None, patch_num=256, keys=None, for_test_index=None):
        super().__init__(mode, shape, max_wall_num, aug, camera_height, patch_num, keys)

        if logger is None:
            logger = get_logger()
        self.root_dir = root_dir

        split_dir = os.path.join(root_dir, 'split')
        label_dir = os.path.join(root_dir, 'label')
        img_dir = os.path.join(root_dir, 'image')

        if split_list is None:
            with open(os.path.join(split_dir, f"{mode}.txt"), 'r') as f:
                split_list = [x.rstrip().split() for x in f]

        split_list.sort()
        if for_test_index is not None:
            split_list = split_list[:for_test_index]

        self.data = []
        invalid_num = 0
        for name in split_list:
            name = "_".join(name)
            img_path = os.path.join(img_dir, f"{name}.png")
            label_path = os.path.join(label_dir, f"{name}.json")

            if not os.path.exists(img_path):
                logger.warning(f"{img_path} not exists")
                invalid_num += 1
                continue
            if not os.path.exists(label_path):
                logger.warning(f"{label_path} not exists")
                invalid_num += 1
                continue

            with open(label_path, 'r') as f:
                label = json.load(f)

                if self.max_wall_num >= 10:
                    if label['layoutWalls']['num'] < self.max_wall_num:
                        invalid_num += 1
                        continue
                elif self.max_wall_num != 0 and label['layoutWalls']['num'] != self.max_wall_num:
                    invalid_num += 1
                    continue

            # print(label['layoutWalls']['num'])
            self.data.append([img_path, label_path])

        logger.info(
            f"Build dataset mode: {self.mode} max_wall_num: {self.max_wall_num} valid: {len(self.data)} invalid: {invalid_num}")

    def __getitem__(self, idx):
        rgb_path, label_path = self.data[idx]
        label = read_label(label_path, data_type='MP3D')
        image = read_image(rgb_path, self.shape)
        output = self.process_data(label, image, self.patch_num)
        return output


if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    from tqdm import tqdm
    from visualization.boundary import draw_boundaries
    from visualization.floorplan import draw_floorplan
    from utils.boundary import depth2boundaries
    from utils.conversion import uv2xyz

    modes = ['test', 'val']
    for i in range(1):
        for mode in modes:
            print(mode)
            mp3d_dataset = MP3DDataset(root_dir='../src/dataset/mp3d', mode=mode, aug={
                'STRETCH': True,
                'ROTATE': True,
                'FLIP': True,
                'GAMMA': True
            })
            save_dir = f'../src/dataset/mp3d/visualization/{mode}'
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            bar = tqdm(mp3d_dataset, ncols=100)
            for data in bar:
                bar.set_description(f"Processing {data['id']}")
                boundary_list = depth2boundaries(data['ratio'], data['depth'], step=None)
                pano_img = draw_boundaries(data['image'].transpose(1, 2, 0), boundary_list=boundary_list, show=True)
                Image.fromarray((pano_img * 255).astype(np.uint8)).save(
                    os.path.join(save_dir, f"{data['id']}_boundary.png"))

                floorplan = draw_floorplan(uv2xyz(boundary_list[0])[..., ::2], show=True,
                                           marker_color=None, center_color=0.8, show_radius=None)
                Image.fromarray((floorplan.squeeze() * 255).astype(np.uint8)).save(
                    os.path.join(save_dir, f"{data['id']}_floorplan.png"))
