"""
@date: 2021/6/16
@description:
"""

import os

from dataset.pano_s2d3d_dataset import PanoS2D3DDataset
from utils.logger import get_logger


class PanoS2D3DMixDataset(PanoS2D3DDataset):
    def __init__(self, root_dir, mode, shape=None, max_wall_num=0, aug=None, camera_height=1.6, logger=None,
                 split_list=None, patch_num=256, keys=None, for_test_index=None, subset=None):
        assert subset == 's2d3d' or subset == 'pano', 'error subset'
        super().__init__(root_dir, None, shape, max_wall_num, aug, camera_height, logger,
                         split_list, patch_num, keys, None, subset)
        if logger is None:
            logger = get_logger()
        self.mode = mode
        if mode == 'train':
            if subset == 'pano':
                s2d3d_train_data = PanoS2D3DDataset(root_dir, 'train', shape, max_wall_num, aug, camera_height, logger,
                                                    split_list, patch_num, keys, None, 's2d3d').data
                s2d3d_val_data = PanoS2D3DDataset(root_dir, 'val', shape, max_wall_num, aug, camera_height, logger,
                                                  split_list, patch_num, keys, None, 's2d3d').data
                s2d3d_test_data = PanoS2D3DDataset(root_dir, 'test', shape, max_wall_num, aug, camera_height, logger,
                                                   split_list, patch_num, keys, None, 's2d3d').data
                s2d3d_all_data = s2d3d_train_data + s2d3d_val_data + s2d3d_test_data

                pano_train_data = PanoS2D3DDataset(root_dir, 'train', shape, max_wall_num, aug, camera_height, logger,
                                                   split_list, patch_num, keys, None, 'pano').data
                self.data = s2d3d_all_data + pano_train_data
            elif subset == 's2d3d':
                pano_train_data = PanoS2D3DDataset(root_dir, 'train', shape, max_wall_num, aug, camera_height, logger,
                                                   split_list, patch_num, keys, None, 'pano').data
                pano_val_data = PanoS2D3DDataset(root_dir, 'val', shape, max_wall_num, aug, camera_height, logger,
                                                 split_list, patch_num, keys, None, 'pano').data
                pano_test_data = PanoS2D3DDataset(root_dir, 'test', shape, max_wall_num, aug, camera_height, logger,
                                                  split_list, patch_num, keys, None, 'pano').data
                pano_all_data = pano_train_data + pano_val_data + pano_test_data

                s2d3d_train_data = PanoS2D3DDataset(root_dir, 'train', shape, max_wall_num, aug, camera_height, logger,
                                                    split_list, patch_num, keys, None, 's2d3d').data
                self.data = pano_all_data + s2d3d_train_data
        else:
            self.data = PanoS2D3DDataset(root_dir, mode, shape, max_wall_num, aug, camera_height, logger,
                                         split_list, patch_num, keys, None, subset).data

        if for_test_index is not None:
            self.data = self.data[:for_test_index]
        logger.info(f"Build dataset mode: {self.mode}  valid: {len(self.data)}")


if __name__ == '__main__':
    import numpy as np
    from PIL import Image

    from tqdm import tqdm
    from visualization.boundary import draw_boundaries
    from visualization.floorplan import draw_floorplan
    from utils.boundary import depth2boundaries
    from utils.conversion import uv2xyz

    modes = ['test', 'val', 'train']
    for i in range(1):
        for mode in modes:
            print(mode)
            mp3d_dataset = PanoS2D3DMixDataset(root_dir='../src/dataset/pano_s2d3d', mode=mode, aug={
                # 'STRETCH': True,
                # 'ROTATE': True,
                # 'FLIP': True,
                # 'GAMMA': True
            }, subset='pano')
            continue
            save_dir = f'../src/dataset/pano_s2d3d/visualization1/{mode}'
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
