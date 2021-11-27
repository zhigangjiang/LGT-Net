"""
@Date: 2021/07/26
@description:
"""
import numpy as np
import torch

from utils.boundary import corners2boundary, visibility_corners, get_heat_map
from utils.conversion import xyz2depth, uv2xyz, uv2pixel
from dataset.communal.data_augmentation import PanoDataAugmentation


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, mode, shape=None, max_wall_num=999, aug=None, camera_height=1.6, patch_num=256, keys=None):
        if keys is None or len(keys) == 0:
            keys = ['image', 'depth', 'ratio', 'id', 'corners']
        if shape is None:
            shape = [512, 1024]

        assert mode == 'train' or mode == 'val' or mode == 'test' or mode is None, 'unknown mode!'
        self.mode = mode
        self.keys = keys
        self.shape = shape
        self.pano_aug = None if aug is None or mode == 'val' else PanoDataAugmentation(aug)
        self.camera_height = camera_height
        self.max_wall_num = max_wall_num
        self.patch_num = patch_num
        self.data = None

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_depth(corners, plan_y=1, length=256, visible=True):
        visible_floor_boundary = corners2boundary(corners, length=length, visible=visible)
        # The horizon-depth relative to plan_y
        visible_depth = xyz2depth(uv2xyz(visible_floor_boundary, plan_y), plan_y)
        return visible_depth

    def process_data(self, label, image, patch_num):
        """
        :param label:
        :param image:
        :param patch_num:
        :return:
        """
        corners = label['corners']
        if self.pano_aug is not None:
            corners, image = self.pano_aug.execute_aug(corners, image if 'image' in self.keys else None)
        eps = 1e-3
        corners[:, 1] = np.clip(corners[:, 1], 0.5+eps, 1-eps)

        output = {}
        if 'image' in self.keys:
            image = image.transpose(2, 0, 1)
            output['image'] = image

        visible_corners = None
        if 'corner_class' in self.keys or 'depth' in self.keys:
            visible_corners = visibility_corners(corners)

        if 'depth' in self.keys:
            depth = self.get_depth(visible_corners, length=patch_num, visible=False)
            assert len(depth) == patch_num, f"{label['id']}, {len(depth)}, {self.pano_aug.parameters}, {corners}"
            output['depth'] = depth

        if 'ratio' in self.keys:
            # Why use ratio? Because when floor_height =y_plan=1, we only need to predict ceil_height(ratio).
            output['ratio'] = label['ratio']

        if 'id' in self.keys:
            output['id'] = label['id']

        if 'corners' in self.keys:
            # all corners for evaluating Full_IoU
            assert len(label['corners']) <= 32, "len(label['corners']):"+len(label['corners'])
            output['corners'] = np.zeros((32, 2), dtype=np.float32)
            output['corners'][:len(label['corners'])] = label['corners']

        if 'corner_heat_map' in self.keys:
            output['corner_heat_map'] = get_heat_map(visible_corners[..., 0])

        if 'object' in self.keys and 'objects' in label:
            output[f'object_heat_map'] = np.zeros((3, patch_num), dtype=np.float32)
            output['object_size'] = np.zeros((3, patch_num), dtype=np.float32)  # width, height, bottom_height
            for i, type in enumerate(label['objects']):
                if len(label['objects'][type]) == 0:
                    continue

                u_s = []
                for obj in label['objects'][type]:
                    center_u = obj['center_u']
                    u_s.append(center_u)
                    center_pixel_u = uv2pixel(np.array([center_u]), w=patch_num, axis=0)[0]
                    output['object_size'][0, center_pixel_u] = obj['width_u']
                    output['object_size'][1, center_pixel_u] = obj['height_v']
                    output['object_size'][2, center_pixel_u] = obj['boundary_v']
                output[f'object_heat_map'][i] = get_heat_map(np.array(u_s))

        return output


if __name__ == '__main__':
    from dataset.communal.read import read_image, read_label
    from visualization.boundary import draw_boundaries
    from utils.boundary import depth2boundaries
    from tqdm import trange

    # np.random.seed(0)
    dataset = BaseDataset()
    dataset.pano_aug = PanoDataAugmentation(aug={
        'STRETCH': True,
        'ROTATE': True,
        'FLIP': True,
    })
    # pano_img = read_image("../src/demo.png")
    # label = read_label("../src/demo.json")
    pano_img_path = "../../src/dataset/mp3d/image/yqstnuAEVhm_6589ad7a5a0444b59adbf501c0f0fe53.png"
    label_path = "../../src/dataset/mp3d/label/yqstnuAEVhm_6589ad7a5a0444b59adbf501c0f0fe53.json"
    pano_img = read_image(pano_img_path)
    label = read_label(label_path)

    # batch test
    for i in trange(1):
        output = dataset.process_data(label, pano_img, 256)
        boundary_list = depth2boundaries(output['ratio'], output['depth'], step=None)
        draw_boundaries(output['image'].transpose(1, 2, 0), boundary_list=boundary_list, show=True)
