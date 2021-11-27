""" 
@Date: 2021/08/12
@description:
"""

import torch
import torch.nn as nn
import numpy as np

from visualization.grad import get_all


class GradLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
        self.cos = nn.CosineSimilarity(dim=-1, eps=0)

        self.grad_conv = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=0, bias=False, padding_mode='circular')
        self.grad_conv.weight = nn.Parameter(torch.tensor([[[1, 0, -1]]]).float())
        self.grad_conv.weight.requires_grad = False

    def forward(self, gt, dt):
        gt_direction, _, gt_angle_grad = get_all(gt['depth'], self.grad_conv)
        dt_direction, _, dt_angle_grad = get_all(dt['depth'], self.grad_conv)

        normal_loss = (1 - self.cos(gt_direction, dt_direction)).mean()
        grad_loss = self.loss(gt_angle_grad, dt_angle_grad)
        return [normal_loss, grad_loss]


if __name__ == '__main__':
    from dataset.mp3d_dataset import MP3DDataset
    from utils.boundary import depth2boundaries
    from utils.conversion import uv2xyz
    from visualization.boundary import draw_boundaries
    from visualization.floorplan import draw_floorplan

    def show_boundary(image, depth, ratio):
        boundary_list = depth2boundaries(ratio, depth, step=None)
        draw_boundaries(image.transpose(1, 2, 0), boundary_list=boundary_list, show=True)
        draw_floorplan(uv2xyz(boundary_list[0])[..., ::2], show=True, center_color=0.8)

    mp3d_dataset = MP3DDataset(root_dir='../src/dataset/mp3d', mode='train', patch_num=256)
    gt = mp3d_dataset.__getitem__(1)
    gt['depth'] = torch.from_numpy(gt['depth'][np.newaxis])  # batch size is 1
    dummy_dt = {
        'depth': gt['depth'].clone(),
    }
    # dummy_dt['depth'][..., 20] *= 3  # some different

    # show_boundary(gt['image'], gt['depth'][0].numpy(), gt['ratio'])
    # show_boundary(gt['image'], dummy_dt['depth'][0].numpy(), gt['ratio'])

    grad_loss = GradLoss()
    loss = grad_loss(gt, dummy_dt)
    print(loss)
