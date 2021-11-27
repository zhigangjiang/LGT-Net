""" 
@Date: 2021/08/12
@description:
"""
import torch
import torch.nn as nn


class LEDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, gt, dt):
        camera_height = 1.6
        gt_depth = gt['depth'] * camera_height

        dt_ceil_depth = dt['ceil_depth'] * camera_height * gt['ratio']
        dt_floor_depth = dt['depth'] * camera_height

        ceil_loss = self.loss(gt_depth, dt_ceil_depth)
        floor_loss = self.loss(gt_depth, dt_floor_depth)

        loss = floor_loss + ceil_loss

        return loss


if __name__ == '__main__':
    import numpy as np
    from dataset.mp3d_dataset import MP3DDataset

    mp3d_dataset = MP3DDataset(root_dir='../src/dataset/mp3d', mode='train')
    gt = mp3d_dataset.__getitem__(0)

    gt['depth'] = torch.from_numpy(gt['depth'][np.newaxis])  # batch size is 1
    gt['ratio'] = torch.from_numpy(gt['ratio'][np.newaxis])  # batch size is 1

    dummy_dt = {
        'depth': gt['depth'].clone(),
        'ceil_depth': gt['depth'] / gt['ratio']
    }
    # dummy_dt['depth'][..., :20] *= 3  # some different

    led_loss = LEDLoss()
    loss = led_loss(gt, dummy_dt)
    print(loss)
