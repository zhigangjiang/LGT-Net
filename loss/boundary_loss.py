""" 
@Date: 2021/08/12
@description: For HorizonNet, using latitudes to calculate loss.
"""
import torch
import torch.nn as nn
from utils.conversion import depth2xyz, xyz2lonlat


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, gt, dt):
        gt_floor_xyz = depth2xyz(gt['depth'])
        gt_ceil_xyz = gt_floor_xyz.clone()
        gt_ceil_xyz[..., 1] = -gt['ratio']

        gt_floor_boundary = xyz2lonlat(gt_floor_xyz)[..., -1:]
        gt_ceil_boundary = xyz2lonlat(gt_ceil_xyz)[..., -1:]

        gt_boundary = torch.cat([gt_floor_boundary, gt_ceil_boundary], dim=-1).permute(0, 2, 1)
        dt_boundary = dt['boundary']

        loss = self.loss(gt_boundary, dt_boundary)
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
        'boundary': torch.cat([
            xyz2lonlat(depth2xyz(gt['depth']))[..., -1:],
            xyz2lonlat(depth2xyz(gt['depth'], plan_y=-gt['ratio']))[..., -1:]
            ], dim=-1).permute(0, 2, 1)
    }
    # dummy_dt['boundary'][:, :, :20] /= 1.2  # some different

    boundary_loss = BoundaryLoss()
    loss = boundary_loss(gt, dummy_dt)
    print(loss)
