""" 
@Date: 2021/08/12
@description:
"""
import torch
import torch.nn as nn
from loss.grad_loss import GradLoss


class ObjectLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.heat_map_loss = HeatmapLoss(reduction='mean')  # FocalLoss(reduction='mean')
        self.l1_loss = nn.SmoothL1Loss()

    def forward(self, gt, dt):
        # TODO::
        return 0


class HeatmapLoss(nn.Module):
    def __init__(self, weight=None, alpha=2, beta=4, reduction='mean'):
        super(HeatmapLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, targets, inputs):
        center_id = (targets == 1.0).float()
        other_id = (targets != 1.0).float()
        center_loss = -center_id * (1.0 - inputs) ** self.alpha * torch.log(inputs + 1e-14)
        other_loss = -other_id * (1 - targets) ** self.beta * inputs ** self.alpha * torch.log(1.0 - inputs + 1e-14)
        loss = center_loss + other_loss

        batch_size = loss.size(0)
        if self.reduction == 'mean':
            loss = torch.sum(loss) / batch_size

        if self.reduction == 'sum':
            loss = torch.sum(loss) / batch_size

        return loss
