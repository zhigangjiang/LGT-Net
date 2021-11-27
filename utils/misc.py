"""
@date: 2021/8/4
@description:
"""
import numpy as np
import torch


def tensor2np(t: torch.Tensor) -> np.array:
    if isinstance(t, torch.Tensor):
        if t.device == 'cpu':
            return t.detach().numpy()
        else:
            return t.detach().cpu().numpy()
    else:
        return t


def tensor2np_d(d: dict) -> dict:
    output = {}
    for k in d.keys():
        output[k] = tensor2np(d[k])
    return output
