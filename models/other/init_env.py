""" 
@Date: 2021/08/15
@description:
"""
import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import cv2


def init_env(seed, deterministic=False, loader_work_num=0):
    # Fix seed
    # Python & NumPy
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    #  PyTorch
    torch.manual_seed(seed)  # 为CPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

    # cuDNN
    if deterministic:
        # 复现
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True  # 将这个 flag 置为 True 的话，每次返回的卷积算法将是确定的，即默认算法
    else:
        cudnn.benchmark = True  # 如果网络的输入数据维度或类型上变化不大，设置true
        torch.backends.cudnn.deterministic = False

    # Using multiple threads in Opencv can cause deadlocks
    if loader_work_num != 0:
        cv2.setNumThreads(0)
