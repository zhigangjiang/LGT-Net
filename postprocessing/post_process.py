""" 
@Date: 2021/10/08
@description:
"""
import numpy as np
import cv2

from postprocessing.dula.layout import fit_layout
from postprocessing.dula.layout_old import fit_layout_old
from utils.conversion import depth2xyz, xyz2depth


def post_process(b_depth, type_name='manhattan', need_cube=False):
    plan_y = 1
    b_xyz = depth2xyz(b_depth, plan_y)

    b_processed_xyz = []
    for xyz in b_xyz:
        if type_name == 'manhattan':
            processed_xz = fit_layout(floor_xz=xyz[..., ::2], need_cube=need_cube, show=False)
        elif type_name == 'manhattan_old':
            processed_xz = fit_layout_old(floor_xz=xyz[..., ::2], need_cube=need_cube, show=False)
        elif type_name == 'atalanta':
            processed_xz = cv2.approxPolyDP(xyz[..., ::2].astype(np.float32), 0.1, False)[:, 0, :]
        else:
            raise NotImplementedError("Unknown post-processing type")

        if need_cube:
            assert len(processed_xz) == 4

        processed_xyz = np.insert(processed_xz, 1, plan_y, axis=1)
        b_processed_xyz.append(processed_xyz)

    return np.array(b_processed_xyz)