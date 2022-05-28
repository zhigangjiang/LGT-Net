""" 
@author: Zhigang Jiang
@time: 2022/01/28
@description:
Holistic 3D Vision Challenge on General Room Layout Estimation Track Evaluation Package
Reference: https://github.com/bertjiazheng/indoor-layout-evaluation
"""

from scipy.optimize import linear_sum_assignment
import numpy as np
import scipy

HEIGHT, WIDTH = 512, 1024
MAX_DISTANCE = np.sqrt(HEIGHT**2 + WIDTH**2)


def f1_score_2d(gt_corners, dt_corners, thresholds):
    distances = scipy.spatial.distance.cdist(gt_corners, dt_corners)
    return eval_junctions(distances, thresholds=thresholds)


def eval_junctions(distances, thresholds=5):
    thresholds = thresholds if isinstance(thresholds, tuple) or isinstance(
        thresholds, list) else list([thresholds])

    num_gts, num_preds = distances.shape

    # filter the matches between ceiling-wall and floor-wall junctions
    mask = np.zeros_like(distances, dtype=np.bool)
    mask[:num_gts//2, :num_preds//2] = True
    mask[num_gts//2:, num_preds//2:] = True
    distances[~mask] = np.inf

    # F-measure under different thresholds
    Fs = []
    Ps = []
    Rs = []
    for threshold in thresholds:
        distances_temp = distances.copy()

        # filter the mis-matched pairs
        distances_temp[distances_temp > threshold] = np.inf

        # remain the rows and columns that contain non-inf elements
        distances_temp = distances_temp[:, np.any(np.isfinite(distances_temp), axis=0)]

        if np.prod(distances_temp.shape) == 0:
            Fs.append(0)
            Ps.append(0)
            Rs.append(0)
            continue

        distances_temp = distances_temp[np.any(np.isfinite(distances_temp), axis=1), :]

        # solve the bipartite graph matching problem
        row_ind, col_ind = linear_sum_assignment_with_inf(distances_temp)
        true_positive = np.sum(np.isfinite(distances_temp[row_ind, col_ind]))

        # compute precision and recall
        precision = true_positive / num_preds
        recall = true_positive / num_gts

        # compute F measure
        Fs.append(2 * precision * recall / (precision + recall))
        Ps.append(precision)
        Rs.append(recall)

    return Fs, Ps, Rs


def linear_sum_assignment_with_inf(cost_matrix):
    """
    Deal with linear_sum_assignment with inf according to
    https://github.com/scipy/scipy/issues/6900#issuecomment-451735634
    """
    cost_matrix = np.copy(cost_matrix)
    cost_matrix[np.isinf(cost_matrix)] = MAX_DISTANCE
    return linear_sum_assignment(cost_matrix)