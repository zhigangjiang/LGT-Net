""" 
@Date: 2021/11/06
@description:
"""
import cv2
import numpy as np


def xyz2json(xyz, ratio, camera_height=1.6):
    xyz = xyz * camera_height
    ceiling_height = camera_height * ratio
    layout_height = camera_height + ceiling_height
    data = {
        'cameraHeight': camera_height,
        'layoutHeight': layout_height,
        'cameraCeilingHeight': ceiling_height,
        'layoutObj2ds': {
            'num': 0,
            'obj2ds': []
        },
        'layoutPoints': {
            'num': xyz.shape[0],
            'points': []
        },
        'layoutWalls': {
            'num': xyz.shape[0],
            'walls': []
        }
    }

    xyz = np.concatenate([xyz, xyz[0:1, :]], axis=0)
    R_180 = cv2.Rodrigues(np.array([0, -1 * np.pi, 0], np.float32))[0]
    for i in range(xyz.shape[0] - 1):
        a = np.dot(R_180, xyz[i, :])
        a[0] *= -1
        b = np.dot(R_180, xyz[i + 1, :])
        b[0] *= -1
        c = a.copy()
        c[1] = 0
        normal = np.cross(a - b, a - c)
        normal /= np.linalg.norm(normal)
        d = -np.sum(normal * a)
        plane = np.asarray([normal[0], normal[1], normal[2], d])

        data['layoutPoints']['points'].append({'xyz': a.tolist(), 'id': i})

        next_i = 0 if i + 1 >= (xyz.shape[0] - 1) else i + 1
        tmp = {
            'normal': normal.tolist(),
            'planeEquation': plane.tolist(),
            'pointsIdx': [i, next_i]
        }
        data['layoutWalls']['walls'].append(tmp)

    return data

