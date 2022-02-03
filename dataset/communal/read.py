""" 
@Date: 2021/07/28
@description:
"""
import os
import numpy as np
import cv2
import json
from PIL import Image
from utils.conversion import xyz2uv, pixel2uv
from utils.height import calc_ceil_ratio


def read_image(image_path, shape=None):
    if shape is None:
        shape = [512, 1024]
    img = np.array(Image.open(image_path)).astype(np.float32) / 255
    if img.shape[0] != shape[0] or img.shape[1] != shape[1]:
        img = cv2.resize(img, dsize=tuple(shape[::-1]), interpolation=cv2.INTER_AREA)

    return np.array(img)


def read_label(label_path, data_type='MP3D'):

    if data_type == 'MP3D':
        with open(label_path, 'r') as f:
            label = json.load(f)
        point_idx = [one['pointsIdx'][0] for one in label['layoutWalls']['walls']]
        camera_height = label['cameraHeight']
        room_height = label['layoutHeight']
        camera_ceiling_height = room_height - camera_height
        ratio = camera_ceiling_height / camera_height

        xyz = [one['xyz'] for one in label['layoutPoints']['points']]
        assert len(xyz) == len(point_idx), "len(xyz) != len(point_idx)"
        xyz = [xyz[i] for i in point_idx]
        xyz = np.asarray(xyz, dtype=np.float32)
        xyz[:, 2] *= -1
        xyz[:, 1] = camera_height
        corners = xyz2uv(xyz)
    elif data_type == 'Pano_S2D3D':
        with open(label_path, 'r') as f:
            lines = [line for line in f.readlines() if
                     len([c for c in line.split(' ') if c[0].isnumeric()]) > 1]

        corners_list = np.array([line.strip().split() for line in lines], np.float32)
        uv_list = pixel2uv(corners_list)
        ceil_uv = uv_list[::2]
        floor_uv = uv_list[1::2]
        ratio = calc_ceil_ratio([ceil_uv, floor_uv], mode='mean')
        corners = floor_uv
    else:
        return None

    output = {
        'ratio': np.array([ratio], dtype=np.float32),
        'corners': corners,
        'id': os.path.basename(label_path).split('.')[0]
    }
    return output


def move_not_simple_image(data_dir, simple_panos):
    import shutil
    for house_index in os.listdir(data_dir):
        house_path = os.path.join(data_dir, house_index)
        if not os.path.isdir(house_path) or house_index == 'visualization':
            continue

        floor_plan_path = os.path.join(house_path, 'floor_plans')
        if os.path.exists(floor_plan_path):
            print(f'move:{floor_plan_path}')
            dst_floor_plan_path = floor_plan_path.replace('zind', 'zind2')
            os.makedirs(dst_floor_plan_path, exist_ok=True)
            shutil.move(floor_plan_path, dst_floor_plan_path)

        panos_path = os.path.join(house_path, 'panos')
        for pano in os.listdir(panos_path):
            pano_path = os.path.join(panos_path, pano)
            pano_index = '_'.join(pano.split('.')[0].split('_')[-2:])
            if f'{house_index}_{pano_index}' not in simple_panos and os.path.exists(pano_path):
                print(f'move:{pano_path}')
                dst_pano_path = pano_path.replace('zind', 'zind2')
                os.makedirs(os.path.dirname(dst_pano_path), exist_ok=True)
                shutil.move(pano_path, dst_pano_path)


def read_zind(partition_path, simplicity_path, data_dir, mode, is_simple=True,
              layout_type='layout_raw', is_ceiling_flat=False, plan_y=1):
    with open(simplicity_path, 'r') as f:
        simple_tag = json.load(f)
        simple_panos = {}
        for k in simple_tag.keys():
            if not simple_tag[k]:
                continue
            split = k.split('_')
            house_index = split[0]
            pano_index = '_'.join(split[-2:])
            simple_panos[f'{house_index}_{pano_index}'] = True

    # move_not_simple_image(data_dir, simple_panos)

    pano_list = []
    with open(partition_path, 'r') as f1:
        house_list = json.load(f1)[mode]

    for house_index in house_list:
        with open(os.path.join(data_dir, house_index, f"zind_data.json"), 'r') as f2:
            data = json.load(f2)

        panos = []
        merger = data['merger']
        for floor in merger.values():
            for complete_room in floor.values():
                for partial_room in complete_room.values():
                    for pano_index in partial_room:
                        pano = partial_room[pano_index]
                        pano['index'] = pano_index
                        panos.append(pano)

        for pano in panos:
            if layout_type not in pano:
                continue
            pano_index = pano['index']

            if is_simple and f'{house_index}_{pano_index}' not in simple_panos.keys():
                continue

            if is_ceiling_flat and not pano['is_ceiling_flat']:
                continue

            layout = pano[layout_type]
            # corners
            corner_xz = np.array(layout['vertices'])
            corner_xz[..., 0] = -corner_xz[..., 0]
            corner_xyz = np.insert(corner_xz, 1, pano['camera_height'], axis=1)
            corners = xyz2uv(corner_xyz).astype(np.float32)

            # ratio
            ratio = np.array([(pano['ceiling_height'] - pano['camera_height']) / pano['camera_height']], dtype=np.float32)

            # Ours future work: detection window, door, opening
            objects = {
                'windows': [],
                'doors': [],
                'openings': [],
            }
            for label_index, wdo_type in enumerate(["windows", "doors", "openings"]):
                if wdo_type not in layout:
                    continue

                wdo_vertices = np.array(layout[wdo_type])
                if len(wdo_vertices) == 0:
                    continue

                assert len(wdo_vertices) % 3 == 0

                for i in range(0, len(wdo_vertices), 3):
                    # In the Zind dataset, the camera height is 1, and the default camera height in our code is also 1,
                    # so the xyz coordinate here can be used directly
                    # Since we're taking the opposite z-axis, we're changing the order of left and right

                    left_bottom_xyz = np.array(
                        [-wdo_vertices[i + 1][0], -wdo_vertices[i + 2][0], wdo_vertices[i + 1][1]])
                    right_bottom_xyz = np.array(
                        [-wdo_vertices[i][0], -wdo_vertices[i + 2][0], wdo_vertices[i][1]])
                    center_bottom_xyz = (left_bottom_xyz + right_bottom_xyz) / 2

                    center_top_xyz = center_bottom_xyz.copy()
                    center_top_xyz[1] = -wdo_vertices[i + 2][1]

                    center_boundary_xyz = center_bottom_xyz.copy()
                    center_boundary_xyz[1] = plan_y

                    uv = xyz2uv(np.array([left_bottom_xyz, right_bottom_xyz,
                                          center_bottom_xyz, center_top_xyz,
                                          center_boundary_xyz]))

                    left_bottom_uv = uv[0]
                    right_bottom_uv = uv[1]
                    width_u = abs(right_bottom_uv[0] - left_bottom_uv[0])
                    width_u = 1 - width_u if width_u > 0.5 else width_u
                    assert width_u > 0, width_u

                    center_bottom_uv = uv[2]
                    center_top_uv = uv[3]
                    height_v = center_bottom_uv[1] - center_top_uv[1]

                    if height_v < 0:
                        continue

                    center_boundary_uv = uv[4]
                    boundary_v = center_boundary_uv[1] - center_bottom_uv[1] if wdo_type == 'windows' else 0
                    boundary_v = 0 if boundary_v < 0 else boundary_v

                    center_u = center_bottom_uv[0]

                    objects[wdo_type].append({
                        'width_u': width_u,
                        'height_v': height_v,
                        'boundary_v': boundary_v,
                        'center_u': center_u
                    })

            pano_list.append({
                'img_path': os.path.join(data_dir, house_index, pano['image_path']),
                'corners': corners,
                'objects': objects,
                'ratio': ratio,
                'id': f'{house_index}_{pano_index}',
                'is_inside': pano['is_inside']
            })
    return pano_list
