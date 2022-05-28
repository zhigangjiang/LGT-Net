"""
@author: Zhigang Jiang
@time: 2022/05/25
@description: reference: https://github.com/sunset1995/PanoPlane360/blob/main/vis_planes.py
"""
import open3d
import numpy as np
from utils.conversion import pixel2lonlat


def create_3d_obj(img, depth, save_path=None, mesh=True, mesh_show_back_face=False, show=False):
    assert img.shape[0] == depth.shape[0], ""
    h = img.shape[0]
    w = img.shape[1]
    # Project to 3d
    lon = pixel2lonlat(np.array(range(w)), w=w, axis=0)[None].repeat(h, axis=0)
    lat = pixel2lonlat(np.array(range(h)), h=h, axis=1)[..., None].repeat(w, axis=1)

    z = depth * np.sin(lat)
    x = depth * np.cos(lat) * np.cos(lon)
    y = depth * np.cos(lat) * np.sin(lon)
    pts_xyz = np.stack([x, -z, y], -1).reshape(-1, 3)
    pts_rgb = img.reshape(-1, 3)

    if mesh:
        pid = np.arange(len(pts_xyz)).reshape(h, w)
        faces = np.concatenate([
            np.stack([
                pid[:-1, :-1], pid[1:, :-1], np.roll(pid, -1, axis=1)[:-1, :-1],
            ], -1),
            np.stack([
                pid[1:, :-1], np.roll(pid, -1, axis=1)[1:, :-1], np.roll(pid, -1, axis=1)[:-1, :-1],
            ], -1)
        ]).reshape(-1, 3).tolist()
        scene = open3d.geometry.TriangleMesh()
        scene.vertices = open3d.utility.Vector3dVector(pts_xyz)
        scene.vertex_colors = open3d.utility.Vector3dVector(pts_rgb)
        scene.triangles = open3d.utility.Vector3iVector(faces)

    else:
        scene = open3d.geometry.PointCloud()
        scene.points = open3d.utility.Vector3dVector(pts_xyz)
        scene.colors = open3d.utility.Vector3dVector(pts_rgb)
    if save_path:
        open3d.io.write_triangle_mesh(save_path, scene, write_triangle_uvs=True)
    if show:
        open3d.visualization.draw_geometries([scene], mesh_show_back_face=mesh_show_back_face)


if __name__ == '__main__':
    from dataset.mp3d_dataset import MP3DDataset
    from utils.boundary import depth2boundaries, layout2depth
    from visualization.boundary import draw_boundaries

    mp3d_dataset = MP3DDataset(root_dir='../src/dataset/mp3d', mode='train', for_test_index=10, patch_num=1024)
    gt = mp3d_dataset.__getitem__(3)

    boundary_list = depth2boundaries(gt['ratio'], gt['depth'], step=None)
    pano_img = draw_boundaries(gt['image'].transpose(1, 2, 0), boundary_list=boundary_list, show=True)
    layout_depth = layout2depth(boundary_list, show=False)
    create_3d_obj(gt['image'].transpose(1, 2, 0), layout_depth, save_path=f"../src/output/{gt['id']}_3d.gltf",
                  mesh=True)
