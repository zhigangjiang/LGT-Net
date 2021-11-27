import cv2
import numpy as np
from visualization.visualizer.earcut import earcut

def xz2lines(wall_xz, h):
    cch = h - 1.6
    ch = 1.6
    strips_ceiling = []
    strips_wall = []
    strips_floor = []
    for i in range(wall_xz.shape[0] // 2):
        pts1 = wall_xz[i*2, :]
        pts2 = wall_xz[i*2+1, :]

        a = [[pts1[0], -cch, pts1[1]]]
        b = [[pts2[0], -cch, pts2[1]]]
        c = [[pts2[0],   ch, pts2[1]]]
        d = [[pts1[0],   ch, pts1[1]]]
        #strip = np.concatenate([a, b, b, c, c, d, d, a], axis=0)
        ceiling = np.concatenate([a, b], axis=0)
        wall = np.concatenate([b, c, d, a], axis=0)
        floor = np.concatenate([c, d], axis=0)
        
        strips_ceiling.append(ceiling)
        strips_wall.append(wall)
        strips_floor.append(floor)

    strips_ceiling = np.concatenate(strips_ceiling, axis=0).astype(np.float32)
    strips_wall = np.concatenate(strips_wall, axis=0).astype(np.float32)
    strips_floor = np.concatenate(strips_floor, axis=0).astype(np.float32)

    return strips_ceiling, strips_wall, strips_floor


def Label2Mesh(label, reverse=False):
    scale = 1.6 / label['cameraHeight']
    layout_height = scale * label['layoutHeight']
    if 'cameraCeilingHeight' not in label:
        label['cameraCeilingHeight'] = label['layoutHeight'] - label['cameraHeight']
    up_down_ratio = label['cameraCeilingHeight'] / label['cameraHeight']
    xyz = np.asarray(label['points'], np.float32)
    xyz *= scale
    point_idxs = label['order']
    
    wall_xz = [np.concatenate((xyz[:, ::2][i[0], :][None, ...], xyz[:, ::2][i[1], :][None, ...]), axis=0) for i in point_idxs]
    wall_xz = np.concatenate(wall_xz, axis=0).astype(np.float32)
    wall_num = wall_xz.shape[0] // 2
    lines_ceiling, lines_wall, lines_floor = xz2lines(wall_xz, layout_height)

    def ProcessOneWall(coord, idx, h):
        cch = h - 1.6
        ch = 1.6

        A = coord[idx[0], :]
        B = coord[idx[1], :]

        a = A.copy()
        b = B.copy()
        c = B.copy()
        a[1] = -cch
        b[1] = -cch
        c[1] = ch
        tmp1 = np.concatenate([a[None, ...], c[None, ...], b[None, ...]], axis=0)

        a = A.copy()
        b = A.copy()
        c = B.copy()
        a[1] = -cch
        b[1] = ch
        c[1] = ch
        tmp2 = np.concatenate([a[None, ...], b[None, ...], c[None, ...]], axis=0)
        
        return np.concatenate([tmp1[None, ...], tmp2[None, ...]], axis=0)
    mesh = [ProcessOneWall(xyz, point_idxs[x], layout_height)[None, ...] for x in range(len(point_idxs))]
    mesh = np.concatenate(mesh, axis=0).reshape([-1, 3])
    top_xz = []
    for i, j in point_idxs:
        if not reverse:
            tmp = np.concatenate([xyz[i, ::2], xyz[j, ::2]])
        else:
            tmp = np.concatenate([xyz[j, ::2], xyz[i, ::2]])
        top_xz += tmp.tolist()
    try:
        indices = np.asarray(earcut(top_xz)).reshape([-1, 3])
        top_xz = np.asarray(top_xz).reshape([-1, 2])
        tmp = []
        for i in range(indices.shape[0]):
            a = indices[i, 0]
            b = indices[i, 1]
            c = indices[i, 2]
            tmp.append(np.concatenate([top_xz[a:a+1, :], top_xz[b:b+1, :], top_xz[c:c+1, :]], axis=0))
        tmp = np.concatenate(tmp, axis=0)
        ceiling_mesh = np.zeros([tmp.shape[0], 3], np.float32)
        ceiling_mesh[:, ::2] = tmp.copy()
        ceiling_mesh[:, 1] = -(layout_height - 1.6)
        floor_mesh = ceiling_mesh.copy()
        floor_mesh[:, 1] = 1.6
        #mesh = np.concatenate([mesh, ceiling_mesh, floor_mesh], axis=0)
        mesh = np.concatenate([mesh, floor_mesh], axis=0)
    except:
        pass
    '''
    print (top_xz)
    top_xz = top_xz[:6]
    a = np.zeros([256, 256], np.uint8)
    b = ((top_xz - top_xz.min()) * 20).astype(int) + 5
    for i in range(0, b.shape[0]-1, 2):
        cv2.line(a, (b[i, 0], b[i, 1]), ((b[i+1, 0], b[i+1, 1])), color=255)
    import matplotlib.pyplot as plt
    plt.imshow(a)
    plt.show()
    exit()
    '''
    return wall_num, wall_xz, [lines_ceiling, lines_wall, lines_floor], mesh

def Label2Points(label):
    scale = 1.6 / label['cameraHeight']
    layout_height = scale * label['layoutHeight']
    up_down_ratio = label['cameraCeilingHeight'] / label['cameraHeight']
    xyz = np.asarray(label['points'], np.float32)
    point_idxs = label['order']
    def ProcessOneWall(coord, idx, h):
        cch = h - 1.6
        ch = 1.6

        a = coord[idx[0], ...].copy()
        b = coord[idx[1], ...].copy()
        a[1] = -cch
        b[1] = ch
        return np.concatenate([a[None, ...], b[None, ...]], axis=0)
    pts = [ProcessOneWall(xyz, point_idxs[x], layout_height)[None, ...] for x in range(len(point_idxs))]
    pts = np.concatenate(pts, axis=0)
    return pts


def OldFormat2Mine(label):
    scale = 1.6 / label['cameraHeight']
    layout_height = scale * label['layoutHeight']
    if 'cameraCeilingHeight' not in label:
        label['cameraCeilingHeight'] = label['layoutHeight'] - label['cameraHeight']

    up_down_ratio = label['cameraCeilingHeight'] / label['cameraHeight']
    xyz = []
    planes = []
    point_idxs = []
    R_180 = cv2.Rodrigues(np.array([0, -1*np.pi, 0], np.float32))[0]
    for one in label['layoutPoints']['points']: 
        xyz.append(one['xyz'])
    for one in label['layoutWalls']['walls']: 
        planes.append(one['planeEquation'])
        point_idxs.append(one['pointsIdx'])
    xyz = np.asarray(xyz)
    xyz[:, 0] *= -1
    xyz = xyz.dot(R_180.T)
    xyz[:, 1] = 0
    xyz *= scale
    
    data = {
            'cameraHeight': scale*label['cameraHeight'],
            'cameraCeilingHeight': scale*label['cameraCeilingHeight'],
            'layoutHeight': scale*label['layoutHeight'],
            'points': xyz.tolist(),
            'order': point_idxs
        }
    return data

def Label2Mesh_oldformat(label):
    scale = 1.6 / label['cameraHeight']
    layout_height = scale * label['layoutHeight']
    up_down_ratio = label['cameraCeilingHeight'] / label['cameraHeight']
    xyz = []
    planes = []
    point_idxs = []
    R_180 = cv2.Rodrigues(np.array([0, -1*np.pi, 0], np.float32))[0]
    for one in label['layoutPoints']['points']: 
        xyz.append(one['xyz'])
    for one in label['layoutWalls']['walls']: 
        planes.append(one['planeEquation'])
        point_idxs.append(one['pointsIdx'])
    xyz = np.asarray(xyz)
    xyz[:, 0] *= -1
    xyz = xyz.dot(R_180.T)
    def ProcessOneWall(coord, idx, h):
        cch = h - 1.6
        ch = 1.6

        A = coord[idx[0], :]
        B = coord[idx[1], :]

        a = A.copy()
        b = B.copy()
        c = B.copy()
        a[1] = -cch
        b[1] = -cch
        c[1] = ch
        tmp1 = np.concatenate([a[None, ...], b[None, ...], c[None, ...]], axis=0)

        a = A.copy()
        b = A.copy()
        c = B.copy()
        a[1] = -cch
        b[1] = ch
        c[1] = ch
        tmp2 = np.concatenate([a[None, ...], b[None, ...], c[None, ...]], axis=0)
        
        return np.concatenate([tmp1[None, ...], tmp2[None, ...]], axis=0)
    mesh = [ProcessOneWall(xyz, point_idxs[x], layout_height)[None, ...] for x in range(label['layoutPoints']['num'])]
    mesh = np.concatenate(mesh, axis=0).reshape([-1, 3])
    top_xz = []
    for i, j in point_idxs:
        tmp = np.concatenate([xyz[i, ::2], xyz[j, ::2]])
        top_xz += tmp.tolist()
    indices = np.asarray(earcut(top_xz)).reshape([-1, 3])
    top_xz = np.asarray(top_xz).reshape([-1, 2])
    tmp = []
    for i in range(indices.shape[0]):
        a = indices[i, 0]
        b = indices[i, 1]
        c = indices[i, 2]
        tmp.append(np.concatenate([top_xz[a:a+1, :], top_xz[b:b+1, :], top_xz[c:c+1, :]], axis=0))
    tmp = np.concatenate(tmp, axis=0)
    ceiling_mesh = np.zeros([tmp.shape[0], 3], np.float32)
    ceiling_mesh[:, ::2] = tmp.copy()
    ceiling_mesh[:, 1] = -(layout_height - 1.6)
    floor_mesh = ceiling_mesh.copy()
    floor_mesh[:, 1] = 1.6
    #mesh = np.concatenate([mesh, ceiling_mesh, floor_mesh], axis=0)
    mesh = np.concatenate([mesh, floor_mesh], axis=0)
    return mesh
