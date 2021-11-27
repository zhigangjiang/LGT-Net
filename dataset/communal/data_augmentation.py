""" 
@Date: 2021/07/27
@description:
"""
import numpy as np
import cv2
import functools

from utils.conversion import pixel2lonlat, lonlat2pixel, uv2lonlat, lonlat2uv, pixel2uv


@functools.lru_cache()
def prepare_stretch(w, h):
    lon = pixel2lonlat(np.array(range(w)), w=w, axis=0)
    lat = pixel2lonlat(np.array(range(h)), h=h, axis=1)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    tan_lat = np.tan(lat)
    return sin_lon, cos_lon, tan_lat


def pano_stretch_image(pano_img, kx, ky, kz):
    """
    Note that this is the inverse mapping, which refers to Equation 3 in HorizonNet paper (the coordinate system in
    the paper is different from here, xz needs to be swapped)
    :param pano_img: a panorama image, shape must be [h,w,c]
    :param kx: stretching along left-right direction
    :param ky: stretching along up-down direction
    :param kz: stretching along front-back direction
    :return:
    """
    w = pano_img.shape[1]
    h = pano_img.shape[0]

    sin_lon, cos_lon, tan_lat = prepare_stretch(w, h)

    n_lon = np.arctan2(sin_lon * kz / kx, cos_lon)
    n_lat = np.arctan(tan_lat[..., None] * np.sin(n_lon) / sin_lon * kx / ky)
    n_pu = lonlat2pixel(n_lon, w=w, axis=0, need_round=False)
    n_pv = lonlat2pixel(n_lat, h=h, axis=1, need_round=False)

    pixel_map = np.empty((h, w, 2), dtype=np.float32)
    pixel_map[..., 0] = n_pu
    pixel_map[..., 1] = n_pv
    map1 = pixel_map[..., 0]
    map2 = pixel_map[..., 1]
    # using wrap mode because it is continues at left or right of panorama
    new_img = cv2.remap(pano_img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return new_img


def pano_stretch_conner(corners, kx, ky, kz):
    """
    :param corners:
    :param kx: stretching along left-right direction
    :param ky: stretching along up-down direction
    :param kz: stretching along front-back direction
    :return:
    """

    lonlat = uv2lonlat(corners)
    sin_lon = np.sin(lonlat[..., 0:1])
    cos_lon = np.cos(lonlat[..., 0:1])
    tan_lat = np.tan(lonlat[..., 1:2])

    n_lon = np.arctan2(sin_lon * kx / kz, cos_lon)

    a = np.bitwise_or(corners[..., 0] == 0.5, corners[..., 0] == 1)
    b = np.bitwise_not(a)
    w = np.zeros_like(n_lon)
    w[b] = np.sin(n_lon[b]) / sin_lon[b]
    w[a] = kx / kz

    n_lat = np.arctan(tan_lat * w / kx * ky)

    lst = [n_lon, n_lat]
    lonlat = np.concatenate(lst, axis=-1)
    new_corners = lonlat2uv(lonlat)
    return new_corners


def pano_stretch(pano_img, corners, kx, ky, kz):
    """
    :param pano_img: a panorama image, shape must be [h,w,c]
    :param corners:
    :param kx: stretching along left-right direction
    :param ky: stretching along up-down direction
    :param kz: stretching along front-back direction
    :return:
    """
    new_img = pano_stretch_image(pano_img, kx, ky, kz)
    new_corners = pano_stretch_conner(corners, kx, ky, kz)
    return new_img, new_corners


class PanoDataAugmentation:
    def __init__(self, aug):
        self.aug = aug
        self.parameters = {}

    def need_aug(self, name):
        return name in self.aug and self.aug[name]

    def execute_space_aug(self, corners, image):
        if image is None:
            return image

        if self.aug is None:
            return corners, image
        w = image.shape[1]
        h = image.shape[0]

        if self.need_aug('STRETCH'):
            kx = np.random.uniform(1, 2)
            kx = 1 / kx if np.random.randint(2) == 0 else kx
            # we found that the ky transform may cause IoU to drop (HorizonNet also only x and z transform)
            # ky = np.random.uniform(1, 2)
            # ky = 1 / ky if np.random.randint(2) == 0 else ky
            ky = 1
            kz = np.random.uniform(1, 2)
            kz = 1 / kz if np.random.randint(2) == 0 else kz
            image, corners = pano_stretch(image, corners, kx, ky, kz)
            self.parameters['STRETCH'] = {'kx': kx, 'ky': ky, 'kz': kz}
        else:
            self.parameters['STRETCH'] = None

        if self.need_aug('ROTATE'):
            d_pu = np.random.randint(w)
            image = np.roll(image, d_pu, axis=1)
            corners[..., 0] = (corners[..., 0] + pixel2uv(np.array([d_pu]), w, h)) % pixel2uv(np.array([w]), w, h)
            self.parameters['ROTATE'] = d_pu
        else:
            self.parameters['ROTATE'] = None

        if self.need_aug('FLIP') and np.random.randint(2) == 0:
            image = np.flip(image, axis=1).copy()
            corners[..., 0] = pixel2uv(np.array([w]), w, h) - corners[..., 0]
            corners = corners[::-1]
            self.parameters['FLIP'] = True
        else:
            self.parameters['FLIP'] = None

        return corners, image

    def execute_visual_aug(self, image):
        if self.need_aug('GAMMA'):
            p = np.random.uniform(1, 2)
            if np.random.randint(2) == 0:
                p = 1 / p
            image = image ** p
            self.parameters['GAMMA'] = p
        else:
            self.parameters['GAMMA'] = None

        # The following visual augmentation methods are only implemented but not tested
        if self.need_aug('HUE') or self.need_aug('SATURATION'):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

            if self.need_aug('HUE') and np.random.randint(2) == 0:
                p = np.random.uniform(-0.1, 0.1)
                image[..., 0] = np.mod(image[..., 0] + p * 180, 180)
                self.parameters['HUE'] = p
            else:
                self.parameters['HUE'] = None

            if self.need_aug('SATURATION') and np.random.randint(2) == 0:
                p = np.random.uniform(0.5, 1.5)
                image[..., 1] = np.clip(image[..., 1] * p, 0, 1)
                self.parameters['SATURATION'] = p
            else:
                self.parameters['SATURATION'] = None

            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        if self.need_aug('CONTRAST') and np.random.randint(2) == 0:
            p = np.random.uniform(0.9, 1.1)
            mean = image.mean(axis=0).mean(axis=0)
            image = (image - mean) * p + mean
            image = np.clip(image, 0, 1)
            self.parameters['CONTRAST'] = p
        else:
            self.parameters['CONTRAST'] = None

        return image

    def execute_aug(self, corners, image):
        corners, image = self.execute_space_aug(corners, image)
        if image is not None:
            image = self.execute_visual_aug(image)
        return corners, image


if __name__ == '__main__1':
    from tqdm import trange
    from visualization.floorplan import draw_floorplan
    from dataset.communal.read import read_image, read_label
    from utils.time_watch import TimeWatch
    from utils.conversion import uv2xyz
    from utils.boundary import corners2boundary

    np.random.seed(123)
    pano_img_path = "../../src/dataset/mp3d/image/TbHJrupSAjP_f320ae084f3a447da3e8ab11dd5f9320.png"
    label_path = "../../src/dataset/mp3d/label/TbHJrupSAjP_f320ae084f3a447da3e8ab11dd5f9320.json"
    pano_img = read_image(pano_img_path)
    label = read_label(label_path)

    corners = label['corners']
    ratio = label['ratio']

    pano_aug = PanoDataAugmentation(aug={
        'STRETCH': True,
        'ROTATE': True,
        'FLIP': True,
        'GAMMA': True,
        # 'HUE': True,
        # 'SATURATION': True,
        # 'CONTRAST': True
    })

    # draw_floorplan(corners, show=True, marker_color=0.5, center_color=0.8, plan_y=1.6, show_radius=8)
    # draw_boundaries(pano_img, corners_list=[corners], show=True, length=1024, ratio=ratio)

    w = TimeWatch("test")
    for i in trange(50000):
        new_corners, new_pano_img = pano_aug.execute_aug(corners.copy(), pano_img.copy())
        # draw_floorplan(uv2xyz(new_corners, plan_y=1.6)[..., ::2], show=True, marker_color=0.5, center_color=0.8,
        #                show_radius=8)
        # draw_boundaries(new_pano_img, corners_list=[new_corners], show=True, length=1024, ratio=ratio)


if __name__ == '__main__':
    from utils.boundary import corners2boundary
    from visualization.floorplan import draw_floorplan
    from utils.boundary import visibility_corners

    corners = np.array([[0.7664539, 0.7416811],
                        [0.06641078, 0.6521386],
                        [0.30997428, 0.57855356],
                        [0.383300784, 0.58726823],
                        [0.383300775, 0.8005296],
                        [0.5062902, 0.74822706]])
    corners = visibility_corners(corners)
    print(corners)
    # draw_floorplan(uv2xyz(corners, plan_y=1.6)[..., ::2], show=True, marker_color=0.5, center_color=0.8,
    #                show_radius=8)
    visible_floor_boundary = corners2boundary(corners, length=256, visible=True)
    # visible_depth = xyz2depth(uv2xyz(visible_floor_boundary, 1), 1)
    print(len(visible_floor_boundary))


if __name__ == '__main__0':
    from visualization.floorplan import draw_floorplan

    from dataset.communal.read import read_image, read_label
    from utils.time_watch import TimeWatch
    from utils.conversion import uv2xyz

    # np.random.seed(1234)
    pano_img_path = "../../src/dataset/mp3d/image/VVfe2KiqLaN_35b41dcbfcf84f96878f6ca28c70e5af.png"
    label_path = "../../src/dataset/mp3d/label/VVfe2KiqLaN_35b41dcbfcf84f96878f6ca28c70e5af.json"
    pano_img = read_image(pano_img_path)
    label = read_label(label_path)

    corners = label['corners']
    ratio = label['ratio']

    # draw_floorplan(corners, show=True, marker_color=0.5, center_color=0.8, plan_y=1.6, show_radius=8)

    w = TimeWatch()
    for i in range(5):
        kx = np.random.uniform(1, 2)
        kx = 1 / kx if np.random.randint(2) == 0 else kx
        ky = np.random.uniform(1, 2)
        ky = 1 / ky if np.random.randint(2) == 0 else ky
        kz = np.random.uniform(1, 2)
        kz = 1 / kz if np.random.randint(2) == 0 else kz
        new_corners = pano_stretch_conner(corners.copy(), kx, ky, kz)
        draw_floorplan(uv2xyz(new_corners, plan_y=1.6)[..., ::2], show=True, marker_color=0.5, center_color=0.8,
                       show_radius=8)
