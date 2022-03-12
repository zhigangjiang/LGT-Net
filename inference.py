""" 
@Date: 2021/09/19
@description:
"""
import json
import os
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import glob

from tqdm import tqdm
from PIL import Image
from config.defaults import merge_from_file, get_config
from dataset.mp3d_dataset import MP3DDataset
from dataset.zind_dataset import ZindDataset
from models.build import build_model
from loss import GradLoss
from postprocessing.post_process import post_process
from preprocessing.pano_lsd_align import panoEdgeDetection, rotatePanorama
from utils.boundary import corners2boundaries
from utils.conversion import depth2xyz
from utils.logger import get_logger
from utils.misc import tensor2np_d, tensor2np
from evaluation.accuracy import show_grad
from models.lgt_net import LGT_Net
from utils.writer import xyz2json
from visualization.boundary import draw_boundaries
from visualization.floorplan import draw_floorplan, draw_iou_floorplan


def parse_option():
    parser = argparse.ArgumentParser(description='Panorama Layout Transformer training and evaluation script')
    parser.add_argument('--img_glob',
                        type=str,
                        required=True,
                        help='image glob path')

    parser.add_argument('--cfg',
                        type=str,
                        required=True,
                        metavar='FILE',
                        help='path of config file')

    parser.add_argument('--post_processing',
                        type=str,
                        default='manhattan',
                        choices=['manhattan', 'atalanta', 'original'],
                        help='post-processing type')

    parser.add_argument('--output_dir',
                        type=str,
                        default='src/output',
                        help='path of output')

    parser.add_argument('--visualize_3d', action='store_true',
                        help='visualize_3d')

    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='device')

    args = parser.parse_args()
    args.mode = 'test'

    print("arguments:")
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    print("-" * 50)
    return args


def visualize_2d(img, dt, show_depth=True, show_floorplan=True, show=False, save_path=None):
    dt_np = tensor2np_d(dt)
    dt_depth = dt_np['depth'][0]
    dt_xyz = depth2xyz(np.abs(dt_depth))
    dt_ratio = dt_np['ratio'][0][0]
    dt_boundaries = corners2boundaries(dt_ratio, corners_xyz=dt_xyz, step=None, visible=False, length=img.shape[1])
    vis_img = draw_boundaries(img, boundary_list=dt_boundaries, boundary_color=[0, 1, 0])

    if 'processed_xyz' in dt:
        dt_boundaries = corners2boundaries(dt_ratio, corners_xyz=dt['processed_xyz'][0], step=None, visible=False,
                                           length=img.shape[1])
        vis_img = draw_boundaries(vis_img, boundary_list=dt_boundaries, boundary_color=[1, 0, 0])

    if show_depth:
        dt_grad_img = show_depth_normal_grad(dt)
        grad_h = dt_grad_img.shape[0]
        vis_merge = [
            vis_img[0:-grad_h, :, :],
            dt_grad_img,
        ]
        vis_img = np.concatenate(vis_merge, axis=0)
        # vis_img = dt_grad_img.transpose(1, 2, 0)[100:]

    if show_floorplan:
        if 'processed_xyz' in dt:
            floorplan = draw_iou_floorplan(dt['processed_xyz'][0][..., ::2], dt_xyz[..., ::2],
                                           dt_board_color=[1, 0, 0, 1], gt_board_color=[0, 1, 0, 1])
        else:
            floorplan = show_alpha_floorplan(dt_xyz)

        vis_img = np.concatenate([vis_img, floorplan[:, 60:-60, :]], axis=1)
    if show:
        plt.imshow(vis_img)
        plt.show()
    if save_path:
        result = Image.fromarray((vis_img * 255).astype(np.uint8))
        result.save(save_path)
    return vis_img


def preprocess(img_ori, q_error=0.7, refine_iter=3, vp_cache_path=None):
    # Align images with VP
    if os.path.exists(vp_cache_path):
        with open(vp_cache_path) as f:
            vp = [[float(v) for v in line.rstrip().split(' ')] for line in f.readlines()]
            vp = np.array(vp)
    else:
        # VP detection and line segment extraction
        _, vp, _, _, _, _, _ = panoEdgeDetection(img_ori,
                                                 qError=q_error,
                                                 refineIter=refine_iter)
    i_img = rotatePanorama(img_ori, vp[2::-1])

    if vp_cache_path is not None:
        with open(vp_cache_path, 'w') as f:
            for i in range(3):
                f.write('%.6f %.6f %.6f\n' % (vp[i, 0], vp[i, 1], vp[i, 2]))

    return i_img, vp


def show_depth_normal_grad(dt):
    grad_conv = GradLoss().to(dt['depth'].device).grad_conv
    dt_grad_img = show_grad(dt['depth'][0], grad_conv, 50)
    dt_grad_img = cv2.resize(dt_grad_img, (1024, 60), interpolation=cv2.INTER_NEAREST)
    return dt_grad_img


def show_alpha_floorplan(dt_xyz, side_l=512):
    fill_color = [0.2, 0.2, 0.2, 0.2]
    dt_floorplan = draw_floorplan(xz=dt_xyz[..., ::2], fill_color=fill_color,
                                  border_color=[1, 0, 0, 1], side_l=side_l, show=False, center_color=[1, 0, 0, 1])
    dt_floorplan = Image.fromarray((dt_floorplan * 255).astype(np.uint8), mode='RGBA')
    back = np.zeros([side_l, side_l, len(fill_color)], dtype=np.float)
    back[..., :] = [0.8, 0.8, 0.8, 1]
    back = Image.fromarray((back * 255).astype(np.uint8), mode='RGBA')
    iou_floorplan = Image.alpha_composite(back, dt_floorplan).convert("RGB")
    dt_floorplan = np.array(iou_floorplan) / 255.0
    return dt_floorplan


def save_pred_json(xyz, ration, save_path):
    # xyz[..., -1] = -xyz[..., -1]
    json_data = xyz2json(xyz, ration)
    with open(save_path, 'w') as f:
        f.write(json.dumps(json_data, indent=4) + '\n')
    return json_data


def inference():
    if len(img_paths) == 0:
        logger.error('No images found')
        return

    bar = tqdm(img_paths, ncols=100)
    for img_path in bar:
        if not os.path.isfile(img_path):
            logger.error(f'The {img_path} not is file')
            continue
        name = os.path.basename(img_path).split('.')[0]
        bar.set_description(name)
        img = np.array(Image.open(img_path).resize((1024, 512), Image.BICUBIC))[..., :3]
        if args.post_processing is not None and 'manhattan' in args.post_processing:
            bar.set_description("Preprocessing")
            img, vp = preprocess(img, vp_cache_path=os.path.join(args.output_dir, f"{name}_vp.txt"))

        img = (img / 255.0).astype(np.float32)
        run_one_inference(img, model, args, name)


def inference_dataset(dataset):
    bar = tqdm(dataset, ncols=100)
    for data in bar:
        bar.set_description(data['id'])
        run_one_inference(data['image'].transpose(1, 2, 0), model, args, name=data['id'])


@torch.no_grad()
def run_one_inference(img, model, args, name):
    model.eval()
    dt = model(torch.from_numpy(img.transpose(2, 0, 1)[None]).to(args.device))
    if args.post_processing != 'original':
        dt['processed_xyz'] = post_process(tensor2np(dt['depth']), type_name=args.post_processing)

    visualize_2d(img, dt, show=True, save_path=os.path.join(args.output_dir, f"{name}_pred.png"))
    output_xyz = dt['processed_xyz'][0] if 'processed_xyz' in dt else depth2xyz(tensor2np(dt['depth'][0]))
    json_data = save_pred_json(output_xyz, tensor2np(dt['ratio'][0])[0],
                               save_path=os.path.join(args.output_dir, f"{name}_pred.json"))
    if args.visualize_3d:
        from visualization.visualizer.visualizer import visualize_3d
        visualize_3d(json_data, (img * 255).astype(np.uint8))


if __name__ == '__main__':
    logger = get_logger()
    args = parse_option()
    config = get_config(args)

    if 'cuda' in args.device and not torch.cuda.is_available():
        logger.info(f'The {args.device} is not available, will use cpu ...')
        config.defrost()
        args.device = "cpu"
        config.TRAIN.DEVICE = "cpu"
        config.freeze()

    model, _, _, _ = build_model(config, logger)
    os.makedirs(args.output_dir, exist_ok=True)
    img_paths = sorted(glob.glob(args.img_glob))

    inference()

    # dataset = MP3DDataset(root_dir='./src/dataset/mp3d', mode='test', split_list=[
    #     ['7y3sRwLe3Va', '155fac2d50764bf09feb6c8f33e8fb76'],
    #     ['e9zR4mvMWw7', 'c904c55a5d0e420bbd6e4e030b9fe5b4'],
    # ])
    # dataset = ZindDataset(root_dir='./src/dataset/zind', mode='test', split_list=[
    #     '1169_pano_21',
    #     '0583_pano_59',
    # ], vp_align=True)
    # inference_dataset(dataset)
