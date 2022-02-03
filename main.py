"""
@Date: 2021/07/17
@description:
"""
import sys
import os
import shutil
import argparse
import numpy as np
import json
import torch
import torch.nn.parallel
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.cuda

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from config.defaults import get_config, get_rank_config
from models.other.criterion import calc_criterion
from models.build import build_model
from models.other.init_env import init_env
from utils.logger import build_logger
from utils.misc import tensor2np_d, tensor2np
from dataset.build import build_loader
from evaluation.accuracy import calc_accuracy, show_heat_map, calc_ce, calc_pe, calc_rmse_delta_1, \
    show_depth_normal_grad, calc_f1_score
from postprocessing.post_process import post_process

try:
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    debug = True if sys.gettrace() else False
    parser = argparse.ArgumentParser(description='Panorama Layout Transformer training and evaluation script')
    parser.add_argument('--cfg',
                        type=str,
                        metavar='FILE',
                        help='path to config file')

    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=['train', 'val', 'test'],
                        help='train/val/test mode')

    parser.add_argument('--val_name',
                        type=str,
                        choices=['val', 'test'],
                        help='val name')

    parser.add_argument('--bs', type=int,
                        help='batch size')

    parser.add_argument('--save_eval', action='store_true',
                        help='save eval result')

    parser.add_argument('--post_processing', type=str,
                        choices=['manhattan', 'atalanta'],
                        help='type of postprocessing ')

    parser.add_argument('--need_cpe', action='store_true',
                        help='need to evaluate corner error and pixel error')

    parser.add_argument('--need_f1', action='store_true',
                        help='need to evaluate f1-score of corners')

    parser.add_argument('--need_rmse', action='store_true',
                        help='need to evaluate root mean squared error and delta error')

    parser.add_argument('--force_cube', action='store_true',
                        help='force cube shape when eval')

    parser.add_argument('--wall_num', type=int,
                        help='wall number')

    args = parser.parse_args()
    args.debug = debug
    print("arguments:")
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    print("-" * 50)
    return args


def main():
    args = parse_option()
    config = get_config(args)

    if config.TRAIN.SCRATCH and os.path.exists(config.CKPT.DIR) and config.MODE == 'train':
        print(f"Train from scratch, delete checkpoint dir: {config.CKPT.DIR}")
        f = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(config.CKPT.DIR) if 'pkl' in f]
        if len(f) > 0:
            last_epoch = np.array(f).max()
            if last_epoch > 10:
                c = input(f"delete it (last_epoch: {last_epoch})?(Y/N)\n")
                if c != 'y' and c != 'Y':
                    exit(0)

        shutil.rmtree(config.CKPT.DIR, ignore_errors=True)

    os.makedirs(config.CKPT.DIR, exist_ok=True)
    os.makedirs(config.CKPT.RESULT_DIR, exist_ok=True)
    os.makedirs(config.LOGGER.DIR, exist_ok=True)

    if ':' in config.TRAIN.DEVICE:
        nprocs = len(config.TRAIN.DEVICE.split(':')[-1].split(','))
    if 'cuda' in config.TRAIN.DEVICE:
        if not torch.cuda.is_available():
            print(f"Cuda is not available(config is: {config.TRAIN.DEVICE}), will use cpu ...")
            config.defrost()
            config.TRAIN.DEVICE = "cpu"
            config.freeze()
            nprocs = 1

    if config.MODE == 'train':
        with open(os.path.join(config.CKPT.DIR, "config.yaml"), "w") as f:
            f.write(config.dump(allow_unicode=True))

    if config.TRAIN.DEVICE == 'cpu' or nprocs < 2:
        print(f"Use single process, device:{config.TRAIN.DEVICE}")
        main_worker(0, config, 1)
    else:
        print(f"Use {nprocs} processes ...")
        mp.spawn(main_worker, nprocs=nprocs, args=(config, nprocs), join=True)


def main_worker(local_rank, cfg, world_size):
    config = get_rank_config(cfg, local_rank, world_size)
    logger = build_logger(config)
    writer = SummaryWriter(config.CKPT.DIR)
    logger.info(f"Comment: {config.COMMENT}")
    cur_pid = os.getpid()
    logger.info(f"Current process id: {cur_pid}")
    torch.hub._hub_dir = config.CKPT.PYTORCH
    logger.info(f"Pytorch hub dir: {torch.hub._hub_dir}")
    init_env(config.SEED, config.TRAIN.DETERMINISTIC, config.DATA.NUM_WORKERS)

    model, optimizer, criterion, scheduler = build_model(config, logger)
    train_data_loader, val_data_loader = build_loader(config, logger)

    if 'cuda' in config.TRAIN.DEVICE:
        torch.cuda.set_device(config.TRAIN.DEVICE)

    if config.MODE == 'train':
        train(model, train_data_loader, val_data_loader, optimizer, criterion, config, logger, writer, scheduler)
    else:
        iou_results, other_results = val_an_epoch(model, val_data_loader,
                                                  criterion, config, logger, writer=None,
                                                  epoch=config.TRAIN.START_EPOCH)
        results = dict(iou_results, **other_results)
        if config.SAVE_EVAL:
            save_path = os.path.join(config.CKPT.RESULT_DIR, f"result.json")
            with open(save_path, 'w+') as f:
                json.dump(results, f, indent=4)


def save(model, optimizer, epoch, iou_d, logger, writer, config):
    model.save(optimizer, epoch, accuracy=iou_d['full_3d'], logger=logger, acc_d=iou_d, config=config)
    for k in model.acc_d:
        writer.add_scalar(f"BestACC/{k}", model.acc_d[k]['acc'], epoch)


def train(model, train_data_loader, val_data_loader, optimizer, criterion, config, logger, writer, scheduler):
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        logger.info("=" * 200)
        train_an_epoch(model, train_data_loader, optimizer, criterion, config, logger, writer, epoch)
        epoch_iou_d, _ = val_an_epoch(model, val_data_loader, criterion, config, logger, writer, epoch)

        if config.LOCAL_RANK == 0:
            ddp = config.WORLD_SIZE > 1
            save(model.module if ddp else model, optimizer, epoch, epoch_iou_d, logger, writer, config)

        if scheduler is not None:
            if scheduler.min_lr is not None and optimizer.param_groups[0]['lr'] <= scheduler.min_lr:
                continue
            scheduler.step()
    writer.close()


def train_an_epoch(model, train_data_loader, optimizer, criterion, config, logger, writer, epoch=0):
    logger.info(f'Start Train Epoch {epoch}/{config.TRAIN.EPOCHS - 1}')
    model.train()

    if len(config.MODEL.FINE_TUNE) > 0:
        model.feature_extractor.eval()

    optimizer.zero_grad()

    data_len = len(train_data_loader)
    start_i = data_len * epoch * config.WORLD_SIZE
    bar = enumerate(train_data_loader)
    if config.LOCAL_RANK == 0 and config.SHOW_BAR:
        bar = tqdm(bar, total=data_len, ncols=200)

    device = config.TRAIN.DEVICE
    epoch_loss_d = {}
    for i, gt in bar:
        imgs = gt['image'].to(device, non_blocking=True)
        gt['depth'] = gt['depth'].to(device, non_blocking=True)
        gt['ratio'] = gt['ratio'].to(device, non_blocking=True)
        if 'corner_heat_map' in gt:
            gt['corner_heat_map'] = gt['corner_heat_map'].to(device, non_blocking=True)
        if config.AMP_OPT_LEVEL != "O0" and 'cuda' in device:
            imgs = imgs.type(torch.float16)
            gt['depth'] = gt['depth'].type(torch.float16)
            gt['ratio'] = gt['ratio'].type(torch.float16)
        dt = model(imgs)
        loss, batch_loss_d, epoch_loss_d = calc_criterion(criterion, gt, dt, epoch_loss_d)
        if config.LOCAL_RANK == 0 and config.SHOW_BAR:
            bar.set_postfix(batch_loss_d)

        optimizer.zero_grad()
        if config.AMP_OPT_LEVEL != "O0" and 'cuda' in device:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        global_step = start_i + i * config.WORLD_SIZE + config.LOCAL_RANK
        for key, val in batch_loss_d.items():
            writer.add_scalar(f'TrainBatchLoss/{key}', val, global_step)

    if config.LOCAL_RANK != 0:
        return

    epoch_loss_d = dict(zip(epoch_loss_d.keys(), [np.array(epoch_loss_d[k]).mean() for k in epoch_loss_d.keys()]))
    s = 'TrainEpochLoss: '
    for key, val in epoch_loss_d.items():
        writer.add_scalar(f'TrainEpochLoss/{key}', val, epoch)
        s += f" {key}={val}"
    logger.info(s)
    writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
    logger.info(f"LearningRate: {optimizer.param_groups[0]['lr']}")


@torch.no_grad()
def val_an_epoch(model, val_data_loader, criterion, config, logger, writer, epoch=0):
    model.eval()
    logger.info(f'Start Validate Epoch {epoch}/{config.TRAIN.EPOCHS - 1}')
    data_len = len(val_data_loader)
    start_i = data_len * epoch * config.WORLD_SIZE
    bar = enumerate(val_data_loader)
    if config.LOCAL_RANK == 0 and config.SHOW_BAR:
        bar = tqdm(bar, total=data_len, ncols=200)
    device = config.TRAIN.DEVICE
    epoch_loss_d = {}
    epoch_iou_d = {
        'visible_2d': [],
        'visible_3d': [],
        'full_2d': [],
        'full_3d': [],
        'height': []
    }

    epoch_other_d = {
        'ce': [],
        'pe': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'rmse': [],
        'delta_1': []
    }

    show_index = np.random.randint(0, data_len)
    for i, gt in bar:
        imgs = gt['image'].to(device, non_blocking=True)
        gt['depth'] = gt['depth'].to(device, non_blocking=True)
        gt['ratio'] = gt['ratio'].to(device, non_blocking=True)
        if 'corner_heat_map' in gt:
            gt['corner_heat_map'] = gt['corner_heat_map'].to(device, non_blocking=True)
        dt = model(imgs)

        vis_w = config.TRAIN.VIS_WEIGHT
        visualization = False  # (config.LOCAL_RANK == 0 and i == show_index) or config.SAVE_EVAL

        loss, batch_loss_d, epoch_loss_d = calc_criterion(criterion, gt, dt, epoch_loss_d)

        if config.EVAL.POST_PROCESSING is not None:
            depth = tensor2np(dt['depth'])
            dt['processed_xyz'] = post_process(depth, type_name=config.EVAL.POST_PROCESSING,
                                               need_cube=config.EVAL.FORCE_CUBE)

            if config.EVAL.FORCE_CUBE and config.EVAL.NEED_CPE:
                ce = calc_ce(tensor2np_d(dt), tensor2np_d(gt))
                pe = calc_pe(tensor2np_d(dt), tensor2np_d(gt))

                epoch_other_d['ce'].append(ce)
                epoch_other_d['pe'].append(pe)

            if config.EVAL.NEED_F1:
                f1, precision, recall = calc_f1_score(tensor2np_d(dt), tensor2np_d(gt))
                epoch_other_d['f1'].append(f1)
                epoch_other_d['precision'].append(precision)
                epoch_other_d['recall'].append(recall)

        if config.EVAL.NEED_RMSE:
            rmse, delta_1 = calc_rmse_delta_1(tensor2np_d(dt), tensor2np_d(gt))
            epoch_other_d['rmse'].append(rmse)
            epoch_other_d['delta_1'].append(delta_1)

        visb_iou, full_iou, iou_height, pano_bds, full_iou_2ds = calc_accuracy(tensor2np_d(dt), tensor2np_d(gt),
                                                                               visualization, h=vis_w // 2)
        epoch_iou_d['visible_2d'].append(visb_iou[0])
        epoch_iou_d['visible_3d'].append(visb_iou[1])
        epoch_iou_d['full_2d'].append(full_iou[0])
        epoch_iou_d['full_3d'].append(full_iou[1])
        epoch_iou_d['height'].append(iou_height)

        if config.LOCAL_RANK == 0 and config.SHOW_BAR:
            bar.set_postfix(batch_loss_d)

        global_step = start_i + i * config.WORLD_SIZE + config.LOCAL_RANK

        if writer:
            for key, val in batch_loss_d.items():
                writer.add_scalar(f'ValBatchLoss/{key}', val, global_step)

        if not visualization:
            continue

        gt_grad_imgs, dt_grad_imgs = show_depth_normal_grad(dt, gt, device, vis_w)

        dt_heat_map_imgs = None
        gt_heat_map_imgs = None
        if 'corner_heat_map' in gt:
            dt_heat_map_imgs, gt_heat_map_imgs = show_heat_map(dt, gt, vis_w)

        if config.TRAIN.VIS_MERGE or config.SAVE_EVAL:
            imgs = []
            for j in range(len(pano_bds)):
                # floorplan = np.concatenate([visb_iou[2][j], full_iou[2][j]], axis=-1)
                floorplan = full_iou[2][j]
                margin_w = int(floorplan.shape[-1] * (60/512))
                floorplan = floorplan[:, :, margin_w:-margin_w]

                grad_h = dt_grad_imgs[0].shape[1]
                vis_merge = [
                    gt_grad_imgs[j],
                    pano_bds[j][:, grad_h:-grad_h],
                    dt_grad_imgs[j]
                ]
                if 'corner_heat_map' in gt:
                    vis_merge = [dt_heat_map_imgs[j], gt_heat_map_imgs[j]] + vis_merge
                img = np.concatenate(vis_merge, axis=-2)

                img = np.concatenate([img, ], axis=-1)
                # img = gt_grad_imgs[j]
                imgs.append(img)
            if writer:
                writer.add_images('VIS/Merge', np.array(imgs), global_step)

            if config.SAVE_EVAL:
                for k in range(len(imgs)):
                    img = imgs[k] * 255.0
                    save_path = os.path.join(config.CKPT.RESULT_DIR, f"{gt['id'][k]}_{full_iou_2ds[k]:.5f}.png")
                    Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8)).save(save_path)

        elif writer:
            writer.add_images('IoU/Visible_Floorplan', visb_iou[2], global_step)
            writer.add_images('IoU/Full_Floorplan', full_iou[2], global_step)
            writer.add_images('IoU/Boundary', pano_bds, global_step)
            writer.add_images('Grad/gt', gt_grad_imgs, global_step)
            writer.add_images('Grad/dt', dt_grad_imgs, global_step)

    if config.LOCAL_RANK != 0:
        return

    epoch_loss_d = dict(zip(epoch_loss_d.keys(), [np.array(epoch_loss_d[k]).mean() for k in epoch_loss_d.keys()]))
    s = 'ValEpochLoss: '
    for key, val in epoch_loss_d.items():
        if writer:
            writer.add_scalar(f'ValEpochLoss/{key}', val, epoch)
        s += f" {key}={val}"
    logger.info(s)

    epoch_iou_d = dict(zip(epoch_iou_d.keys(), [np.array(epoch_iou_d[k]).mean() for k in epoch_iou_d.keys()]))
    s = 'ValEpochIoU: '
    for key, val in epoch_iou_d.items():
        if writer:
            writer.add_scalar(f'ValEpochIoU/{key}', val, epoch)
        s += f" {key}={val}"
    logger.info(s)
    epoch_other_d = dict(zip(epoch_other_d.keys(),
                             [np.array(epoch_other_d[k]).mean() if len(epoch_other_d[k]) > 0 else 0 for k in
                              epoch_other_d.keys()]))

    logger.info(f'other acc: {epoch_other_d}')
    return epoch_iou_d, epoch_other_d


if __name__ == '__main__':
    main()
