""" 
@Date: 2021/07/18
@description:
"""
import os
import models
import torch.distributed as dist
import torch

from torch.nn import init
from torch.optim import lr_scheduler
from utils.time_watch import TimeWatch
from models.other.optimizer import build_optimizer
from models.other.criterion import build_criterion


def build_model(config, logger):
    name = config.MODEL.NAME
    w = TimeWatch(f"Build model: {name}", logger)

    ddp = config.WORLD_SIZE > 1
    if ddp:
        logger.info(f"use ddp")
        dist.init_process_group("nccl", init_method='tcp://127.0.0.1:23456', rank=config.LOCAL_RANK,
                                world_size=config.WORLD_SIZE)

    device = config.TRAIN.DEVICE
    logger.info(f"Creating model: {name} to device:{device}, args:{config.MODEL.ARGS[0]}")

    net = getattr(models, name)
    ckpt_dir = os.path.abspath(os.path.join(config.CKPT.DIR, os.pardir)) if config.DEBUG else config.CKPT.DIR
    if len(config.MODEL.ARGS) != 0:
        model = net(ckpt_dir=ckpt_dir, **config.MODEL.ARGS[0])
    else:
        model = net(ckpt_dir=ckpt_dir)
    logger.info(f'model dropout: {model.dropout_d}')
    model = model.to(device)
    optimizer = None
    scheduler = None

    if config.MODE == 'train':
        optimizer = build_optimizer(config, model, logger)

    config.defrost()
    config.TRAIN.START_EPOCH = model.load(device, logger,  optimizer, best=config.MODE != 'train' or not config.TRAIN.RESUME_LAST)
    config.freeze()

    if config.MODE == 'train' and len(config.MODEL.FINE_TUNE) > 0:
        for param in model.parameters():
            param.requires_grad = False
        for layer in config.MODEL.FINE_TUNE:
            logger.info(f'Fine-tune: {layer}')
            getattr(model, layer).requires_grad_(requires_grad=True)
            getattr(model, layer).reset_parameters()

    model.show_parameter_number(logger)

    if config.MODE == 'train':
        if len(config.TRAIN.LR_SCHEDULER.NAME) > 0:
            if 'last_epoch' not in config.TRAIN.LR_SCHEDULER.ARGS[0].keys():
                config.TRAIN.LR_SCHEDULER.ARGS[0]['last_epoch'] = config.TRAIN.START_EPOCH - 1

            scheduler = getattr(lr_scheduler, config.TRAIN.LR_SCHEDULER.NAME)(optimizer=optimizer,
                                                                              **config.TRAIN.LR_SCHEDULER.ARGS[0])
            logger.info(f"Use scheduler: name:{config.TRAIN.LR_SCHEDULER.NAME} args: {config.TRAIN.LR_SCHEDULER.ARGS[0]}")
            logger.info(f"Current scheduler last lr: {scheduler.get_last_lr()}")
        else:
            scheduler = None

        if config.AMP_OPT_LEVEL != "O0" and 'cuda' in device:
            import apex
            logger.info(f"use amp:{config.AMP_OPT_LEVEL}")
            model, optimizer = apex.amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL, verbosity=0)
        if ddp:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.TRAIN.DEVICE],
                                                              broadcast_buffers=True)  # use rank:0 bn

    criterion = build_criterion(config, logger)
    if optimizer is not None:
        logger.info(f"Finally lr: {optimizer.param_groups[0]['lr']}")
    return model, optimizer, criterion, scheduler
