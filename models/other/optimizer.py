""" 
@Date: 2021/07/18
@description:
"""
from torch import optim as optim


def build_optimizer(config, model, logger):
    name = config.TRAIN.OPTIMIZER.NAME.lower()

    optimizer = None
    if name == 'sgd':
        optimizer = optim.SGD(model.parameters(), momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif name == 'adam':
        optimizer = optim.Adam(model.parameters(), eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                               lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    logger.info(f"Build optimizer: {name}, lr:{config.TRAIN.BASE_LR}")

    return optimizer
