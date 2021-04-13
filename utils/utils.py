import os
import re
import math
import json
import cv2
import random
from glob import glob

import shutil
from datetime import timezone, datetime, timedelta

import torch
import numpy as np
import matplotlib.pyplot as plt


def save_checkpoint(state, is_best, filename, best_name):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_name)


def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate based on schedule"""
    lr = config.lr
    if config.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / config.max_epochs))
    else:  # stepwise lr schedule
        for milestone in config.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_multi']
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = lr_schedule[iteration]


def prepare_for_training(config):
    assert config.dataset == "GiW", NotImplementedError('Unknown dataset ' + config.dataset)
    config.train_data_dir = '{}/training'.format(config.dataset_root)
    config.val_data_dir = '{}/validation'.format(config.dataset_root)
    config.cur_shape = [7, 360, 640]
    config.fur_shape = [7, 45, 80]
    config.label_shape = [1, 23, 40]
    config.n_positive = 4
    config.positive_ratio = 1. / 2.
    config.n_fur_states = 35
    # prepare_output_dirs
    config.save_dir = output_subdir(config)
    config.checkpoint_dir = os.path.join(config.save_dir, 'checkpoints')
    config.log_dir = os.path.join(config.save_dir, 'logs')

    # And create them
    if os.path.exists(config.save_dir):
        # Only occurs when experiment started the same minute
        shutil.rmtree(config.save_dir)

    os.mkdir(config.save_dir)
    os.mkdir(config.checkpoint_dir)
    os.mkdir(config.log_dir)
    with open(os.path.join(config.save_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(config), indent=4, sort_keys=True))

    return config


def output_subdir(config):
    prefix = datetime.now(tz=timezone(timedelta(hours=+8))).strftime('%Y%m%d_%H%M')
    subdir = "{}_{}_{}_lr{:.3f}".format(prefix, config.dataset, config.backbone, config.lr)
    return os.path.join(config.save_dir, subdir)


def cleanup_checkpoint_dir(config):
    checkpoint_files = glob(os.path.join(config.checkpoint_dir, 'save_*.pth'))
    checkpoint_files.sort()
    if len(checkpoint_files) > config.checkpoints_num_keep:
        os.remove(checkpoint_files[0])


def duration_to_string(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02d}:{:02d}:{:02d}'.format(int(hours), int(minutes), int(seconds))


def init_cropping_scales(config):
    # Determine cropping scales
    config.scales = [config.initial_scale]
    for i in range(1, config.num_scales):
        config.scales.append(config.scales[-1] * config.scale_step)
    return config


def set_lr_scheduling_policy(config):
    if config.lr_plateau_patience > 0 and not config.no_eval:
        config.lr_scheduler = 'plateau'
    else:
        config.lr_scheduler = 'multi_step'
    return config


def normalize(samples, min, max):
    # type: (np.ndarray, float, float) -> np.ndarray
    """
    Normalize scores as in Eq. 10

    :param samples: the scores to be normalized.
    :param min: the minimum of the desired scores.
    :param max: the maximum of the desired scores.
    :return: the normalized scores
    """
    return (samples - min) / (max - min)


def set_random_seed(seed):
    # type: (int) -> None
    """
    Sets random seeds.
    :param seed: the seed to be set for all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
