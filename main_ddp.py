import os
import math
import time
import shutil
import warnings
import builtins
import random

import torch
import torchvision
import torch.optim
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models import GiWModel
from datasets import GiWDataset
from models.loss_functions import MSELoss
from utils import AverageMeter, ProgressMeter
from utils import adjust_learning_rate, save_checkpoint
from utils import prepare_for_training, is_main_process, get_rank, synchronize, torch_distributed_zero_first, \
    accumulate_predictions_from_multiple_gpus, set_random_seed
from opts import parser

best_mse = 0


def main():
    # Lock seeds
    # set_random_seed(30101990)

    args = parser.parse_args()
    args = prepare_for_training(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_mse
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        synchronize()

    args = parser.parse_args()

    print("#" * 60)
    print("Configurations:")
    for key in sorted(args.__dict__.keys()):
        print("- {}: {}".format(key, args.__dict__[key]))
    print("#" * 60)

    model = GiWModel(cur_image_shape=args.cur_shape, fur_image_shape=args.fur_shape,
                     n_fur_states_training=int(args.n_positive / args.positive_ratio))
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            model.load_w(args.resume)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    torch.autograd.set_detect_anomaly(True)
    cudnn.benchmark = True

    train_loader, val_loader = get_loader(args)
    criterion = MSELoss()
    # criterion = torch.nn.MSELoss(reduction='mean')

    init_params_group = [
        {'params': list(filter(lambda p: p.requires_grad, model.module.cur_state_encoder.parameters())),
         'name': 'cur_state_encoder',
         'lr_multi': .5,
         'decay_multi': 1},
        {'params': list(filter(lambda p: p.requires_grad, model.module.fur_state_encoder.parameters())),
         'name': 'fur_state_encoder',
         'lr_multi': .5,
         'decay_multi': 1},
        {'params': list(filter(lambda p: p.requires_grad, model.module.action_estimator.parameters())),
         'name': 'action_estimator',
         'lr_multi': 1,
         'decay_multi': 1}
    ]
    optimizer = torch.optim.Adam(init_params_group, lr=1e-4, weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.max_epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        train(train_loader, model, criterion, optimizer, epoch, args)
        synchronize()

        if epoch % args.eval_freq == 0:
            # evaluate on validation set
            mse = validate(val_loader, model, criterion, epoch, args)

            with torch_distributed_zero_first(get_rank()):
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    # remember validation best mse and save checkpoint
                    is_best = mse > best_mse
                    best_mse = max(mse, best_mse)
                    print("Best mse: {:.4f}".format(best_mse))
                    if is_best or args.save_checkpoint_freq:
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'best_mse': best_mse,
                            'optimizer': optimizer.state_dict(),
                        }, is_best=is_best,
                            filename=os.path.join(args.checkpoint_dir, 'checkpoint_{:04d}.pth.tar'.format(epoch)),
                            best_name=os.path.join(args.checkpoint_dir, 'model_best.pth.tar')
                        )


def train(loader, model, criterion, optimizer, epoch, config):
    forward_time = AverageMeter('forward', ':6.3f')
    backward_time = AverageMeter('backward', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    MSE = AverageMeter('MSE', ':.4e')
    MAE = AverageMeter('MAE', ':.4e')
    progress = ProgressMeter(
        len(loader),
        [data_time, forward_time, backward_time, MSE],
        prefix="Train Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    optimizer.zero_grad()

    end = time.time()
    for step, (cur_state, n_fur_states, n_label_images) in enumerate(loader):
        cur_state = cur_state.cuda(config.gpu, non_blocking=True)
        n_fur_states = n_fur_states.cuda(config.gpu, non_blocking=True)
        n_label_images = n_label_images.cuda(config.gpu, non_blocking=True)
        data_time.update(time.time() - end)
        end = time.time()

        n_pred_images = model(cur_state, n_fur_states)

        loss = criterion(n_pred_images, n_label_images)

        forward_time.update(time.time() - end)
        end = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_time.update(time.time() - end)
        end = time.time()

        if step % config.print_freq == 0 or step == len(loader):
            progress.display(step)


def validate(loader, model, criterion, epoch, config):
    forward_time = AverageMeter('forward', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    MSE = AverageMeter('MSE', ':.4e')
    MAE = AverageMeter('MAE', ':.4e')
    progress = ProgressMeter(
        len(loader),
        [data_time, forward_time, MSE],
        prefix="VAl Epoch: [{}]".format(epoch))

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for step, (cur_state, n_fur_states, n_label_images) in enumerate(loader):
            cur_state = cur_state.cuda(config.gpu, non_blocking=True)
            n_fur_states = n_fur_states.cuda(config.gpu, non_blocking=True)
            n_label_images = n_label_images.cuda(config.gpu, non_blocking=True)
            data_time.update(time.time() - end)
            end = time.time()

            n_pred_images = model(cur_state, n_fur_states)

            loss = criterion(n_pred_images, n_label_images)
            MSE.update(loss.item())

            forward_time.update(time.time() - end)
            end = time.time()

            if step % config.print_freq == 0 or step == len(loader):
                progress.display(step)
        # TODO: accumulate predictions from_multiple gpus

    return MSE.avg


def get_loader(config):
    # TODO: preprocess data to .npy
    # TODO: data transforms, from numpy images to torch tensor with data augmentation
    GiW_cur_trans = torchvision.transforms.Compose([
    ])
    GiW_fur_trans = torchvision.transforms.Compose([
    ])
    GiW_label_trans = torchvision.transforms.Compose([
    ])
    # TODO: dataset partition
    train_dataset = GiWDataset(root_path=config.train_data_dir,
                               transforms=[GiW_cur_trans, GiW_fur_trans, GiW_label_trans],
                               sampling=True,
                               n_positive=config.n_positive,
                               positive_ratio=config.positive_ratio)

    val_dataset = GiWDataset(root_path=config.val_data_dir,
                             transforms=[GiW_cur_trans, GiW_fur_trans, GiW_label_trans],
                             sampling=True,
                             n_positive=config.n_positive,
                             positive_ratio=config.positive_ratio)

    # TODO: Distributed best action state (35 classes) balance Sampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if config.distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if config.distributed else None

    training_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False)

    return training_loader, val_loader


if __name__ == '__main__':
    main()
