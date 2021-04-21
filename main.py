import os
import math
import time
import shutil
import warnings

import torch
import torchvision
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from models import GiWModel
from datasets import GiWDataset
from models.loss_functions import MSELoss
from utils import AverageMeter, ProgressMeter
from utils import prepare_for_training
from utils import adjust_learning_rate, save_checkpoint
from opts import parser

best_mse = 0


def main():
    args = parser.parse_args()
    args = prepare_for_training(args)

    print("#" * 60)
    print("Configurations:")
    for key in sorted(args.__dict__.keys()):
        print("- {}: {}".format(key, args.__dict__[key]))
    print("#" * 60)

    # n_fur_states_training=int(args.n_positive / args.positive_ratio)
    model = GiWModel(cur_image_shape=args.cur_shape, fur_image_shape=args.fur_shape,
                     n_fur_states_training=args.n_fur_states)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            model.load_w(args.resume)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    if args.distributed:
        pass
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        warnings.warn("Using CPU for Model Training!!!")

    torch.autograd.set_detect_anomaly(True)
    cudnn.benchmark = True

    train_loader, val_loader = get_loader(args)
    criterion = MSELoss()

    init_params_group = [
        {'params': list(filter(lambda p: p.requires_grad, model.cur_state_encoder.parameters())),
         'name': 'cur_state_encoder',
         'lr_multi': .5,
         'decay_multi': 1},
        {'params': list(filter(lambda p: p.requires_grad, model.fur_state_encoder.parameters())),
         'name': 'fur_state_encoder',
         'lr_multi': .5,
         'decay_multi': 1},
        {'params': list(filter(lambda p: p.requires_grad, model.action_estimator.parameters())),
         'name': 'action_estimator',
         'lr_multi': 1,
         'decay_multi': 1}
    ]
    optimizer = torch.optim.Adam(init_params_group, lr=1e-4, weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.max_epochs):

        adjust_learning_rate(optimizer, epoch, args)

        train(train_loader, model, criterion, optimizer, epoch, args)

        if epoch % args.eval_freq == 0:
            # evaluate on validation set
            mse = validate(val_loader, model, criterion, epoch, args)
            # remember validation best mse and save checkpoint
            is_best = mse < best_mse
            best_mse = min(mse, best_mse)
            print("Best mse: {:.4f}".format(best_mse))
            if is_best or epoch % args.save_checkpoint_freq == 0:
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
        if config.gpu is not None:
            cur_state = cur_state.cuda(config.gpu)
            n_fur_states = n_fur_states.cuda(config.gpu)
            n_label_images = n_label_images.cuda(config.gpu)

        data_time.update(time.time() - end)
        end = time.time()

        n_pred_images = model(cur_state, n_fur_states)

        loss = criterion(n_pred_images, n_label_images)
        loss = torch.sum(loss)

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
            if config.gpu is not None:
                cur_state = cur_state.cuda(config.gpu)
                n_fur_states = n_fur_states.cuda(config.gpu)
                n_label_images = n_label_images.cuda(config.gpu)

            data_time.update(time.time() - end)
            end = time.time()

            n_pred_images = model(cur_state, n_fur_states)

            loss = criterion(n_pred_images, n_label_images)
            MSE.update(loss.item())

            forward_time.update(time.time() - end)
            end = time.time()

            if step % config.print_freq == 0 or step == len(loader):
                progress.display(step)

    return MSE.avg


def get_loader(config):
    # TODO: preprocess data to .npy
    # TODO: data transforms, from numpy images to torch tensor with augmentation
    GiW_cur_trans = torchvision.transforms.Compose([
    ])
    GiW_fur_trans = torchvision.transforms.Compose([
    ])
    GiW_label_trans = torchvision.transforms.Compose([
    ])
    # TODO: dataset partition
    train_dataset = GiWDataset(root_path=config.train_data_dir,
                               transforms=[GiW_cur_trans, GiW_fur_trans, GiW_label_trans],
                               sampling=False)

    val_dataset = GiWDataset(root_path=config.val_data_dir,
                             transforms=[GiW_cur_trans, GiW_fur_trans, GiW_label_trans],
                             sampling=False)

    # TODO: best action state (35 classes) balance Sampler
    action_weights = list()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(action_weights, config)

    training_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
        drop_last=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
        drop_last=False)

    return training_loader, val_loader


if __name__ == '__main__':
    main()
