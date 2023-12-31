"""
Author: wzh
Since: 2023-9-19
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.utils.data
import os
import shutil
import numpy as np
import logging
from models.ics import EfficientNet
from dataloader import DataFolder
from loss import dice_loss, object_dice_losses
from my_transforms import get_transforms
import warnings
import utils
import argparse

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description="Train I2CS Model")
parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training (Glas:4, CRAG:12)')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers to load images')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--checkpoint', type=str, default=None, help='start from checkpoint')
parser.add_argument('--checkpoint_freq', type=int, default=50, help='epoch to save checkpoints')
parser.add_argument('--val_freq', type=int, default=50, help='epoch to validate')
parser.add_argument('--epochs', type=int, default=400, help='number of epochs to train')
parser.add_argument('--save_dir', type=str, default='./experiments/')
parser.add_argument('--dataset', type=str, choices=['GlaS', 'CRAG'], default='CRAG', help='which dataset be used')
parser.add_argument('--gpu', type=list, default=[3,], help='GPUs for training')
parser.add_argument('--gamma1', type=float, default=1, help='weight for dice loss')
parser.add_argument('--gamma2', type=float, default=0.5, help='weight for object-level dice loss')
args = parser.parse_args()

def main():
    global opt, best_iou, num_iter, logger, logger_results
    best_iou = 0

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)

    # set up logger
    logger, logger_results = setup_logging()

    # ----- create model ----- #
    model = EfficientNet(compound_coef=1)
    model = model.cuda()
    torch.backends.cudnn.benchmark = True

    # ----- define optimizer and lr_scheduler----- #
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99),
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 250], gamma=0.1)

    # ----- define criterion ----- #
    criterion = nn.CrossEntropyLoss()

    # ----- define augmentation ----- #
    data_transforms = {
        'train': get_transforms({
        'random_crop' : 416,
        'horizontal_flip': True,
        'vertical_flip': True,
        'random_rotation': 90,
        'to_tensor': 1,
    }),
        'val': get_transforms({
        'to_tensor': 1,
    })}

    # ----- load data ----- #
    dsets = {}
    for x in ['train', 'val']:
        # img_dir = '/home/data2/MedImg/GlandSeg/%s/wzh/train/480x480/Images/' % (args.dataset)
        # target_dir = '/home/data2/MedImg/GlandSeg/%s/wzh/train/480x480/Annotation/' % (args.dataset)
        ## GlaS
        # img_dir = '/home/data2/MedImg/GlandSeg/%s/train/Images' % (args.dataset)
        # target_dir = '/home/data2/MedImg/GlandSeg/%s/train/Annotation' % (args.dataset)
        ## CRAG
        img_dir = '/home/data2/MedImg/GlandSeg/%s/train/TrainSet/Images' % (args.dataset)
        target_dir = '/home/data2/MedImg/GlandSeg/%s/train/TrainSet/Annotation' % (args.dataset)
        dir_list = [img_dir, target_dir]
        # post_fix = ['weight.png', 'label.png']

        dsets[x] = DataFolder(dir_list, data_transform=data_transforms[x])
    train_loader = DataLoader(dsets['train'], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(dsets['val'], batch_size=1, shuffle=False,
                            num_workers=args.num_workers)

    start_epoch = 0
    # ----- optionally load from a checkpoint for validation or resuming training ----- #
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            logger.info("=> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            start_epoch = checkpoint['epoch']
            best_iou = checkpoint['best_iou']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.checkpoint, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.checkpoint))


    # ----- training and validation ----- #
    for epoch in range(start_epoch, args.epochs):
        # train for one epoch or len(train_loader) iterations
        logger.info('Epoch: [{:d}/{:d}]'.format(epoch+1, args.epochs))
        train_results = train(train_loader, model, optimizer, criterion, epoch)
        train_loss, train_loss_ce, train_loss_dice, train_loss_obj_dice, train_pixel_acc, train_iou = train_results

        if (epoch+1) % args.val_freq == 0:
            # evaluate on validation set
            with torch.no_grad():
                val_loss, val_pixel_acc, val_iou = validate(val_loader, model, criterion)

            # check if it is the best accuracy
            is_best = val_iou > best_iou
            best_iou = max(val_iou, best_iou)

        cp_flag = (epoch+1) % args.checkpoint_freq == 0
        scheduler.step()

        if cp_flag:
            save_dir = "%s/%s/%d/" % (args.save_dir, args.dataset, epoch + 1)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_iou': best_iou,
                'optimizer' : optimizer.state_dict(),
            }, epoch, is_best, save_dir, cp_flag)

        if (epoch+1) % args.val_freq == 0:
            # save the training results to txt files
            logger_results.info('{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
                                .format(epoch+1, train_loss, train_loss_ce, train_loss_dice, train_pixel_acc,
                                        train_iou, val_loss, val_pixel_acc, val_iou))


def train(train_loader, model, optimizer, criterion, epoch):
    # list to store the average loss and iou for this epoch
    results = utils.AverageMeter(6)

    # switch to train mode
    model.train()

    for i, sample in enumerate(train_loader):
        img, label = sample

        seg_label = torch.gt(label, 0).int().type(torch.LongTensor)
        img = img.cuda()
        label = label.cuda()
        seg_label = seg_label.cuda()

        # compute output
        output, feature_maps = model(img)

        # compute ce loss
        loss_ce = criterion(output, seg_label)

        # compute dice loss
        score = F.softmax(output, dim=1)
        loss_dice = dice_loss(seg_label, score[:, 1, :, :])

        # compute object-level dice loss
        loss_obj_dice = object_dice_losses(label, score[:, 1, :, :])

        loss = loss_ce + args.gamma1 * loss_dice + args.gamma2 * loss_obj_dice

        # measure accuracy and record loss
        pred = np.argmax(output.data.cpu().numpy(), axis=1)
        metrics = utils.accuracy_pixel_level(pred, seg_label.detach().cpu().numpy())
        pixel_accu, iou = metrics[0], metrics[1]

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        result = [loss.item(), loss_ce.item(), loss_dice.item(), loss_obj_dice.item(),
                  pixel_accu, iou]
        results.update(result, img.size(0))

        if i % 50 == 0:
            logger.info('\tIteration: [{:d}/{:d}]'
                        '\tLoss {r[0]:.4f}'
                        '\tLoss_CE {r[1]:.4f}'
                        '\tLoss_Dice {r[2]:.4f}'
                        '\tLoss_ObjDice {r[3]:.4f}'
                        '\tPixel_Accu {r[4]:.4f}'
                        '\tIoU {r[5]:.4f}'.format(i, len(train_loader), r=results.avg))

    logger.info('\t=> Train Avg: Loss {r[0]:.4f}'
                '\tLoss_Object {r[1]:.4f}'
                '\tLoss_Dice {r[2]:.4f}'
                '\tLoss_ObjDice {r[3]:.4f}'
                '\tPixel_Accu {r[4]:.4f}'
                '\tIoU {r[5]:.4f}'.format(epoch, args.epochs, r=results.avg))

    return results.avg


def validate(val_loader, model, criterion):
    # list to store the losses and accuracies: [loss, pixel_acc, iou ]
    results = utils.AverageMeter(3)

    # switch to evaluate mode
    model.eval()

    for i, sample in enumerate(val_loader):
        img, label = sample
        seg_label = torch.gt(label, 0).int().type(torch.LongTensor)
        img = img.cuda()
        label = label.cuda()
        seg_label = seg_label.cuda()

        # compute output
        output, feature_maps = model(img)

        # compute ce loss
        loss_ce = criterion(output, seg_label)

        # compute dice loss
        score = F.softmax(output, dim=1)
        loss_dice = dice_loss(seg_label, score[:, 1, :, :])

        # compute object-level dice loss
        loss_obj_dice = object_dice_losses(label, score[:, 1, :, :])

        loss = loss_ce + args.gamma1 * loss_dice + args.gamma2 * loss_obj_dice

        # measure accuracy and record loss
        pred = np.argmax(output.cpu().numpy(), axis=1)
        metrics = utils.accuracy_pixel_level(pred, seg_label.detach().cpu().numpy())
        pixel_accu = metrics[0]
        iou = metrics[1]

        results.update([loss.item(), pixel_accu, iou])

    logger.info('\t=> Val Avg:   Loss {r[0]:.4f}\tPixel_Acc {r[1]:.4f}'
                '\tIoU {r[2]:.4f}'.format(r=results.avg))

    return results.avg


def save_checkpoint(state, epoch, is_best, save_dir, cp_flag):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    if not os.path.exists(cp_dir):
        os.mkdir(cp_dir)
    filename = '{:s}/checkpoint.pth.tar'.format(cp_dir)
    torch.save(state, filename)
    if cp_flag:
        shutil.copyfile(filename, '{:s}/checkpoint_{:d}.pth.tar'.format(cp_dir, epoch+1))
    if is_best:
        shutil.copyfile(filename, '{:s}/checkpoint_best.pth.tar'.format(cp_dir))


def setup_logging():
    mode = 'w'

    save_dir = "%s/%s/" % (args.save_dir, args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train.log'.format(save_dir, mode=mode))
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%Y-%m-%d %I:%M')
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create logger for epoch results
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(save_dir), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(save_dir))
    if mode == 'w':
        logger_results.info('epoch\ttrain_loss\ttrain_loss_CE\ttrain_loss_var\ttrain_acc\ttrain_iou\t'
                            'val_loss\tval_acc\tval_iou')

    return logger, logger_results


if __name__ == '__main__':
    main()
