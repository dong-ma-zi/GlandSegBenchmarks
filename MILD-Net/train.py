"""
Author: my
Since: 2023-9-8
Modifier: wzh
"""
import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.utils.data
import os
import shutil
import numpy as np
import logging
from models.mildNet import MILDNet
from dataloader import DataFolder
from my_transforms import get_transforms
import warnings
import utils
import argparse

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description="Train DCAN Model")
parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers to load images')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--checkpoint', type=str, default=None, help='start from checkpoint')
parser.add_argument('--checkpoint_freq', type=int, default=10, help='epoch to save checkpoints')
parser.add_argument('--val_freq', type=int, default=10, help='epoch to validate')
parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
parser.add_argument('--save_dir', type=str, default='./experiments')
parser.add_argument('--dataset', type=str, choices=['GlaS', 'CRAG'], default='GlaS', help='which dataset be used')
parser.add_argument('--gpu', type=list, default=[3,], help='GPUs for training')
parser.add_argument('--discount_weight', type=float, default=1, help='discount weight')
args = parser.parse_args()

def main():
    global best_iou, num_iter, logger, logger_results
    best_iou = 0

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)

    # set up logger
    logger, logger_results = setup_logging()

    # ----- create model ----- #
    model = MILDNet(n_class=2)
    model = model.cuda()
    torch.backends.cudnn.benchmark = True

    # ----- define optimizer and lr_scheduler----- #
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99),
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)

    # ----- define criterion ----- #
    criterion = nn.CrossEntropyLoss()

    # ----- define augmentation ----- #
    data_transforms = {
        'train': get_transforms({
        #'random_crop': 480,
        'horizontal_flip': True,
        #'random_affine': 0.3,
        'random_elastic': [6, 15],
        'random_rotation': 90,
        'to_tensor': 1,
    }),
        'val': get_transforms({
        #'random_crop': 480,
        'to_tensor': 1,
    })}

    # ----- load data ----- #
    dsets = {}
    for x in ['train', 'val']:
        img_dir = '/home/data2/MedImg/GlandSeg/%s/wzh/train/480x480/Images/' % (args.dataset)
        target_dir = '/home/data2/MedImg/GlandSeg/%s/wzh/train/480x480/Annotation/' % (args.dataset)
        dir_list = [img_dir, target_dir]

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
        train_loss, train_loss_ce, train_loss_var, train_pixel_acc, train_iou = train_results

        if (epoch+1) % args.val_freq == 0:
            # evaluate on validation set
            with torch.no_grad():
                val_loss, val_pixel_acc, val_iou = validate(val_loader, model, criterion)

            # check if it is the best accuracy
            is_best = val_iou > best_iou
            best_iou = max(val_iou, best_iou)

        cp_flag = (epoch+1) % args.checkpoint_freq == 0
        scheduler.step()
        if epoch != 0 and (epoch + 1) % 50 == 0:
            args.discount_weight = 0.1 * args.discount_weight
            logger.info("=> discount weight degrade to {:.6f} ".format(args.discount_weight))

        if cp_flag:
            save_dir = "%s/%s/%d/" % (args.save_dir, args.dataset, epoch + 1)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_iou': best_iou,
                'optimizer' : optimizer.state_dict(),
            }, epoch, is_best, save_dir, cp_flag)

        if (epoch+1) % args.val_freq == 0:
            # save the training results to txt files
            logger_results.info('{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
                                .format(epoch+1, train_loss, train_loss_ce, train_loss_var, train_pixel_acc,
                                        train_iou, val_loss, val_pixel_acc, val_iou))


def train(train_loader, model, optimizer, criterion, epoch):
    # list to store the average loss and iou for this epoch
    results = utils.AverageMeter(5)

    # switch to train mode
    model.train()

    for i, sample in enumerate(train_loader):
        img, inst_label, label_contour = sample

        label = torch.gt(inst_label, 0).int().type(torch.LongTensor)
        img = img.cuda()
        label = label.cuda()
        label_contour = label_contour.int().type(torch.LongTensor).cuda()

        # compute output
        o_output, a_o_output, c_output, a_c_output = model(img)

        # compute loss
        loss_object = criterion(o_output, label)
        loss_object1 = criterion(a_o_output, label)
        loss_contour = criterion(c_output, label_contour)
        loss_contour1 = criterion(a_c_output, label_contour)

        loss_o = loss_object + args.discount_weight * loss_object1
        loss_c = loss_contour + args.discount_weight * loss_contour1
        loss = loss_o + loss_c

        # measure accuracy and record loss
        pred = np.argmax(o_output.data.cpu().numpy(), axis=1)
        metrics = utils.accuracy_pixel_level(pred, label.detach().cpu().numpy())
        pixel_accu, iou = metrics[0], metrics[1]

        result = [loss.item(), loss_o.item(), loss_c.item(), pixel_accu, iou]

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            logger.info('\tIteration: [{:d}/{:d}]'
                        '\tLoss {r[0]:.4f}'
                        '\tLoss_Object {r[1]:.4f}'
                        '\tLoss_Contour {r[2]:.4f}'
                        '\tPixel_Accu {r[3]:.4f}'
                        '\tIoU {r[4]:.4f}'.format(i, len(train_loader), r=[loss.item(), loss_o.item(), loss_c.item(), pixel_accu, iou]))

    logger.info('\t=> Train Avg: Loss {r[0]:.4f}'
                '\tLoss_Object {r[1]:.4f}'
                '\tLoss_Contour {r[2]:.4f}'
                '\tPixel_Accu {r[3]:.4f}'
                '\tIoU {r[4]:.4f}'.format(epoch, args.epochs, r=[loss.item(), loss_o.item(), loss_c.item(), pixel_accu, iou]))

    return results.avg


def validate(val_loader, model, criterion):
    # list to store the losses and accuracies: [loss, pixel_acc, iou ]
    results = utils.AverageMeter(3)

    # switch to evaluate mode
    model.eval()

    for i, sample in enumerate(val_loader):
        img, inst_label, label_contour = sample
        label = torch.gt(inst_label, 0).int().type(torch.LongTensor)
        img = img.cuda()
        label = label.cuda()
        label_contour = label_contour.int().type(torch.LongTensor).cuda()

        # compute output
        o_output, a_o_output, c_output, a_c_output = model(img)

        # compute loss
        loss_object = criterion(o_output, label)
        loss_object1 = criterion(a_o_output, label)
        loss_contour = criterion(c_output, label_contour)
        loss_contour1 = criterion(a_c_output, label_contour)

        loss_o = loss_object + args.discount_weight * loss_object1
        loss_c = loss_contour + args.discount_weight * loss_contour1
        loss = loss_o + loss_c

        # measure accuracy and record loss
        pred = np.argmax(o_output.cpu().numpy(), axis=1)
        metrics = utils.accuracy_pixel_level(pred, label.detach().cpu().numpy())
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
