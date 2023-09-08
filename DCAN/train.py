"""
Author: my
Since: 2023-9-8
Modifier: wzh
"""

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data
import os
import shutil
import numpy as np
from skimage import measure
import logging
from models import DCAN
from dataloader import DataFolder
from my_transforms import get_transforms
import utils
import argparse

parser = argparse.ArgumentParser(description="Train DCAN Model")
parser.add_argument('--batch-size', type=int, default=4, help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--checkpoint', type=str, default=None, help='start from checkpoint')
parser.add_argument('--checkpoint_freq', type=int, default=10, help='epoch to save checkpoints')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
args = parser.parse_args()

def main():
    global opt, best_iou, num_iter, logger, logger_results
    best_iou = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpu'])

    # set up logger
    logger, logger_results = setup_logging(opt)
    opt.print_options(logger)

    # ----- create model ----- #
    model = DCAN(n_class=2)
    model = nn.DataParallel(model)
    model = model.cuda()
    torch.backends.cudnn.benchmark = True

    # ----- define optimizer ----- #
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99),
                                 weight_decay=args.weight_decay)

    # ----- define criterion ----- #
    criterion = nn.CrossEntropyLoss()

    data_transforms = {'train': get_transforms(opt.transform['train']),
                       'val': get_transforms(opt.transform['val'])}

    # ----- load data ----- #
    dsets = {}
    for x in ['train', 'val']:
        img_dir = '{:s}/{:s}'.format(opt.train['img_dir'], x)
        target_dir = '{:s}/{:s}'.format(opt.train['label_dir'], x)
        weight_map_dir = '{:s}/{:s}'.format(opt.train['weight_map_dir'], x)
        dir_list = [img_dir, weight_map_dir, target_dir]
        post_fix = ['weight.png', 'label.png']

        num_channels = [3, 1, 3]
        dsets[x] = DataFolder(dir_list, post_fix, num_channels, data_transforms[x])
    train_loader = DataLoader(dsets['train'], batch_size=opt.train['batch_size'], shuffle=True,
                              num_workers=opt.train['workers'])
    val_loader = DataLoader(dsets['val'], batch_size=1, shuffle=False,
                            num_workers=opt.train['workers'])

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
    for epoch in range(start_epoch, args.epoch):
        # train for one epoch or len(train_loader) iterations
        logger.info('Epoch: [{:d}/{:d}]'.format(epoch+1, args.epoch))
        train_results = train(train_loader, model, optimizer, criterion, epoch)
        train_loss, train_loss_ce, train_loss_var, train_pixel_acc, train_iou = train_results

        # evaluate on validation set
        with torch.no_grad():
            val_loss, val_pixel_acc, val_iou = validate(val_loader, model, criterion)

        # check if it is the best accuracy
        is_best = val_iou > best_iou
        best_iou = max(val_iou, best_iou)

        cp_flag = (epoch+1) % args.checkpoint_freq == 0

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_iou': best_iou,
            'optimizer' : optimizer.state_dict(),
        }, epoch, is_best, opt.train['save_dir'], cp_flag)

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
        input, weight_map, target = sample
        weight_map = weight_map.float().div(20)
        if weight_map.dim() == 4:
            weight_map = weight_map.squeeze(1)
        weight_map_var = weight_map.cuda()

        if torch.max(target) == 255:
            target = target / 255
        if target.dim() == 4:
            target = target.squeeze(1)

        input_var = input.cuda()
        target_var = target.cuda()

        # compute output
        output = model(input_var)

        log_prob_maps = F.log_softmax(output, dim=1)
        loss_map = criterion(log_prob_maps, target_var)
        loss_map *= weight_map_var
        loss_CE = loss_map.mean()

        if opt.train['alpha'] != 0:
            prob_maps = F.softmax(output, dim=1)

            # label instances in target
            target_labeled = torch.zeros(target.size()).long()
            for k in range(target.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(target[k].numpy() == 1))
                # utils.show_figures((target[k].numpy(), target[k].numpy()==1, target_labeled[k].numpy()))
            loss = loss_CE
        else:
            loss = loss_CE

        # measure accuracy and record loss
        pred = np.argmax(log_prob_maps.data.cpu().numpy(), axis=1)
        metrics = utils.accuracy_pixel_level(pred, target.numpy())
        pixel_accu, iou = metrics[0], metrics[1]

        result = [loss.item(), loss_CE.item(), pixel_accu, iou]
        results.update(result, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del input_var, output, target_var, log_prob_maps, loss

        if i % opt.train['log_interval'] == 0:
            logger.info('\tIteration: [{:d}/{:d}]'
                        '\tLoss {r[0]:.4f}'
                        '\tLoss_CE {r[1]:.4f}'
                        '\tLoss_var {r[2]:.4f}'
                        '\tPixel_Accu {r[3]:.4f}'
                        '\tIoU {r[4]:.4f}'.format(i, len(train_loader), r=results.avg))

    logger.info('\t=> Train Avg: Loss {r[0]:.4f}'
                '\tLoss_CE {r[1]:.4f}'
                '\tLoss_var {r[2]:.4f}'
                '\tPixel_Accu {r[3]:.4f}'
                '\tIoU {r[4]:.4f}'.format(epoch, opt.train['num_epochs'], r=results.avg))

    return results.avg


def validate(val_loader, model, criterion):
    # list to store the losses and accuracies: [loss, pixel_acc, iou ]
    results = utils.AverageMeter(3)

    # switch to evaluate mode
    model.eval()

    for i, sample in enumerate(val_loader):
        input, weight_map, target = sample
        weight_map = weight_map.float().div(20)
        if weight_map.dim() == 4:
            weight_map = weight_map.squeeze(1)
        weight_map_var = weight_map.cuda()

        # for b in range(input.size(0)):
        #     utils.show_figures((input[b, 0, :, :].numpy(), target[b,0,:,:].numpy(), weight_map[b, :, :]))

        if torch.max(target) == 255:
            target = target / 255
        if target.dim() == 4:
            target = target.squeeze(1)

        target_var = target.cuda()

        size = opt.train['input_size']
        overlap = opt.train['val_overlap']
        output = utils.split_forward(model, input, size, overlap, opt.model['out_c'])

        log_prob_maps = F.log_softmax(output, dim=1)
        loss_map = criterion(log_prob_maps, target_var)
        loss_map *= weight_map_var
        loss_CE = loss_map.mean()

        if opt.train['alpha'] != 0:
            prob_maps = F.softmax(output, dim=1)

            target_labeled = torch.zeros(target.size()).long()
            for k in range(target.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(target[k].numpy() == 1))
                # utils.show_figures((target[k].numpy(), target[k].numpy()==1, target_labeled[k].numpy()))
            loss = loss_CE
        else:
            loss = loss_CE

        # measure accuracy and record loss
        pred = np.argmax(log_prob_maps.data.cpu().numpy(), axis=1)
        metrics = utils.accuracy_pixel_level(pred, target.numpy())
        pixel_accu = metrics[0]
        iou = metrics[1]

        results.update([loss.item(), pixel_accu, iou])

        del output, target_var, log_prob_maps, loss

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


def setup_logging(opt):
    mode = 'a' if opt.train['checkpoint'] else 'w'

    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train.log'.format(opt.train['save_dir']), mode=mode)
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
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(opt.train['save_dir']))
    if mode == 'w':
        logger_results.info('epoch\ttrain_loss\ttrain_loss_CE\ttrain_loss_var\ttrain_acc\ttrain_iou\t'
                            'val_loss\tval_acc\tval_iou')

    return logger, logger_results


if __name__ == '__main__':
    main()
