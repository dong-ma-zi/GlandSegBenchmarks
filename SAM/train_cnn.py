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
from models.pspnet import PSPNet
from options import Options
from dataloader import DataFolder
from my_transforms import get_transforms
import warnings
import utils
import argparse
# from loss_func import cross_entropy

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description="Train SAM Segmentation Model")
parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers to load images')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--checkpoint', type=str, default=None, help='start from checkpoint')
parser.add_argument('--checkpoint_freq', type=int, default=10, help='epoch to save checkpoints')
parser.add_argument('--val_freq', type=int, default=10, help='epoch to validate')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--save_dir', type=str, default='./experimentsP')
parser.add_argument('--dataset', type=str, choices=['GlaS', 'CRAG'], default='MoNuSeg', help='which dataset be used')
parser.add_argument('--desc', type=str, default='PSP')
parser.add_argument('--gpu', type=list, default=[0,], help='GPUs for training')
args = parser.parse_args()

def main():
    global best_iou, num_iter, logger, logger_results
    best_iou = 0
    # opt = Options(isTrain=True)
    # opt.parse()
    # opt.save_options()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)

    # set up logger
    logger, logger_results = setup_logging()
    # opt.print_options(logger)

    # ----- create model ----- #
    model = PSPNet(classes=2)
    # load pretrained vit res
    # weight_dict = torch.load("/home/data1/my/Project/segment-anything-main/sam_vit_b.pth")
    # weight_dict_load = {k: v for k, v in weight_dict.items() if k in model.state_dict() and model.state_dict()[k].numel() == v.numel()}
    # model.load_state_dict(weight_dict_load,
    #                       strict=False)
    model = model.cuda()
    torch.backends.cudnn.benchmark = True

    # ----- define optimizer and lr_scheduler----- #
    # frozen_list = ['image_encoder']
    # for param in model.named_parameters():
    #     if param[0].split('.')[0] in frozen_list:
    #         print('frozen layers:', param[0])
    #         param[1].requires_grad = False

    # Init optimizer
    # optimizer = torch.optim.Adam(
    #     [{'params': model.image_encoder.parameters(), 'lr': args.lr * 0.01},
    #      {'params': model.mask_decoder.parameters()}],
    #     lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99),
                                 weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300], gamma=0.1)

    # ----- define criterion ----- #
    criterion = nn.CrossEntropyLoss()

    # ----- define augmentation ----- #
    data_transforms = {
        'train': get_transforms({
        'horizontal_flip': True,
        'vertical_flip': True,
        # 'random_elastic': [6, 15],
        'random_rotation': 90,
        'to_tensor': 1,
        # 'normalize': [[0.787, 0.511, 0.785], [0.167, 0.248, 0.131]],
    }),
        'val': get_transforms({
        'to_tensor': 1,
        # 'normalize': [[0.787, 0.511, 0.785], [0.167, 0.248, 0.131]],
    })}

    # ----- load data ----- #
    # data_path = {'train': '/home/data2/MedImg/GlandSeg/GlaS/train/TrainSet',
    #              'val': '/home/data2/MedImg/GlandSeg/GlaS/train/ValidSet'}
    data_path = {'train': '/home/data2/MedImg/NucleiSeg/MoNuSeg/extracted_mirror/train/512x512_256x256/',
                 'val': '/home/data2/MedImg/NucleiSeg/MoNuSeg/Test'}

    dsets = {}
    for x in ['train', 'val']:
        img_dir = os.path.join(data_path[x], 'Images')
        target_dir = os.path.join(data_path[x], 'Annotation')

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
        train_results = train(train_loader, model, criterion, optimizer, epoch)
        train_loss, train_pixel_acc, train_iou = train_results

        if (epoch + 1) % args.val_freq == 0:
            # evaluate on validation set
            with torch.no_grad():
                val_loss, val_pixel_acc, val_iou = validate(val_loader, model, criterion)

            # check if it is the best accuracy
            is_best = val_iou > best_iou
            best_iou = max(val_iou, best_iou)

        cp_flag = (epoch + 1) % args.checkpoint_freq == 0
        # scheduler.step()

        if cp_flag:
            save_dir = "%s/%s_%s/" % (args.save_dir, args.dataset, args.desc)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_iou': best_iou,
                'optimizer': optimizer.state_dict(),
            }, epoch, is_best, save_dir, cp_flag)

        if (epoch + 1) % args.val_freq == 0:
            # save the training results to txt files
            logger_results.info('{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
                                .format(epoch+1, train_loss, train_pixel_acc,
                                        train_iou, val_loss, val_pixel_acc, val_iou))


def train(train_loader, model, criterion, optimizer, epoch):
    # list to store the average loss and iou for this epoch
    results = utils.AverageMeter(3)

    # switch to train mode
    model.train()

    for i, sample in enumerate(train_loader):
        img, label = sample

        # label = label_seg.int().type(torch.LongTensor).cuda()
        img = img.cuda()
        b, c, h, w = img.shape
        label = label.cuda()

        # compute output
        o_output = model(img)
        # mask size: [batch*num_classes, num_multi_class, H, W], iou_pred: [batch*num_classes, 1]
        # o_output = o_output.view(b, -1, h, w)
        # compute loss
        loss = criterion(o_output, label)

        # measure accuracy and record loss
        pred = np.argmax(o_output.data.cpu().numpy(), axis=1)
        metrics = utils.accuracy_pixel_level(pred, label.detach().cpu().numpy())
        pixel_accu, iou = metrics[0], metrics[1]

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if i % opt.train['log_interval'] == 0:
        # if i % 10 == 0:
        #     logger.info('\tIteration: [{:d}/{:d}]'
        #                 '\tLoss {r[0]:.4f}'
        #                 '\tPixel_Accu {r[1]:.4f}'
        #                 '\tIoU {r[2]:.4f}'.format(i, len(train_loader), r=[loss.item(), pixel_accu, iou]))

    logger.info('\t=> Train Avg: Loss {r[0]:.4f}'
                '\tPixel_Accu {r[1]:.4f}'
                '\tIoU {r[2]:.4f}'.format(epoch, args.epochs, r=[loss.item(), pixel_accu, iou]))

    return results.avg


def validate(val_loader, model, criterion):
    # list to store the losses and accuracies: [loss, pixel_acc, iou ]
    results = utils.AverageMeter(3)

    # switch to evaluate mode
    model.eval()

    for i, sample in enumerate(val_loader):
        img, label_seg = sample
        label = label_seg.int().type(torch.LongTensor).cuda()
        img = img.cuda()
        b, c, h, w = img.shape
        label = label.cuda()

        # compute output
        o_output = model(img)
        # mask size: [batch*num_classes, num_multi_class, H, W], iou_pred: [batch*num_classes, 1]
        # o_output = o_output.view(b, -1, h, w)
        # compute loss
        loss = criterion(o_output, label)

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
    # mode = 'a' if opt.train['checkpoint'] else 'w'
    mode = 'w'
    save_dir = "%s/%s_%s/" % (args.save_dir, args.dataset, args.desc)
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

