# Code for MedT

import lib
import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import torch
import torch.nn.init as init
from utils import JointTransform2D, ImageToImage2D, Image2D
from metrics import jaccard_index, f1_score, LogNLLLoss,classwise_f1
from utils import chk_mkdir, Logger, MetricList
import cv2
from functools import partial
from random import randint
import timeit
import utils
import logging

parser = argparse.ArgumentParser(description='MedT')
parser.add_argument('--workers', default=16, type=int, help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=400, type=int, help='number of total epochs to run(default: 400)')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=1, type=int, help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float, help='weight decay (default: 1e-5)')
parser.add_argument('--save_freq', type=int,default = 10)
parser.add_argument('--dataset', type=str, choices=['GlaS', 'CRAG'], default='GlaS', help='which dataset be used')

parser.add_argument('--modelname', default='MedT', type=str, help='type of model')
parser.add_argument('--cuda', default="on", type=str, help='switch on/off cuda option (default: on)')
parser.add_argument('--aug', default='off', type=str, help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='default', type=str, help='load a pretrained model')
parser.add_argument('--save_dir', type=str, default='./experiments/')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--imgsize', type=int, default=480)
parser.add_argument('--gray', default='no', type=str)
parser.add_argument('--device', default='cuda:3', type=str)
parser.add_argument('--gpu', type=list, default=[3], help='GPUs for training')
args = parser.parse_args()

gray_ = args.gray
aug = args.aug
modelname = args.modelname
imgsize = args.imgsize

def main():
    global logger

    if gray_ == "yes":
        from utils_gray import JointTransform2D, ImageToImage2D, Image2D
        imgchant = 1
    else:
        from utils import JointTransform2D, ImageToImage2D, Image2D
        imgchant = 3

    if args.crop is not None:
        crop = (args.crop, args.crop)
    else:
        crop = None


    # set up logger
    logger, logger_results = setup_logging()


    # ----- load data ----- #
    if args.dataset == 'GlaS':
        img_dir = '/home/data2/MedImg/GlandSeg/%s/wzh/train/480x480/' % (args.dataset)
    elif args.dataset == 'CRAG':
        img_dir = '/home/data2/MedImg/GlandSeg/%s/wzh/train/480x480/' % (args.dataset)

    tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
    tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
    train_dataset = ImageToImage2D(img_dir, tf_train)
    val_dataset = ImageToImage2D(img_dir, tf_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, 1, shuffle=True)

    # ----- create model ----- #
    device = torch.device(args.device)
    if modelname == "axialunet":
        model = lib.models.axialunet(img_size = imgsize, imgchan = imgchant)
    elif modelname == "MedT":
        model = lib.models.axialnet.MedT(img_size = imgsize, imgchan = imgchant)
    elif modelname == "gatedaxialunet":
        model = lib.models.axialnet.gated(img_size = imgsize, imgchan = imgchant)
    elif modelname == "logo":
        model = lib.models.axialnet.logo(img_size = imgsize, imgchan = imgchant)
    logger.info("=> Creating model!")
    if len(args.gpu) > 1:
        print("Let's use", len(args.gpu), "GPUs!")
        model = nn.DataParallel(model, device_ids=args.gpu)
    model.to(device)

    criterion = LogNLLLoss()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,
                                 weight_decay=1e-5)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("=> Total_params: {}".format(pytorch_total_params))

    seed = 3000
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.set_deterministic(True)
    # random.seed(seed)

    for epoch in range(args.epochs):
        train(train_loader, model, optimizer, criterion, epoch)
        if epoch == 10:
            for param in model.parameters():
                param.requires_grad =True
        if (epoch % args.save_freq) ==0:
            with torch.no_grad():
                validate(val_loader, model, criterion)

            save_dir = "%s/%s/%d/" % (args.save_dir, args.dataset, epoch + 1)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(model.state_dict(), save_dir + args.modelname + ".pth")
            torch.save(model.state_dict(), save_dir + "final_model.pth")


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

def train(train_loader, model, optimizer, criterion, epoch):
    epoch_running_loss = 0
    results = utils.AverageMeter(3)
    for batch_idx, (imgs, labels, *rest) in enumerate(train_loader):
        imgs = Variable(imgs.to(device=args.device))
        labels = Variable(labels.to(device=args.device))

        # ===================forward=====================
        output = model(imgs)

        # measure accuracy and record loss
        seg_labels = labels.detach().cpu().numpy()
        pred = np.argmax(output.data.cpu().numpy(), axis=1)
        seg_labels = np.where(seg_labels>0, 1, 0)
        seg_labels = seg_labels.astype(int)
        pred = pred.astype(int)
        metrics = utils.accuracy_pixel_level(pred, seg_labels)
        pixel_accu, iou = metrics[0], metrics[1]

        # compute loss
        loss = criterion(output, labels)

        result = [loss.item(), pixel_accu, iou]
        results.update(result, 1)

        yHaT = pred
        yval = seg_labels
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            logger.info('\tIteration: [{:d}/{:d}]'
                        '\tLoss: {r[0]:.4f}'
                        '\tPixel_Accu {r[1]:.4f}'
                        '\tIoU {r[2]:.4f}'.format(batch_idx, len(train_loader), r=results.avg))

    # ===================log========================
    logger.info('\t=> Train Epoch [{:d}/{:d}]'
                '\tAvg Loss: {r[0]:.4f}'
                '\tPixel_Accu {r[1]:.4f}'
                '\tIoU {r[2]:.4f}'.format(epoch, args.epochs, r=results.avg))

def validate(val_loader, model, criterion):
    results = utils.AverageMeter(3)

    for batch_idx, (imgs, labels, *rest) in enumerate(val_loader):
        # if isinstance(rest[0][0], str):
        #     image_filename = rest[0][0]
        # else:
        #     image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

        imgs = Variable(imgs.to(device=args.device))
        labels = Variable(labels.to(device=args.device))
        # start = timeit.default_timer()
        output = model(imgs)
        # stop = timeit.default_timer()
        # print('Time: ', stop - start)

        # measure accuracy and record loss
        seg_labels = labels.detach().cpu().numpy()
        pred = np.argmax(output.data.cpu().numpy(), axis=1)
        seg_labels = np.where(seg_labels > 0, 1, 0)
        seg_labels = seg_labels.astype(int)
        pred = pred.astype(int)
        metrics = utils.accuracy_pixel_level(pred, seg_labels)
        pixel_accu, iou = metrics[0], metrics[1]

        loss = criterion(output, labels)

        result = [loss.item(), pixel_accu, iou]
        results.update(result, 1)

        # print(np.unique(tmp2))
        yHaT = pred
        yval = seg_labels

        epsilon = 1e-20

        del imgs, labels, pred, seg_labels, output

        yHaT[yHaT == 1] = 255
        yval[yval == 1] = 255
        # cv2.imwrite(fulldir+image_filename, yHaT[0,1,:,:])
    # ===================log========================
    logger.info('\t=> Val Avg: Loss: {r[0]:.4f}'
                '\tPixel_Accu {r[1]:.4f}'
                '\tIoU {r[2]:.4f}'.format(r=results.avg))


if __name__ == '__main__':
    main()

            


