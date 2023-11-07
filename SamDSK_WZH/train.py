# Code for MedT

import argparse
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from dataloader import DataFolder
import numpy as np
import torch
from my_transforms import get_transforms
from loss import dice_loss, object_dice_losses, cross_entropy_loss
import utils
from utils import accuracy_pixel_level
from models.swin_unet import SwinUnet
import logging

parser = argparse.ArgumentParser(description='MedT')
parser.add_argument('--num_workers', default=4, type=int, help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=500, type=int, help='number of total epochs to run(default: 400)')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=4, type=int, help='batch size (default: 1)')
## GlaS 1e-4，这个非常重要，学习率轻微调整都对结果有很大影响, CRAG 5e-5；
parser.add_argument('--learning_rate', default=1e-4, type=float, help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float, help='weight decay (default: 1e-5)')
parser.add_argument('--save_freq', type=int,default = 50)
parser.add_argument('--dataset', type=str, choices=['GlaS', 'CRAG'], default='GlaS', help='which dataset be used')

parser.add_argument('--last_model_path', type=str, default="../MedT/swin_tiny_patch4_window7_224.pth")
                    #default="/home/data1/wzh/code/GlandSegBenchmarks/SamDSK_WZH/experiments/GlaS_10labeled_round1/500/swinUnet.pth")

parser.add_argument('--modelname', default='swinUnet', type=str, help='type of model')
parser.add_argument('--cuda', default="on", type=str, help='switch on/off cuda option (default: on)')
parser.add_argument('--aug', default='off', type=str, help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='default', type=str, help='load a pretrained model')
parser.add_argument('--save_dir', type=str, default='experiments/')
parser.add_argument('--refine_dir', type=str, default='experiments/GlaS_10labeled_round1/refine_inst_pred/')
parser.add_argument('--crop', type=int, default=448)
parser.add_argument('--imgsize', type=int, default=448)
parser.add_argument('--gray', default='no', type=str)
parser.add_argument('--device', default='cuda:3', type=str)
parser.add_argument('--gpu', type=list, default=[3], help='GPUs for training')
parser.add_argument('--round', type=int, default=2, help='number of round for self-training process')

parser.add_argument('--gamma1', type=float, default=1, help='weight for dice loss')
parser.add_argument('--gamma2', type=float, default=0.5, help='weight for object-level dice loss')
args = parser.parse_args()

gray_ = args.gray
aug = args.aug
modelname = args.modelname
imgsize = args.imgsize
torch.cuda.set_device(args.device)

def main():
    global logger

    args.save_dir = "%s/%s_10labeled_round%s/" % (args.save_dir, args.dataset, args.round)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.crop is not None:
        crop = (args.crop, args.crop)
    else:
        crop = None

    # set up logger
    logger, logger_results = setup_logging()


    # ---------------- load data -------------------- #
    # if args.dataset == 'GlaS':
    #     img_dir = '/home/data2/MedImg/GlandSeg/%s/wzh/train/480x480/' % (args.dataset)
    # elif args.dataset == 'CRAG':
    #     img_dir = '/home/data2/MedImg/GlandSeg/%s/wzh/train/480x480/' % (args.dataset)
    #
    # tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=(0.1, 0.1, 0.1, 0.1), long_mask=True)
    # tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
    # train_dataset = ImageToImage2D(img_dir, tf_train)
    # val_dataset = ImageToImage2D(img_dir, tf_val)
    #
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, 1, shuffle=True)
    # ---------------------------------------------- #
    # ----- define augmentation ----- #
    data_transforms = {
        'train': get_transforms({
        'random_crop' : args.imgsize,
        'horizontal_flip': True,
        'vertical_flip': True,
        'random_rotation': 90,
        'to_tensor': 1,
    }),
        'val': get_transforms({
        'random_crop': args.imgsize,
        'to_tensor': 1,
    })}

    dsets = {}
    for x in ['train', 'val']:
        # img_dir = '/home/data2/MedImg/GlandSeg/%s/wzh/train/480x480/Images/' % (args.dataset)
        # target_dir = '/home/data2/MedImg/GlandSeg/%s/wzh/train/480x480/Annotation/' % (args.dataset)
        ## GlaS
        img_dir = '/home/data2/MedImg/GlandSeg/%s/train/Images' % (args.dataset)
        target_dir = '/home/data2/MedImg/GlandSeg/%s/train/Annotation' % (args.dataset)
        ## CRAG
        # img_dir = '/home/data2/MedImg/GlandSeg/%s/train/TrainSet/Images' % (args.dataset)
        # target_dir = '/home/data2/MedImg/GlandSeg/%s/train/TrainSet/Annotation' % (args.dataset)
        dir_list = [img_dir, target_dir]
        # post_fix = ['weight.png', 'label.png']
        if x == 'train':
            dsets[x] = DataFolder(dir_list, args.refine_dir, data_transform=data_transforms[x],
                                  dataset = args.dataset, semi_rate=0.1, round=args.round, mode = 'train')
        else:
            dsets[x] = DataFolder(dir_list, args.refine_dir, data_transform=data_transforms[x],
                                  dataset = args.dataset, mode = 'val')
    train_loader = DataLoader(dsets['train'], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(dsets['val'], batch_size=1, shuffle=False,
                            num_workers=args.num_workers)
    ########################################################################

    # ----- create model ----- #
    device = torch.device(args.device)
    model = SwinUnet(img_size = imgsize, num_classes = 2)
    logger.info("=> Creating {} model!".format(modelname))

    # ------ 加载预训练权重 ------ #
    # model.load_state_dict(torch.load("swin_tiny_patch4_window7_224.pth"))
    model.load_from(args.device, logger, args.last_model_path)
    logger.info("=> Loaded pretrained model weight!")

    if len(args.gpu) > 1:
        print("Let's use", len(args.gpu), "GPUs!")
        model = nn.DataParallel(model, device_ids=args.gpu)
    model.to(device)

    # criterion = LogNLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=1e-4)
    ## GlaS 学习率调整策略
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400], gamma=0.5)

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
        # if epoch == 10:
        #     for param in model.parameters():
        #         param.requires_grad =True
        if ((epoch + 1) % args.save_freq) ==0:
            with torch.no_grad():
                validate(val_loader, model, criterion)

            save_dir = "%s/%d/" % (args.save_dir, epoch + 1)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(model.state_dict(), save_dir + args.modelname + ".pth")
        scheduler.step()


def setup_logging():
    mode = 'w'
    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train.log'.format(args.save_dir, mode=mode))
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
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(args.save_dir), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(args.save_dir))
    if mode == 'w':
        logger_results.info('epoch\ttrain_loss\ttrain_loss_CE\ttrain_loss_var\ttrain_acc\ttrain_iou\t'
                            'val_loss\tval_acc\tval_iou')

    return logger, logger_results

def train(train_loader, model, optimizer, criterion, epoch):
    epoch_running_loss = 0
    results = utils.AverageMeter(6)
    for batch_idx, (imgs, labels, *rest) in enumerate(train_loader):
        imgs = Variable(imgs.to(device=args.device))
        test = labels.numpy()
        # if np.min(test) == -1:
        #     print('ERROR')
        labels = Variable(labels.to(device=args.device))

        ## 将不确定区域，label为-1的像素都置为0，不参与loss计算
        mask = torch.eq(labels, -1).int().type(torch.LongTensor).to(args.device)
        mask = 1 - mask
        seg_labels = torch.gt(labels, 0).int().type(torch.LongTensor)
        seg_labels = seg_labels.to(device=args.device)

        # ===================forward=====================
        output = model(imgs)

        # measure accuracy and record loss
        pred = np.argmax(output.data.cpu().numpy(), axis=1)
        pred = pred.astype(int)
        metrics = accuracy_pixel_level(pred, seg_labels.detach().cpu().numpy())
        pixel_accu, iou = metrics[0], metrics[1]

        # compute dice loss
        score = F.softmax(output, dim=1)
        loss_dice = dice_loss(seg_labels, score[:, 1, :, :], mask)
        # compute object-level dice loss
        loss_obj_dice = object_dice_losses(labels, score[:, 1, :, :], mask)
        # compute ce loss
        #loss_ce = criterion(output, seg_labels)
        loss_ce = cross_entropy_loss(seg_labels, score, mask)
        loss = loss_ce + args.gamma1 * loss_dice + args.gamma2 * loss_obj_dice

        result = [loss.item(),loss_ce.item(), loss_dice.item(), loss_obj_dice.item(), pixel_accu, iou]
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
                        '\tLoss_CE {r[1]:.4f}'
                        '\tLoss_Dice {r[2]:.4f}'
                        '\tLoss_Obj_Dice {r[3]:.4f}'
                        '\tPixel_Accu {r[4]:.4f}'
                        '\tIoU {r[5]:.4f}'.format(batch_idx, len(train_loader), r=results.avg))

    # ===================log========================
    logger.info('\t=> Train Epoch [{:d}/{:d}]'
                '\tAvg Loss: {r[0]:.4f}'
                '\tLoss_CE {r[1]:.4f}'
                '\tLoss_Dice {r[2]:.4f}'
                '\tLoss_Obj_Dice {r[3]:.4f}'
                '\tPixel_Accu {r[4]:.4f}'
                '\tIoU {r[5]:.4f}'.format(epoch, args.epochs, r=results.avg))

def validate(val_loader, model, criterion):
    results = utils.AverageMeter(6)

    for batch_idx, (imgs, labels, *rest) in enumerate(val_loader):
        # if isinstance(rest[0][0], str):
        #     image_filename = rest[0][0]
        # else:
        #     image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

        imgs = Variable(imgs.to(device=args.device))
        labels = Variable(labels.to(device=args.device))
        mask = torch.eq(labels, -1).int().type(torch.LongTensor).to(args.device)
        mask = 1 - mask
        seg_labels = torch.gt(labels, 0).int().type(torch.LongTensor)
        seg_labels = seg_labels.to(device=args.device)
        # start = timeit.default_timer()
        output = model(imgs)
        # stop = timeit.default_timer()
        # print('Time: ', stop - start)

        # measure accuracy and record loss
        pred = np.argmax(output.data.cpu().numpy(), axis=1)
        pred = pred.astype(int)
        metrics = accuracy_pixel_level(pred, seg_labels.detach().cpu().numpy())
        pixel_accu, iou = metrics[0], metrics[1]

        # compute dice loss
        score = F.softmax(output, dim=1)
        loss_dice = dice_loss(seg_labels, score[:, 1, :, :], mask)
        # compute object-level dice loss
        loss_obj_dice = object_dice_losses(labels, score[:, 1, :, :], mask)
        # compute ce loss
        #loss_ce = criterion(output, seg_labels)
        loss_ce = cross_entropy_loss(seg_labels, score, mask)
        loss = loss_ce + args.gamma1 * loss_dice + args.gamma2 * loss_obj_dice

        result = [loss.item(), loss_ce.item(), loss_dice.item(), loss_obj_dice.item(), pixel_accu, iou]
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
                '\tLoss_CE {r[1]:.4f}'
                '\tLoss_Dice {r[2]:.4f}'
                '\tLoss_Obj_Dice {r[3]:.4f}'
                '\tPixel_Accu {r[4]:.4f}'
                '\tIoU {r[5]:.4f}'.format(r=results.avg))


if __name__ == '__main__':
    main()

            


