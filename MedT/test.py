import argparse
import lib
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
from torchvision import transforms as T
from skimage.segmentation import watershed
import torch.nn.functional as F
import math
import cv2
from my_utils import gland_accuracy_object_level_all_images, gland_accuracy_object_level, \
    accuracy_pixel_level, draw_rand_inst_overlay
from scipy import ndimage
import skimage.morphology as morph
from scipy.ndimage.morphology import binary_fill_holes
from skimage import measure, io
from metrics import ObjectHausdorff, ObjectDice, ObjectF1score, Dice


parser = argparse.ArgumentParser(description='Test MedT')
parser.add_argument('--cuda', default="on", type=str, help='switch on/off cuda option (default: off)')
parser.add_argument('--modelname', default='swinUnet', type=str, help='type of model')

parser.add_argument('--save_dir', type=str, default='./experiments/')
parser.add_argument('--mask_save_dir', type=str, default='./experiments/GlaS/Image_segmentation/')
parser.add_argument('--overlay_save_dir', type=str, default='./experiments/GlaS/overlay/')
parser.add_argument('--img_dir', type=str, default='/home/data2/MedImg/GlandSeg/GlaS/test/Images')
parser.add_argument('--label_dir', type=str, default='/home/data2/MedImg/GlandSeg/GlaS/test/Annotation')
parser.add_argument('--dataset', type=str, choices=['GlaS', 'CRAG'], default='GlaS', help='which dataset be used')


parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--loaddirec', default='/home/data1/wzh/code/GlandSegBenchmarks/MedT/experiments/GlaS/500/swinUnet.pth', type=str)
parser.add_argument('--imgsize', type=int, default=448)
parser.add_argument('--gray', default='no', type=str)
parser.add_argument('--device', default='cuda:3', type=str)
parser.add_argument('--gpu', type=list, default=[3], help='GPUs for training')
parser.add_argument('--aug', default='off', type=str, help='turn on img augmentation (default: False)')

# 后处理参数
parser.add_argument('--min_area', type=int, default=100, help='minimum area for an object')
parser.add_argument('--radius', type=int, default=4)
args = parser.parse_args()


gray_ = args.gray
aug = args.aug
modelname = args.modelname
imgsize = args.imgsize
loaddirec = args.loaddirec

def main():
    global eval_flag, transforms, save_flag
    eval_flag = True
    save_flag = True

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

    # ----- define augmentation ----- #
    transforms = T.ToTensor()

    # ----- load data ----- #
    tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
    # val_dataset = ImageToImage2D(args.img_dir, tf_val)
    val_dataset = Image2D(args.img_dir)
    valloader = DataLoader(val_dataset, 1, shuffle=True)

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
    elif modelname == "swinUnet":
        model = lib.models.swin_unet.SwinUnet(img_size=imgsize, num_classes=2)

    if len(args.gpu) > 1:
        print("Let's use", len(args.gpu), "GPUs!")
        model = nn.DataParallel(model, device_ids=args.gpu)
    model.to(args.device)

    # ----- load trained model ----- #
    model.load_state_dict(torch.load(args.loaddirec))
    print("=> Load trained model")
    # model.eval()

    save_dir = "%s/%s/" % (args.save_dir, args.dataset)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if not os.path.exists(args.mask_save_dir):
        os.mkdir(args.mask_save_dir)
    if not os.path.exists(args.overlay_save_dir):
        os.mkdir(args.overlay_save_dir)

    with torch.no_grad():
        validate(valloader, model, save_dir)


def proc(inst_raw):
    ksize = 3
    k_disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

    inst_mask = np.array(inst_raw > 0.5)
    #cv2.imwrite('test_mask_ori.jpg', 255 * np.uint8(inst_mask))
    if np.sum(inst_mask) > 0:
        inst_mask = cv2.erode(inst_mask.astype('uint8'), k_disk, iterations=1)

        dist = cv2.distanceTransform(inst_mask, cv2.DIST_L2, 3)
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        # cv2.imwrite('test_dist.jpg', np.uint8(dist * 255))

        inst_mask = ndimage.measurements.label(inst_mask)[0]
        inst_mask = morph.remove_small_objects(inst_mask, min_size=4)
        mask = np.array(inst_mask > 0)

        _, inter_mask = cv2.threshold(dist, 0.7, 1, 0)
        inst_mrk = ndimage.measurements.label(inter_mask)[0]
        inst_mrk = morph.remove_small_objects(inst_mrk, min_size=4)
        marker = binary_fill_holes(inst_mrk)
        marker = ndimage.measurements.label(marker)[0]

        output_map = watershed(-dist, marker, mask=mask)
    else:
        output_map = np.zeros([inst_mask.shape[0], inst_mask.shape[1]])
    #cv2.imwrite('test_mask.jpg', 255 * mask)
    #cv2.imwrite('test_marker.jpg', 20 * marker)
    #cv2.imwrite('test_watershed.jpg', 20 * output_map)
    return output_map


def validate(val_loader, model, save_dir):
    accumulated_metrics = np.zeros(11)
    all_results = dict()

    images_list = os.listdir(args.img_dir)
    for batch_idx, image_filename in enumerate(images_list):
        stride = args.imgsize
        print('=> Processing image {:s}'.format(image_filename))

        image = cv2.imread(os.path.join(args.img_dir, image_filename))
        ## ------------ 切分为imgsize x imgsize的patch，如果大小不足则padding ---------- ##
        h, w, _ = image.shape
        padding_h =  int(math.ceil(h / stride) * stride)
        padding_w = int(math.ceil(w / stride) * stride)
        padding_output = np.zeros((1, 2, padding_h, padding_w))
        padding_weight_mask = np.zeros_like(padding_output)

        name = image_filename.split('.')[0]
        if args.dataset == 'GlaS':
            label_path = '{:s}/{:s}_anno.bmp'.format(args.label_dir, name)
        ## CRAG
        elif args.dataset == 'CRAG':
            label_path = '{:s}/{:s}.png'.format(args.label_dir, name)
        label_img = io.imread(label_path)

        for h_ in range(0, padding_h - stride + 1, int(stride // 4)):
            for w_ in range(0, padding_w - stride + 1, int(stride // 4)):
                img_padding = np.zeros((args.imgsize, args.imgsize, 3), np.uint8)
                slice = image[h_:h_ + args.imgsize, w_:w_ + args.imgsize, :]
                t_h, t_w, _ = slice.shape
                img_padding[:t_h, :t_w] = slice

                img = transforms(img_padding).unsqueeze(0)
                img = Variable(img.to(device=args.device))
                output = model(img)

                output = F.softmax(output, dim=1).cpu().numpy()
                weight_mask = np.ones_like(output)

                padding_output[:, :, h_:h_ + args.imgsize, w_:w_ + args.imgsize] += output
                padding_weight_mask[:, :, h_:h_ + args.imgsize, w_:w_ + args.imgsize] += weight_mask
                del output, img, img_padding, slice

        padding_output /= padding_weight_mask
        output = padding_output[:, :, :h, :w]
        pred = np.argmax(output, axis=1)
        pred = pred[0].astype(int)

        ## calculate performance
        pred_inside = pred == 1
        pred2 = ndimage.binary_fill_holes(pred_inside)
        pred_labeled = measure.label(pred2)
        pred_labeled = morph.remove_small_objects(pred_labeled, args.min_area)

        ## proc 后处理
        # pred_labeled = proc(output[0, 1, :, :])

        print('\tComputing metrics...')
        result = accuracy_pixel_level(np.expand_dims(pred_labeled > 0, 0), np.expand_dims(label_img > 0, 0))
        pixel_accu = result[0]

        # single_image_result = gland_accuracy_object_level(pred_labeled, label_img)
        objF1, _, _, _ = ObjectF1score(pred_labeled, label_img)
        objDice = ObjectDice(pred_labeled, label_img)
        dice = Dice(np.where(pred_labeled>0, 1, 0), np.where(label_img>0, 1, 0))
        objHaus = ObjectHausdorff(pred_labeled, label_img)
        single_image_result = (objF1, objDice, dice, objHaus)
        accumulated_metrics += gland_accuracy_object_level_all_images(pred_labeled, label_img)
        all_results[name] = tuple([pixel_accu, *single_image_result])

        # save image
        if save_flag:
            print('\tSaving image results...')
            ## draw semantic mask
            final_pred = pred_labeled.astype(np.uint8) * 100
            cv2.imwrite('{:s}/{:s}_seg.jpg'.format(args.mask_save_dir, name), final_pred)

            ## draw overlay
            overlay = draw_rand_inst_overlay(image, pred_labeled)
            cv2.imwrite('{:s}/{:s}_seg.jpg'.format(args.overlay_save_dir, name), overlay)


        # 打印每张test的指标
        print('Pixel Acc: {r[0]:.4f}\n'
              'F1: {r[1]:.4f}\n'
              'ObjDice: {r[2]:.4f}\n'
              'Dice: {r[3]:.4f}\n'
              'haus: {r[4]:.4f}'.format(r=[pixel_accu, objF1, objDice, dice, objHaus]))

    avg_pq = []
    avg_f1 = []
    avg_objDice = []
    avg_dice = []
    avg_haus = []
    for name in all_results:
        pq, f1, objDice, dice, haus = all_results[name]
        avg_pq += [pq]
        avg_f1 += [f1]
        avg_objDice += [objDice]
        avg_dice += [dice]
        avg_haus += [haus]
    avg_pq = np.nanmean(avg_pq)
    avg_f1 = np.nanmean(avg_f1)
    avg_objDice = np.nanmean(avg_objDice)
    avg_dice = np.nanmean(avg_dice)
    avg_haus = np.nanmean(avg_haus)
    header = ['pixel_acc', 'objF1', 'objDice', 'Dice', 'objHaus']
    save_results(header, [avg_pq, avg_f1, avg_objDice, avg_dice, avg_haus], all_results,
                 '{:s}/test_result.txt'.format(save_dir))

    print('Average Dice: {:.4f}'.format(avg_dice))

    TP, FP, FN, dice_g, dice_s, iou_g, iou_s, hausdorff_g, hausdorff_s, \
    gt_objs_area, pred_objs_area = accumulated_metrics

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F1 = 2 * TP / (2 * TP + FP + FN)
    dice = (dice_g / gt_objs_area + dice_s / pred_objs_area) / 2
    iou = (iou_g / gt_objs_area + iou_s / pred_objs_area) / 2
    haus = (hausdorff_g / gt_objs_area + hausdorff_s / pred_objs_area) / 2

    avg_pixel_accu = -1
    avg_results = [avg_pixel_accu, recall, precision, F1, dice, iou, haus]

    print('=> Processed all {:d} images'.format(len(val_loader)))
    if eval_flag:
        print('Average of all images:\n'
              'recall: {r[1]:.4f}\n'
              'precision: {r[2]:.4f}\n'
              'F1: {r[3]:.4f}\n'
              'dice: {r[4]:.4f}\n'
              'iou: {r[5]:.4f}\n'
              'haus: {r[6]:.4f}'.format(r=avg_results))

        strs = args.img_dir.split('/')
        header = ['pixel_acc', 'recall', 'precision', 'F1', 'Dice', 'IoU', 'Hausdorff']
        save_results(header, avg_results, all_results,
                     '{:s}/{:s}_test_result_ck.txt'.format(save_dir, strs[-1]))


def save_results(header, avg_results, all_results, filename, mode='w'):
    """ Save the result of metrics
        results: a list of numbers
    """
    N = len(header)
    assert N == len(avg_results)
    with open(filename, mode) as file:
        # header
        file.write('Metrics:\t')
        for i in range(N - 1):
            file.write('{:s}\t'.format(header[i]))
        file.write('{:s}\n'.format(header[N - 1]))

        # average results
        file.write('Average:\t')
        for i in range(N - 1):
            file.write('{:.4f}\t'.format(avg_results[i]))
        file.write('{:.4f}\n'.format(avg_results[N - 1]))
        file.write('\n')

        # all results
        for key, values in sorted(all_results.items()):
            file.write('{:s}:'.format(key))
            for value in values:
                file.write('\t{:.4f}'.format(value))
            file.write('\n')



if __name__ == '__main__':
    main()



