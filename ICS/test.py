import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import skimage.morphology as morph
from skimage import measure, io
from models.ics import EfficientNet
import utils
from metrics import ObjectHausdorff, ObjectDice, ObjectF1score
# from vis import draw_overlay_rand
import torchvision.transforms as transforms
import cv2
import argparse
from scipy import ndimage
import glob
import re

parser = argparse.ArgumentParser(description="Test I2CS Model")
parser.add_argument('--num_workers', type=int, default=2, help='number of workers to load images')
parser.add_argument('--checkpoint', type=str, default=None, help='start from checkpoint')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--save_dir', type=str, default='./experiments')
parser.add_argument('--img_dir', type=str, default='/home/data2/MedImg/GlandSeg/GlaS/test/Images')
parser.add_argument('--label_dir', type=str, default='/home/data2/MedImg/GlandSeg/GlaS/test/Annotation')
parser.add_argument('--model_path', type=str, default='/home/data1/wzh/code/GlandSegBenchmarks/ICS/experiments/GlaS/150/checkpoints/checkpoint_150.pth.tar')
parser.add_argument('--dataset', type=str, choices=['GlaS', 'CRAG'], default='GlaS', help='which dataset be used')
parser.add_argument('--gpu', type=list, default=[2,], help='GPUs for training')

# 后处理参数
parser.add_argument('--min_area', type=int, default=100, help='minimum area for an object')
parser.add_argument('--radius', type=int, default=4)
args = parser.parse_args()

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)

    img_dir = args.img_dir
    label_dir = args.label_dir
    save_dir = "%s/%s" % (args.save_dir, args.dataset)
    model_path = args.model_path
    save_flag = True
    tta = False

    # check if it is needed to compute accuracies
    eval_flag = True if label_dir else False

    # data transforms
    test_transform = transforms.Compose([transforms.ToTensor()])

    # load model
    model = EfficientNet(compound_coef=1)
    model = model.cuda()
    cudnn.benchmark = True

    # ----- load trained model ----- #
    print("=> loading trained model")
    best_checkpoint = torch.load(model_path)

    model.load_state_dict(best_checkpoint['state_dict'])
    epoch = best_checkpoint['epoch']
    print("=> loaded model at epoch {}".format(best_checkpoint['epoch']))

    # switch to evaluate mode
    model.eval()
    counter = 0
    print("=> Test begins:")

    img_names = os.listdir(img_dir)

    # TP, FP, FN, dice_g, dice_s, iou_g, iou_s, haus_g, haus_s, gt_area, seg_area
    accumulated_metrics = np.zeros(11)
    all_results = dict()

    if save_flag:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        strs = img_dir.split('/')
        prob_maps_folder = '{:s}/{:s}/'.format(args.save_dir, args.dataset, strs[-1])
        seg_folder = '{:s}/{:s}/{:s}_segmentation'.format(args.save_dir, args.dataset, strs[-1])

        if not os.path.exists(prob_maps_folder):
            os.mkdir(prob_maps_folder)
        if not os.path.exists(seg_folder):
            os.mkdir(seg_folder)

    for img_name in img_names:
        print('=> Processing image {:s}'.format(img_name))
        img_path = '{:s}/{:s}'.format(img_dir, img_name)
        img = Image.open(img_path)

        name = img_name.split('.')[0].split('_')[-1]

        ########################### esenbling from multi-scale #################################
        h, w = img.size
        ensemble_cam = np.zeros((2, w, h))

        ## 从小图进行推理，然后
        for scale in [1, 1.25, 1.5, 1.75, 2]:
            image_per_scale_path = '/home/data1/my/Project/GlandSegBenchmark/OEEM/classification/glas_valid/crop_images_320_256/' + \
                                   name + \
                                   '/' + str(scale)
            scale = float(scale)
            w_ = int(w * scale)
            h_ = int(h * scale)
            side_length = 320
            interpolatex = side_length
            interpolatey = side_length
            if w_ < side_length:
                interpolatex = w_
            if h_ < side_length:
                interpolatey = h_

            with torch.no_grad():
                cam_list = []
                position_list = []
                for image_path in glob.glob(os.path.join(image_per_scale_path, '*.png')):
                    img = Image.open(image_path)
                    img = test_transform(img).unsqueeze(0).cuda()
                    positions = os.path.basename(image_path)
                    positions = list(map(lambda x: int(x), re.findall(r'\d+', positions)))
                    cam_scores, _ = model(img)
                    cam_scores = F.interpolate(cam_scores.detach(), (interpolatex, interpolatey), mode='bilinear',
                                               align_corners=False)
                    cam_scores = F.softmax(cam_scores, dim=1).cpu().numpy()
                    cam_list.append(cam_scores)
                    position_list.append(np.array(positions).reshape(1, -1))
                cam_list = np.concatenate(cam_list)
                position_list = np.concatenate(position_list)
                sum_cam = np.zeros((2, w_, h_))
                sum_counter = np.zeros_like(sum_cam)

                for k in range(cam_list.shape[0]):
                    y, x = position_list[k][0], position_list[k][1]
                    crop = cam_list[k]
                    sum_cam[:, y:y + side_length, x:x + side_length] += crop
                    sum_counter[:, y:y + side_length, x:x + side_length] += 1
                sum_counter[sum_counter < 1] = 1
                norm_cam = sum_cam / sum_counter
                norm_cam = F.interpolate(torch.unsqueeze(torch.tensor(norm_cam), 0), (w, h), mode='bilinear',
                                         align_corners=False).detach().cpu().numpy()[0]

                # Use the image-level label to eliminate impossible pixel classes
                ensemble_cam += norm_cam


        if eval_flag:
            ## GlaS
            label_path = '{:s}/{:s}_anno.bmp'.format(label_dir, name)
            ## CRAG
            # label_path = '{:s}/{:s}.png'.format(label_dir, name)
            label_img = io.imread(label_path)

        # input = test_transform(img).unsqueeze(0).cuda()

        # print('\tComputing output probability maps...')
        # prob_maps = get_probmaps(input, model)


        pred = np.argmax(ensemble_cam, axis=0)
        pred_inside = pred == 1
        pred2 = morph.remove_small_objects(pred_inside, args.min_area)  # remove small object


        pred2 = ndimage.binary_fill_holes(pred2)
        pred_labeled = measure.label(pred2)   # connected component labeling

        if eval_flag:
            print('\tComputing metrics...')
            result = utils.accuracy_pixel_level(np.expand_dims(pred_labeled > 0, 0), np.expand_dims(label_img > 0, 0))
            pixel_accu = result[0]

            # single_image_result = utils.gland_accuracy_object_level(pred_labeled, label_img)
            objF1, _, _, _ = ObjectF1score(pred_labeled, label_img)
            objDice = ObjectDice(pred_labeled, label_img)
            objHaus = ObjectHausdorff(pred_labeled, label_img)
            single_image_result = (objF1, objDice, objHaus)
            accumulated_metrics += utils.gland_accuracy_object_level_all_images(pred_labeled, label_img)
            all_results[name] = tuple([pixel_accu, *single_image_result])

            # 打印每张test的指标
            print('Pixel Acc: {r[0]:.4f}\n'
                  'F1: {r[1]:.4f}\n'
                  'dice: {r[2]:.4f}\n'
                  'haus: {r[3]:.4f}'.format(r=[pixel_accu, objF1, objDice, objHaus]))

        # save image
        if save_flag:
            print('\tSaving image results...')
            final_pred = pred_labeled.astype(np.uint8) * 100
            cv2.imwrite('{:s}/{:s}_seg.jpg'.format(seg_folder, name), final_pred)

        counter += 1
        if counter % 10 == 0:
            print('\tProcessed {:d} images'.format(counter))

    avg_pq = []
    avg_f1 = []
    avg_dice = []
    avg_haus = []
    for name in all_results:
        pq, f1, dice, haus = all_results[name]
        avg_pq += [pq]
        avg_f1 += [f1]
        avg_dice += [dice]
        avg_haus += [haus]
    avg_pq = np.nanmean(avg_pq)
    avg_f1 = np.nanmean(avg_f1)
    avg_dice = np.nanmean(avg_dice)
    avg_haus = np.nanmean(avg_haus)
    header = ['pixel_acc', 'objF1', 'objDice', 'objHaus']
    save_results(header, [avg_pq, avg_f1, avg_dice, avg_haus], all_results,
                 '{:s}/test_result_epoch{}.txt'.format(save_dir, epoch))

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

    print('=> Processed all {:d} images'.format(counter))
    if eval_flag:
        print('Average of all images:\n'
              'recall: {r[1]:.4f}\n'
              'precision: {r[2]:.4f}\n'
              'F1: {r[3]:.4f}\n'
              'dice: {r[4]:.4f}\n'
              'iou: {r[5]:.4f}\n'
              'haus: {r[6]:.4f}'.format(r=avg_results))

        strs = img_dir.split('/')
        header = ['pixel_acc','recall', 'precision', 'F1', 'Dice', 'IoU', 'Hausdorff']
        save_results(header, avg_results, all_results,
                     '{:s}/{:s}_test_result_ck_epoch{}.txt'.format(save_dir, strs[-1], epoch))


def get_probmaps(input, model):
    with torch.no_grad():
        output, feature_maps = model(input)

    prob_maps = F.softmax(output, dim=1).cpu().numpy()
    return prob_maps[:, 1, :, :][0]


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
