import glob
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import skimage.morphology as morph
from skimage.measure import label
from skimage import measure, io
from model.pspnet import PSPNet
import utils
from metrics import dice_coefficient, iou_metrics
# from vis import draw_overlay_rand
from scipy import ndimage
import torchvision.transforms as transforms
import argparse
from multiprocessing import Array, Process
import random
import cv2
import re

parser = argparse.ArgumentParser(description="Testing oeem segmentation Model")
parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers to load images')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--checkpoint', type=str, default=None, help='start from checkpoint')
parser.add_argument('--checkpoint_freq', type=int, default=10, help='epoch to save checkpoints')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--save_dir', type=str, default='./experiments')
parser.add_argument('--img_dir', type=str, default='/home/data1/my/Project/GlandSegBenchmark/OEEM/classification/glas_cls/2.validation/img')
parser.add_argument('--label_dir', type=str, default='/home/data1/my/Project/GlandSegBenchmark/OEEM/classification/glas_cls/2.validation/mask')

# parser.add_argument('--img_dir', type=str, default='/home/data2/MedImg/GlandSeg/GlaS/wzh/valid/480x480/Images/')
# parser.add_argument('--label_dir', type=str, default='/home/data2/MedImg/GlandSeg/GlaS/wzh/valid/480x480/Annotation/')
parser.add_argument('--desc', type=str, choices=['CELoss', 'OEEMLoss'],
                    default='OEEMLoss_0919',
                    # default='CELoss_0915-2'
                    )

parser.add_argument('--model_path', type=str,
                    default="/home/data1/my/Project/GlandSegBenchmark/OEEM/my_segmentation/experiments/GlaS_OEEMLoss_0919/checkpoints/checkpoint_best.pth.tar")
                    # default='/home/data1/my/Project/GlandSegBenchmark/OEEM/my_segmentation/experiments/GlaS_CELoss_0915-2/checkpoints/checkpoint_best.pth.tar')
                    # default="/home/data1/my/Project/GlandSegBenchmark/OEEM/my_segmentation/experiments/GlaS_CELoss_0915/checkpoints/checkpoint_100.pth.tar")
parser.add_argument('--dataset', type=str, choices=['GlaS', 'CRAG'], default='GlaS', help='which dataset be used')
parser.add_argument('--gpu', type=list, default=[2, ], help='GPUs for training')

# 后处理参数
parser.add_argument('--min_area', type=int, default=400, help='minimum area for an object')
parser.add_argument('--radius', type=int, default=4)
args = parser.parse_args()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)

    img_dir = args.img_dir
    label_dir = args.label_dir
    save_dir = "%s/%s_%s" % (args.save_dir, args.dataset, args.desc)
    # model_path = args.model_path
    save_flag = True

    # check if it is needed to compute accuracies
    eval_flag = True if label_dir else False

    # data transforms
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.787, 0.511, 0.785],
                                                              std=[0.167, 0.248, 0.131])])

    # load model
    model = PSPNet(classes=2)

    model = model.cuda()
    cudnn.benchmark = True

    for model_path in glob.glob('/home/data1/my/Project/GlandSegBenchmark/OEEM/my_segmentation/experiments/GlaS_OEEMLoss_0919/checkpoints/checkpoint_4*'):
        # ----- load trained model ----- #
        print("=> loading trained model")

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        # epoch = best_checkpoint['epoch']
        epoch = os.path.basename(model_path).split('.')[0].split('_')[-1]
        print("=> loaded model at epoch {}".format(epoch))

        # switch to evaluate mode
        model.eval()
        counter = 0
        print("=> Test begins:")

        img_names = os.listdir(img_dir)

        # TP, FP, FN, dice_g, dice_s, iou_g, iou_s, haus_g, haus_s, gt_area, seg_area

        all_results = dict()

        if save_flag:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            prob_maps_folder = '{:s}/{:s}_{:s}/{:s}'.format(args.save_dir, args.dataset, args.desc, 'mask_pred')
            if not os.path.exists(prob_maps_folder):
                os.mkdir(prob_maps_folder)
            vis_maps_folder = '{:s}/{:s}_{:s}/{:s}'.format(args.save_dir, args.dataset, args.desc, 'vis_pred')
            if not os.path.exists(vis_maps_folder):
                os.mkdir(vis_maps_folder)


        for img_name in img_names:
            # load test image
            print('=> Processing image {:s}'.format(img_name))
            img_path = '{:s}/{:s}'.format(img_dir, img_name)
            orig_img = Image.open(img_path)
            name = os.path.splitext(img_name)[0]

            ########################### esenbling from multi-scale #################################
            h, w = orig_img.size
            ensemble_cam = np.zeros((2, w, h))

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
                    for img_path in glob.glob(os.path.join(image_per_scale_path, '*.png')):
                        img = Image.open(img_path)
                        img = test_transform(img).unsqueeze(0).cuda()
                        positions = os.path.basename(img_path)
                        positions = list(map(lambda x: int(x), re.findall(r'\d+', positions)))
                        cam_scores = model(img)
                        cam_scores = F.interpolate(cam_scores, (interpolatex, interpolatey), mode='bilinear',
                                                   align_corners=False).detach().cpu().numpy()
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

            ########################################################################################
            pred = np.argmin(ensemble_cam, axis=0)

            ############################### post proc ################################################
            # 将二值图像转换为标签图像
            label_image = label(pred)
            # 根据标记删除小连通域
            filtered_label_image = morph.remove_small_objects(label_image, min_size=args.min_area)
            # 将标签图像转换回二值图像
            pred = (filtered_label_image > 0).astype(np.uint8)

            # smooth the contour prediction
            # pred = ndimage.binary_opening(pred, structure=morph.disk(5))
            # pred = ndimage.binary_closing(pred, structure=morph.disk(5))
            # pred = ndimage.binary_opening(pred, structure=morph.disk(1))
            # pred = ndimage.binary_closing(pred, structure=morph.disk(1))

            # fill holes
            pred = ndimage.binary_fill_holes(pred)
            ################################################################################################ # remove small object

            if eval_flag:
                # label_path = '{:s}/{:s}_anno.bmp'.format(label_dir, name)
                label_path = '{:s}/{:s}.png'.format(label_dir, name)
                label_img = io.imread(label_path)
                label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
                label_img = np.array(label_img == 76, dtype=np.uint8)
            # input = test_transform(img).unsqueeze(0)

            # print('\tComputing output probability maps...')
            # pred = get_predmaps(input, model)


            if eval_flag:
                img_show = np.concatenate([np.array(orig_img),
                                           np.stack((label_img * 255, label_img * 255, label_img * 255), axis=-1),
                                           np.stack((pred * 255, pred * 255, pred * 255), axis=-1)],
                                           axis=1)
                cv2.imwrite('{:s}/{}.png'.format(vis_maps_folder, name), img_show)

                np.save('{:s}/{}.npy'.format(prob_maps_folder, name), pred)
                print('\tComputing metrics...')
                result = utils.accuracy_pixel_level(np.expand_dims(pred > 0, 0), np.expand_dims(label_img > 0, 0))
                pixel_accu = result[0]

                # single_image_result = utils.gland_accuracy_object_level(pred_labeled, label_img)
                IoU = iou_metrics(pred, label_img)
                Dice = 2 * IoU / (1 + IoU)

                single_image_result = (IoU, Dice)
                all_results[name] = tuple([pixel_accu, *single_image_result])
                # 打印每张test的指标
                print('Pixel Acc: {r[0]:.4f}\n'
                      'IoU: {r[1]:.4f}\n'
                      'Dice: {r[2]:.4f}'.format(r=[pixel_accu, IoU, Dice]))

        over_all_iou, over_all_f1 = get_overall_valid_score('{:s}'.format(prob_maps_folder), args.label_dir)

        avg_pq = []
        avg_iou = []
        avg_dice = []

        for name in all_results:
            pq, iou, dice = all_results[name]
            avg_pq += [pq]
            avg_iou += [iou]
            avg_dice += [dice]

        avg_pq = np.nanmean(avg_pq)
        avg_iou = np.nanmean(avg_iou)
        avg_dice = np.nanmean(avg_dice)
        header = ['pixel_acc', 'Iou', 'Dice']
        save_results(header, [avg_pq, avg_iou, avg_dice], all_results,
                     f'{save_dir:s}/test_result_epoch{epoch}_overall_iou_{over_all_iou:.4f}_dice_{2 * over_all_iou / (1 + over_all_iou):.4f}_'
                     f'f1score_{over_all_f1:.4f}.txt')




def chunks(lst, num_workers=None, n=None):
    """
    a helper function for seperate the list to chunks

    Args:
        lst (list): the target list
        num_workers (int, optional): Default is None. When num_workers are not None, the function divide the list into num_workers chunks
        n (int, optional): Default is None. When the n is not None, the function divide the list into n length chunks

    Returns:
        llis: a list of small chunk lists
    """
    chunk_list = []
    if num_workers is None and n is None:
        print("the function should at least pass one positional argument")
        exit()
    elif n == None:
        n = int(np.ceil(len(lst)/num_workers))
        for i in range(0, len(lst), n):
            chunk_list.append(lst[i:i + n])
        return chunk_list
    else:
        for i in range(0, len(lst), n):
            chunk_list.append(lst[i:i + n])
        return chunk_list


def get_overall_valid_score(pred_image_path, groundtruth_path, num_workers=1, num_class=2):
    """
    get the scores with validation groundtruth, the background will be masked out
    and return the score for all photos

    Args:
        pred_image_path (str): the prediction require to test, npy format
        groundtruth_path (str): groundtruth images, png format
        num_workers (int): number of process in parallel, default is 5.
        mask_path (str): the white background, png format
        num_class (int): default is 2.

    Returns:
        float: the mIOU score
    """
    image_names = list(map(lambda x: x.split('.')[0], os.listdir(pred_image_path)))
    random.shuffle(image_names)
    image_list = chunks(image_names, num_workers)

    def f(intersection, union, image_list):
        gt_list = []
        pred_list = []

        for im_name in image_list:
            cam = np.load(os.path.join(pred_image_path, f"{im_name}.npy"), allow_pickle=True).astype(np.uint8).reshape(-1)
            # groundtruth = np.asarray(Image.open(groundtruth_path + f"/{im_name}_anno.bmp")).reshape(-1)
            # groundtruth = np.asarray(Image.open(groundtruth_path + f"/{im_name}.png")).reshape(-1)

            groundtruth = io.imread(groundtruth_path + f"/{im_name}.png")
            groundtruth = cv2.cvtColor(groundtruth, cv2.COLOR_BGR2GRAY)
            groundtruth = np.array(groundtruth == 76, dtype=np.uint8).reshape(-1)

            gt_list.extend(groundtruth)
            pred_list.extend(cam)

        pred = np.array(pred_list)
        real = np.array(gt_list)
        for i in range(num_class):
            if i in pred:
                inter = sum(np.logical_and(pred == i, real == i))
                u = sum(np.logical_or(pred == i, real == i))
                fp = sum(np.logical_and(pred == i, real != i))
                fn = sum(np.logical_or(pred != i, real == i))
                intersection[i] += inter
                union[i] += u
                FP[i] += fp
                FN[i] += fn

    intersection = Array("d", [0] * num_class)
    union = Array("d", [0] * num_class)
    FP = Array("d", [0] * num_class)
    FN = Array("d", [0] * num_class)

    p_list = []
    for i in range(len(image_list)):
        p = Process(target=f, args=(intersection, union, image_list[i]))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()

    eps = 1e-7
    total = 0
    total_f1 = 0
    for i in range(num_class):
        class_i = intersection[i] / (union[i] + eps)
        total += class_i
        if i == 1:
            class_precision = union[i] / (union[i] + FP[i])
            class_recall = union[i] / (union[i] + FN[i])
            total_f1 += (2 * class_precision * class_recall) / (class_precision + class_recall)
    return total / num_class, total_f1


def get_predmaps(input, model):
    with torch.no_grad():
        o_output = model(input.cuda())

    prob_maps = F.softmax(o_output[0], dim=0).cpu().numpy()
    pred = np.argmin(prob_maps, axis=0)

    return pred


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

