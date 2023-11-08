import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import skimage.morphology as morph
from skimage.measure import label
import utils
from metrics import iou_metrics, ObjectHausdorff, ObjectDice, ObjectF1score, Dice
import scipy.io as scio
from scipy import ndimage
import torchvision.transforms as transforms
from multiprocessing import Array, Process
import random
import cv2
from segment_anything import sam_model_registry, SamPredictor
from segment_anything import build_sam, SamAutomaticMaskGenerator
import os
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Testing SAM Model")
parser.add_argument('--num_workers', type=int, default=2, help='number of workers to load images')
parser.add_argument('--save_dir', type=str, default='./experimentsP')
parser.add_argument('--img_dir', type=str, default='/home/data2/MedImg/GlandSeg/CRAG/train/Images')
parser.add_argument('--label_dir', type=str, default='/home/data2/MedImg/GlandSeg/CRAG/train/Annotation')
# parser.add_argument('--prompt_dir', type=str, default='/home/data2/MedImg/GlandSeg/GlaS/test/Prompts', help='.mat file, contain point and box prompt')
parser.add_argument('--prompt_dir', type=str, default='experiments/CRAG_10labeled_round1/BoxPrompt/', help='.mat file, contain point and box prompt')

parser.add_argument('--desc', type=str, default='SAM')

parser.add_argument('--model', type=str, default="vit_h")
parser.add_argument('--model_path', type=str,
                    default="/home/data1/my/Project/segment-anything-main/sam_vit_h.pth", help='model weights for SAM model')

parser.add_argument('--dataset', type=str, choices=['GlaS', 'CRAG'], default='CRAG', help='which dataset be used')
parser.add_argument('--gpu', type=list, default=[3], help='GPUs for training')

# 后处理参数
parser.add_argument('--min_area', type=int, default=400, help='minimum area for an object')
parser.add_argument('--radius', type=int, default=4)
parser.add_argument('--mode', type=str, choices=['everything', 'prompt'], default='prompt', help='mode for SAM')
parser.add_argument('--prompt_mode', type=str, choices=['randomPoint', 'point', 'box'], default='box', help='prompt mode for SAM')
parser.add_argument('--round', type=int, default=1, help='number of round for self-training process')

args = parser.parse_args()
#torch.cuda.set_device(args.gpu)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)

    img_dir = args.img_dir
    label_dir = args.label_dir
    prompt_dir = args.prompt_dir
    save_flag = True

    # check if it is needed to compute accuracies
    eval_flag = True if label_dir else False
    img_names = os.listdir(img_dir)

    all_results = dict()

    save_dir = "%s/%s_%s_%s_%s_round%s" % (args.save_dir, args.dataset, args.desc, args.mode, args.prompt_mode, args.round)
    if save_flag:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        prob_maps_folder = '{:s}/{:s}'.format(save_dir, 'mask_pred')
        if not os.path.exists(prob_maps_folder):
            os.mkdir(prob_maps_folder)
        inst_maps_folder = '{:s}/{:s}'.format(save_dir, 'inst_pred')
        if not os.path.exists(inst_maps_folder):
            os.mkdir(inst_maps_folder)
        vis_maps_folder = '{:s}/{:s}'.format(save_dir, 'vis_pred')
        if not os.path.exists(vis_maps_folder):
            os.mkdir(vis_maps_folder)

    sam = sam_model_registry[args.model](checkpoint=args.model_path).cuda()
    if args.mode == 'prompt':
        predictor = SamPredictor(sam)
    elif args.mode == 'everything':
        mask_generator = SamAutomaticMaskGenerator(sam)

    for img_name in img_names:
        # load test image
        print('=> Processing image {:s}'.format(img_name))
        img_path = '{:s}/{:s}'.format(img_dir, img_name)
        orig_img = Image.open(img_path)
        name = os.path.splitext(img_name)[0]

        ########################### esenbling from multi-scale #################################
        img = Image.open(img_path)
        # x = np.array(img)

        ## 试下做一次颜色变换
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        contour_map = img.copy()
        if args.mode == 'everything':
            #======================= no prompt anything mode========================================#
            masks = mask_generator.generate(img)
            height, width, _ = img.shape
            pred = np.zeros(shape=(height, width))
            max_area = height * width
            all_binary_maps = []
            for mask in masks:
                if mask['area'] > max_area * 0.5:
                    continue
                binary_map = np.array(mask['segmentation'], np.uint8)
                all_binary_maps.append(binary_map)
                contours, hierarchy = cv2.findContours(binary_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in range(len(contours)):
                    contour_map = cv2.drawContours(contour_map, contours, contour, (0, 255, 0), 2, 8)
                pred += binary_map

        elif args.mode == 'prompt':
            # ============================= Prompt ===============================================#
            predictor.set_image(img)

            points = scio.loadmat('{:s}/{:s}.mat'.format(prompt_dir, name))['points']
            boxes = scio.loadmat('{:s}/{:s}.mat'.format(prompt_dir, name))['boxes']

            if args.prompt_mode == 'randomPoint':
                ## 随机取点
                pointNums = 5
                pointNums = pointNums if pointNums < points.shape[0] else points.shape[0]
                random_indices = np.random.choice(points.shape[0], pointNums, replace=False)
                points_selected = points[random_indices, :]

                mask, _, _ = predictor.predict(point_coords=points_selected, point_labels=np.ones(shape=(points_selected.shape[0], )),
                                               multimask_output=False)
                pred = mask[0]
            elif args.prompt_mode == 'point':
                ## point prompt
                mask, _, _ = predictor.predict(point_coords=points, point_labels=np.ones(shape=(points.shape[0], )),
                                               multimask_output=False)
                pred = mask[0]
            elif args.prompt_mode == 'box':
                ## box prompt
                height, width, _ = img.shape
                pred = np.zeros(shape=(height, width))
                all_binary_maps = []
                for b in range(boxes.shape[0]):
                    box = boxes[b]
                    mask, _, _ = predictor.predict(box=box, multimask_output=False)
                    binary_map = np.array(mask[0], np.uint8)
                    all_binary_maps.append(binary_map)
                    contours, hierarchy = cv2.findContours(binary_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in range(len(contours)):
                        contour_map = cv2.drawContours(contour_map, contours, contour, (0, 255, 0), 2, 8)
                    pred += mask[0]
                # ======================================================================================

        ############################### post proc ################################################
        # 将二值图像转换为标签图像
        label_image = label(pred)
        # 根据标记删除小连通域
        filtered_label_image = morph.remove_small_objects(label_image, min_size=args.min_area)
        # 将标签图像转换回二值图像
        pred = (filtered_label_image > 0).astype(np.uint8)
        # fill holes
        pred = ndimage.binary_fill_holes(pred)
        ###############################################################################################

        if eval_flag:
            # label_img = scio.loadmat('{:s}/{:s}.mat'.format(label_dir, name))['inst_map']
            if args.dataset == 'GlaS':
                label_img = cv2.imread('{:s}/{:s}_anno.bmp'.format(label_dir, name))[:, :, 0]
            elif args.dataset == 'CRAG':
                label_img = cv2.imread('{:s}/{:s}.png'.format(label_dir, name))[:, :, 0]
            label_img = np.array(label_img != 0, dtype=np.uint8)

        if eval_flag:
            img_show = np.concatenate([np.array(orig_img),
                                       np.stack((label_img * 255, label_img * 255, label_img * 255), axis=-1),
                                       np.stack((pred * 255, pred * 255, pred * 255), axis=-1)],
                                       axis=1)
            cv2.imwrite('{:s}/{}.png'.format(vis_maps_folder, name), img_show)
            cv2.imwrite('{:s}/{}_contour.png'.format(vis_maps_folder, name), contour_map)

            np.save('{:s}/{}.npy'.format(prob_maps_folder, name), pred)
            all_binary_maps = np.array(all_binary_maps)
            np.save('{:s}/{}.npy'.format(inst_maps_folder, name), all_binary_maps)
            print('\tComputing metrics...')
            result = utils.accuracy_pixel_level(np.expand_dims(pred > 0, 0), np.expand_dims(label_img > 0, 0))
            pixel_accu = result[0]

            # single_image_result = utils.gland_accuracy_object_level(pred_labeled, label_img)
            IoU = iou_metrics(pred, label_img)
            # dice = 2 * IoU / (1 + IoU)
            dice = Dice(np.where(pred>0, 1, 0), np.where(label_img>0, 1, 0))

            single_image_result = (IoU, dice)
            all_results[name] = tuple([pixel_accu, *single_image_result])
            # 打印每张test的指标
            print('Pixel Acc: {r[0]:.4f}\n'
                  'IoU: {r[1]:.4f}\n'
                  'Dice: {r[2]:.4f}'.format(r=[pixel_accu, IoU, dice]))

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
    print('Average Result =>\n'
          'IoU: {r[0]:.4f}\n'
          'Dice: {r[1]:.4f}'.format(r=[avg_iou, avg_dice]))
    save_results(header, [avg_pq, avg_iou, avg_dice], all_results,
                 f'{save_dir:s}/test_result__overall_iou_{over_all_iou:.4f}_dice_{2 * over_all_iou / (1 + over_all_iou):.4f}_'
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

            # groundtruth = io.imread(groundtruth_path + f"/{im_name}.png")
            # groundtruth = cv2.cvtColor(groundtruth, cv2.COLOR_BGR2GRAY)
            # groundtruth = np.array(groundtruth == 76, dtype=np.uint8).reshape(-1)
            # groundtruth = scio.loadmat(groundtruth_path + f"/{im_name}.mat")['inst_map']
            # groundtruth = np.array(groundtruth != 0, dtype=np.uint8).reshape(-1)
            groundtruth = cv2.imread(groundtruth_path + f"/{im_name}_anno.bmp")[:, :, 0].reshape(-1)
            groundtruth = np.array(groundtruth != 0, dtype=np.uint8)

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
    # for i in range(num_class):
    for i in [1]:
        class_i = intersection[i] / (union[i] + eps)
        total += class_i
        if i == 1:
            class_precision = union[i] / (union[i] + FP[i] + eps)
            class_recall = union[i] / (union[i] + FN[i] + eps)
            total_f1 += (2 * class_precision * class_recall) / (class_precision + class_recall + eps)
    # return total / num_class, total_f1
    return total, total_f1


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

