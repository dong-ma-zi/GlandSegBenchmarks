import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import skimage.morphology as morph
from skimage.measure import label
import utils
from metrics import iou_metrics
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

parser = argparse.ArgumentParser(description="Testing oeem segmentation Model")
parser.add_argument('--save_dir', type=str, default='./experimentsP3')

parser.add_argument('--img_dir', type=str, default='/home/data2/MedImg/NucleiSeg/MoNuSeg/Test/Images')
parser.add_argument('--label_dir', type=str, default='/home/data2/MedImg/NucleiSeg/MoNuSeg/Test/Annotation')
parser.add_argument('--prompt_dir', type=str, default='/home/data2/MedImg/NucleiSeg/MoNuSeg/Test/Prompts')

# parser.add_argument('--img_dir', type=str, default='/home/data2/MedImg/GlandSeg/GlaS/test/Images')
# parser.add_argument('--label_dir', type=str, default='/home/data2/MedImg/GlandSeg/GlaS/test/Annotation')
# parser.add_argument('--prompt_dir', type=str, default='/home/data2/MedImg/GlandSeg/GlaS/test/Prompts')

parser.add_argument('--desc', type=str,
                    # default='SAM-vit-h-sam',
                    default='SAM-vit-b-total-points',
                    )

parser.add_argument('--model', type=str, default="vit_b")
parser.add_argument('--model_path', type=str,
                    default="/home/data1/my/Project/segment-anything-main/sam_vit_b.pth")

parser.add_argument('--dataset', type=str, choices=['GlaS', 'CRAG'], default='MoNuSeg', help='which dataset be used')
parser.add_argument('--gpu', type=list, default=[0, ], help='GPUs for training')

# 后处理参数
parser.add_argument('--min_area', type=int, default=32, help='minimum area for an object')
args = parser.parse_args()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)

    img_dir = args.img_dir
    label_dir = args.label_dir
    prompt_dir = args.prompt_dir
    save_dir = "%s/%s_%s" % (args.save_dir, args.dataset, args.desc)
    save_flag = True

    # check if it is needed to compute accuracies
    eval_flag = True if label_dir else False

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

    sam = sam_model_registry[args.model](checkpoint=args.model_path).cuda()

    # predictor mode
    predictor = SamPredictor(sam)

    # anything mode
    # mask_generator = SamAutomaticMaskGenerator(sam)


    for img_name in img_names:
        # load test image
        print('=> Processing image {:s}'.format(img_name))
        img_path = '{:s}/{:s}'.format(img_dir, img_name)
        orig_img = Image.open(img_path)
        name = os.path.splitext(img_name)[0]

        ################################################################################################
        img = Image.open(img_path)

        #-------------------------- Any Mode -------------------------------#
        # masks = mask_generator.generate(np.array(img))
        # width, height = img.size
        # pred = np.zeros(shape=(height, width))
        # for mask in masks:
        #     # if mask['area'] > 150000: continue
        #     if mask['area'] > 10000: continue
        #     pred += mask['segmentation']
        # ------------------------------------------------------------------#

        # get points/boxes prompts
        predictor.set_image(np.array(img))
        points = scio.loadmat('{:s}/{:s}.mat'.format(prompt_dir, name))['points']
        points = points[:, np.newaxis, :] # K, 2 --> N, K, 2
        boxes = scio.loadmat('{:s}/{:s}.mat'.format(prompt_dir, name))['boxes']

        #----------------------- ponits promp ------------------------------#
        # selected points
        # pointNums = 20
        # pointNums = pointNums if pointNums < points.shape[0] else points.shape[0]
        # random_indices = np.random.choice(points.shape[0], pointNums, replace=False)
        # points_selected = points[random_indices, :, :]
        # width, height = img.size
        # pred = np.zeros(shape=(height, width))
        # for i in range(points_selected.shape[0]):
        #     mask, _, _ = predictor.predict(point_coords=points_selected[i],
        #                                    point_labels=np.array([1]),
        #                                    # point_labels=np.ones(shape=(points_selected.shape[0], )),
        #                                    multimask_output=False)
        #     pred += mask[0]


        # total points
        width, height = img.size
        pred = np.zeros(shape=(height, width))
        for i in range(points.shape[0]):
            mask, _, _ = predictor.predict(point_coords=points[i], point_labels=np.array([1]),
                                           multimask_output=False)
            pred += mask[0]
        # ----------------------- ponits promp ------------------------------#

        # ------------------------ boxes promp ------------------------------#
        # width, height = img.size
        # pred = np.zeros(shape=(height, width))
        # for b in range(boxes.shape[0]):
        #     box = boxes[b]
        #     mask, _, _ = predictor.predict(box=box, multimask_output=False)
        #     pred += mask[0]
        # ------------------------ boxes promp ------------------------------#

        ########################################################################################################

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
            label_img = scio.loadmat('{:s}/{:s}.mat'.format(label_dir, name))['inst_map']
            # label_img = cv2.imread('{:s}/{:s}_anno.bmp'.format(label_dir, name))[:, :, 0]
            label_img = np.array(label_img != 0, dtype=np.uint8)


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
            groundtruth = scio.loadmat(groundtruth_path + f"/{im_name}.mat")['inst_map']
            groundtruth = np.array(groundtruth != 0, dtype=np.uint8).reshape(-1)

            # groundtruth = cv2.imread(groundtruth_path + f"/{im_name}_anno.bmp")[:, :, 0].reshape(-1)
            # groundtruth = np.array(groundtruth != 0, dtype=np.uint8)

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

