import numpy as np
import torch.nn.functional as F
import skimage.morphology as morph
from skimage.measure import label
from skimage import measure, io
import utils
from metrics import dice_coefficient, iou_metrics
from transforms import ResizeLongestSide
from scipy import ndimage
# import torchvision.transforms as transforms
import argparse
from multiprocessing import Array, Process
from models.sam_orig import SamPredictor, sam_model_registry
# from models.sam import SamPredictor, sam_model_registry
import cv2
from utils import *
import scipy.io as scio

parser = argparse.ArgumentParser(description="Testing oeem segmentation Model")

parser.add_argument('--save_dir', type=str, default='./experimentsP')

# parser.add_argument('--img_dir', type=str, default='/home/data2/MedImg/GlandSeg/GlaS/test_proc/Images')
# parser.add_argument('--label_dir', type=str, default='/home/data2/MedImg/GlandSeg/GlaS/test_proc/Annotation')

parser.add_argument('--img_dir', type=str, default='/home/data2/MedImg/NucleiSeg/MoNuSeg/Test/Images')
parser.add_argument('--label_dir', type=str, default='/home/data2/MedImg/NucleiSeg/MoNuSeg/Test/Annotation/')

parser.add_argument('--mode', type=str, default='orig')
parser.add_argument('--desc', type=str,
                    # default='SAM-vit-h',
                    default='SAM-vit-b-wo-ft',
                    )

parser.add_argument('--model_path', type=str,
                    # w/o finrtuning
                    default=None
                    # glas orig sam
                    # default="/home/data1/my/Project/GlandSegBenchmark/Medical-SAM-Adapter/logs_p/glas-samOrig-b-1024-16-256_2023_10_29_20_02/Model/checkpoint_400.pth",
                    # default="/home/data1/my/Project/GlandSegBenchmark/Medical-SAM-Adapter/logs_p/glas-samOrig-h-1024-16-256_2023_10_29_20_04/Model/checkpoint_200.pth"
                    # glas adpt sam
                    # default="/home/data1/my/Project/GlandSegBenchmark/Medical-SAM-Adapter/logs_p/glas-samAdpt-b-1024-16-256_2023_10_31_21_38/Model/checkpoint_380.pth"
                    # default="/home/data1/my/Project/GlandSegBenchmark/Medical-SAM-Adapter/logs_p/glas-samAdpt-h-1024-16-256_2023_10_31_21_42/Model/checkpoint_200.pth"
                    # monuseg orig sam
                    # default="/home/data1/my/Project/GlandSegBenchmark/Medical-SAM-Adapter/logs_p/monuseg-samOrig-b-1024-16-256_2023_10_28_20_29/Model/checkpoint_100"
                    # default="/home/data1/my/Project/GlandSegBenchmark/Medical-SAM-Adapter/logs_p/monuseg-samOrig-h-1024-16-256_2023_11_01_15_49/Model/checkpoint_40.pth"
                    # monuseg adpt sam
                    # default="/home/data1/my/Project/GlandSegBenchmark/Medical-SAM-Adapter/logs_p/monuseg-samAdpt-b-1024-16-256_2023_11_01_16_10/Model/checkpoint_50.pth"
                    )

parser.add_argument('--dataset', type=str, choices=['GlaS', 'CRAG'], default='MoNuSeg', help='which dataset be used')
parser.add_argument('--gpu', type=list, default=[0, ], help='GPUs for training')

# 后处理参数
parser.add_argument('--min_area', type=int, default=32, help='minimum area for an object')
# parser.add_argument('--radius', type=int, default=4)
args = parser.parse_args()


def img_preprocessing(image, sam):
    original_image_size = (image.shape[0], image.shape[1])
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    input_image = transform.apply_image(np.array(image))
    input_image = torch.as_tensor(input_image, dtype=torch.float32, device=sam.device) # set to float32
    input_image = input_image.permute(2, 0, 1).contiguous()[None, :, :, :]
    input_size = input_image.shape[-2:]
    input_image = sam.preprocess(input_image) # do not need padding here
    return input_image, original_image_size, input_size

def get_scaled_prompt(points, sam, original_image_size, if_transform: bool = True):
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    points = transform.apply_coords(points, original_image_size) if if_transform else points
    points = torch.as_tensor(points, device=sam.device).unsqueeze(1)
    points = (points, torch.ones(points.shape[0], 1))
    return points

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)

    img_dir = args.img_dir
    label_dir = args.label_dir
    if args.mode == 'adpt':
        save_dir = "%s/%s_%s-adpt" % (args.save_dir, args.dataset, args.desc)
    else:
        save_dir = "%s/%s_%s" % (args.save_dir, args.dataset, args.desc)

    model_path = args.model_path
    save_flag = True
    points_batch_size = 16

    # check if it is needed to compute accuracies
    eval_flag = True if label_dir else False

    # model = sam_model_registry['vit_h']().cuda()
    # model = sam_model_registry['vit_b'](checkpoint="/home/data1/my/Project/segment-anything-main/sam_vit_b.pth").cuda()
    # model = sam_model_registry['vit_b'](args).cuda()

    sam = sam_model_registry['vit_b'](checkpoint="/home/data1/my/Project/segment-anything-main/sam_vit_b.pth").cuda()
    # predictor mode
    predictor = SamPredictor(sam)

    epoch = '0'

    print("=> Test begins:")

    img_names = os.listdir(img_dir)

    # TP, FP, FN, dice_g, dice_s, iou_g, iou_s, haus_g, haus_s, gt_area, seg_area

    all_results = dict()

    if save_flag:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        prob_maps_folder = '{:s}/{:s}'.format(save_dir, 'mask_pred')
        if not os.path.exists(prob_maps_folder):
            os.mkdir(prob_maps_folder)
        vis_maps_folder = '{:s}/{:s}'.format(save_dir, 'vis_pred')
        if not os.path.exists(vis_maps_folder):
            os.mkdir(vis_maps_folder)

    for img_name in img_names:
        # load test image
        print('=> Processing image {:s}'.format(img_name))
        img_path = '{:s}/{:s}'.format(img_dir, img_name)
        orig_img = Image.open(img_path)
        name = os.path.splitext(img_name)[0]

        ########################### esenbling from multi-scale #################################
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        h, w, _ = image.shape
        res_mask = np.zeros((h, w), np.uint8)

        # label_path = '{:s}/{:s}_anno.bmp'.format(label_dir, name)
        # label_img = np.array(Image.open(label_path))
        label_img = scio.loadmat('{:s}/{:s}.mat'.format(label_dir, name))['inst_map']

        ''' preprocess '''
        # input_image, original_image_size, input_size = img_preprocessing(image, sam)
        # pts_orig_scale, masks = generate_click_prompt_all_inst(label_img)
        pts_orig_scale, masks = generate_centroid_click_prompt_all_inst(label_img)
        pts_orig_scale = pts_orig_scale[:, np.newaxis, :]  # K, 2 --> N, K, 2

        # pts_orig_scale = scio.loadmat('{:s}/{:s}.mat'.format('/home/data2/MedImg/NucleiSeg/MoNuSeg/Test/Prompts',
        #                                              name))['points']
        # pts_orig_scale = pts_orig_scale[:, np.newaxis, :]  # K, 2 --> N, K, 2

        predictor.set_image(image)
        for i in range(pts_orig_scale.shape[0]):
            ''' forward '''
            mask, _, _ = predictor.predict(point_coords=pts_orig_scale[i], point_labels=np.array([1]),
                                           multimask_output=False)
            res_mask += mask[0]

            # res_mask[image_mask == 1] = 1
        pred = res_mask

        ############################### post proc ################################################
        # 将二值图像转换为标签图像
        label_image = label(pred)
        # 根据标记删除小连通域
        filtered_label_image = morph.remove_small_objects(label_image, min_size=args.min_area)
        # 将标签图像转换回二值图像
        pred = (filtered_label_image > 0).astype(np.uint8)
        # fill holes
        pred = ndimage.binary_fill_holes(pred)
        ################################################################################################ # remove small object

        if eval_flag:
            label_img = np.array(label_img != 0, dtype=np.uint8)
            label_img_vis = cv2.cvtColor(label_img * 255, cv2.COLOR_GRAY2RGB)
            pred_vis = cv2.cvtColor(pred.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
            # for point in pts_orig_scale:
            #     cv2.circle(pred_vis, (point[1], point[0]), 3, (0, 0, 255), -1)
            img_show = np.concatenate([np.array(orig_img),
                                       label_img_vis,
                                       pred_vis],
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

            # groundtruth = np.asarray(Image.open(groundtruth_path + f"/{im_name}_anno.bmp"))
            groundtruth = scio.loadmat(groundtruth_path + f"/{im_name}.mat")['inst_map']
            groundtruth = np.array(groundtruth != 0, dtype=np.uint8).reshape(-1)

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

# from PIL import Image
# import numpy as np
# import os
# from skimage import measure
# import glob
# import scipy.io as scio
#
# label_list = glob.glob('/home/data2/MedImg/NucleiSeg/MoNuSeg/Test/Annotation/*.mat')
# nuclei_sum = 0
# for label_path in label_list:
#     label_img = scio.loadmat(label_path)['inst_map']
#     nuclei_sum += label_img.max()
# print(nuclei_sum)


# for label_name in ['testB_3_anno.bmp', 'testB_11_anno.bmp',
#                    'testB_18_anno.bmp', 'testB_19_anno.bmp']:
#     label_path = os.path.join('/home/data2/MedImg/GlandSeg/GlaS/test_proc/Annotation/',
#                              label_name)
#     label_img = np.array(Image.open(label_path))
#     x = 1
#     # label_img_labeled = measure.label(label_img).astype(np.uint8) # connected component labeling
#     # label_img_labeled = Image.fromarray(label_img_labeled)
#     # label_img_labeled.save(label_path)