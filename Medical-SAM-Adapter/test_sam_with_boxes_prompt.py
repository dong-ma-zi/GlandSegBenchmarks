import numpy as np
import torch.nn.functional as F
import skimage.morphology as morph
from skimage.measure import label
from skimage import measure, io
import utils
from metrics import dice_coefficient, iou_metrics, generate_cls_info, run_nuclei_type_stat
from transforms import ResizeLongestSide
from scipy import ndimage
# import torchvision.transforms as transforms
import argparse
from multiprocessing import Array, Process
# from models.sam_orig import SamPredictor, sam_model_registry
from models.sam import SamPredictor, sam_model_registry
# import cv2
from utils import *
import scipy.io as scio

parser = argparse.ArgumentParser(description="Testing oeem segmentation Model")
parser.add_argument('--save_dir', type=str, default='./experimentsBoxes')

# parser.add_argument('--img_dir', type=str, default='/home/data2/MedImg/GlandSeg/GlaS/test_proc/Images')
# parser.add_argument('--label_dir', type=str, default='/home/data2/MedImg/GlandSeg/GlaS/test_proc/Annotation')

# parser.add_argument('--img_dir', type=str, default='/home/data2/MedImg/NucleiSeg/MoNuSeg/Test/Images')
# parser.add_argument('--label_dir', type=str, default='/home/data2/MedImg/NucleiSeg/MoNuSeg/Test/Annotation/')

parser.add_argument('--img_dir', type=str, default='/home/data1/my/dataset/consep/Test/Images/')
parser.add_argument('--label_dir', type=str, default='/home/data1/my/dataset/consep/Test/Labels/')

# parser.add_argument('--img_dir', type=str, default='/home/data1/my/dataset/monusac/Test/Images/')
# parser.add_argument('--label_dir', type=str, default='/home/data1/my/dataset/monusac/Test/Labels/')

parser.add_argument('--mode', type=str,
                    default='adpt',
                    )

parser.add_argument('--desc', type=str,
                    # default='SAM-vit-h',
                    default='SAM-vit-b',
                    )

parser.add_argument('--model_path', type=str,
                    # w/o finetuning
                    # default=None
                    # glas orig sam
                    # default="/home/data1/my/Project/GlandSegBenchmark/Medical-SAM-Adapter/logs_boxes_mod1107/glas-samOrig-b-1024-16-256_2023_11_08_10_32/Model/checkpoint_99.pth"
                    # default="/home/data1/my/Project/GlandSegBenchmark/Medical-SAM-Adapter/logs_boxes_mod1107/glas-samOrig-h-1024-16-256_2023_11_08_21_38/Model/checkpoint_50.pth"
                    # glas adpt sam
                    # default="/home/data1/my/Project/GlandSegBenchmark/Medical-SAM-Adapter/logs_boxes_mod1107/glas-samAdpt-b-1024-16-256_2023_11_07_14_16/Model/checkpoint_100.pth"
                    # default="/home/data1/my/Project/GlandSegBenchmark/Medical-SAM-Adapter/logs_boxes_mod1107/glas-samAdpt-h-1024-16-256_2023_11_08_18_26/Model/checkpoint_50.pth"
                    # monuseg orig sam
                    # default="/home/data1/my/Project/GlandSegBenchmark/Medical-SAM-Adapter/logs_boxes_mod1107/monuseg-samOrig-b-1024-16-256_2023_11_08_00_00/Model/checkpoint_50.pth"
                    # monuseg adpt sam
                    # default="/home/data1/my/Project/GlandSegBenchmark/Medical-SAM-Adapter/logs_boxes_mod1107/monuseg-samAdpt-b-1024-16-256_2023_11_08_00_03/Model/checkpoint_20.pth"
                    # consep orig sam
                    # default="/home/data1/my/Project/GlandSegBenchmark/Medical-SAM-Adapter/logs_boxes_mod1107/consep-samOrig-b-1024-16-256_2023_11_13_23_50/Model/checkpoint_45.pth"
                    # consep adpt sam
                    default="/home/data1/my/Project/GlandSegBenchmark/Medical-SAM-Adapter/logs_boxes_mod1107/consep-samAdpt-b-1024-16-256_2023_11_13_23_46/Model/checkpoint_15.pth"
                    # monusac
                    # default="/home/data1/my/Project/GlandSegBenchmark/Medical-SAM-Adapter/logs_boxes_mod1107/monusac-samOrig-b-1024-16-256_2023_11_21_20_47/Model/checkpoint_30.pth"
                    # monusac
                    # default="/home/data1/my/Project/GlandSegBenchmark/Medical-SAM-Adapter/logs_boxes_mod1107/monusac-samAdpt-b-1024-16-256_2023_11_22_11_39/Model/checkpoint_30.pth"
                    )

parser.add_argument('--dataset', type=str, choices=['GlaS', 'MoNuSeg', 'CoNSeP','MoNuSAC'],
                    default='CoNSeP', help='which dataset be used')
parser.add_argument('--gpu', type=list, default=[2, ], help='GPUs for training')

# auto prompt
parser.add_argument('--auto_prompt', type=bool, default=True)
parser.add_argument('--box_txt', type=str,
                    # default='/home/data1/my/Project/yolov5/runs/detect/exp/labels/'
                    # default='/home/data1/my/Project/nucleiDetCls/DeGPR/Detection/yolov5/runs/detect/exp/labels/'
                    # default='/home/data1/my/Project/yolov5/runs/detect/exp3/labels/',
                    # default='/home/data1/my/Project/yolov5/runs/detect/exp4/labels',
                    # default='/home/data1/my/Project/nucleiDetCls/Deformable-DETR/output/monusac1200/test_labels/'
                    default='/home/data1/my/Project/nucleiDetCls/Deformable-DETR/output/consep2400/test_labels/'
                    )
# 后处理参数
parser.add_argument('--min_area', type=int, default=32, help='minimum area for an object')
args = parser.parse_args()


import random
def draw_rand_inst_color(inst_map):
    color_map = np.zeros(shape=(inst_map.shape[0], inst_map.shape[1], 3))

    ind_list = np.unique(inst_map).tolist()
    ind_list.remove(0)

    for ind in ind_list:
        R = random.randint(0, 255)
        G = random.randint(0, 255)
        B = random.randint(0, 255)
        color_map[inst_map == ind, :] = [B, G, R]
    return color_map


# color_dict = {1: [0, 255, 0], 2: [0, 0, 250], 3: [255, 144, 30]}
color_dict = {1: [0, 0, 255], 2: [0, 255, 255], 3: [0, 255, 0], 4:[255, 0, 0]}
def draw_inst_overlay(ori_img, inst_map, cls_map=None):
    img = np.copy(ori_img).astype(np.uint8)
    img = np.ascontiguousarray(img)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    inst_map = np.copy(inst_map)
    insts = np.unique(inst_map).tolist()
    insts.remove(0)
    for ins in insts:
        inst_mask = np.copy(inst_map)
        inst_mask[inst_mask != ins] = 0

        if isinstance(cls_map, np.ndarray):
            cls_num = list(set(cls_map[inst_mask == ins].flatten().tolist()))
            color = color_dict[int(cls_num[0])]
        else:
            color = [0, 0, 0]
        inst_mask[inst_mask > 0] = 255
        inst_mask = inst_mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(inst_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in range(len(contours)):
            img = cv2.drawContours(img, contours, contour, color, 2, 8)

    return img

def img_preprocessing(image, sam):
    original_image_size = (image.shape[0], image.shape[1])
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    input_image = transform.apply_image(np.array(image))
    input_image = torch.as_tensor(input_image, dtype=torch.float32, device=sam.device) # set to float32
    input_image = input_image.permute(2, 0, 1).contiguous()[None, :, :, :]
    input_size = input_image.shape[-2:]
    input_image = sam.preprocess(input_image) # do not need padding here
    return input_image, original_image_size, input_size


def get_scaled_prompt(boxes, sam, original_image_size, if_transform: bool = True):
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    boxes = transform.apply_boxes(boxes, original_image_size) if if_transform else boxes
    boxes = torch.as_tensor(boxes, device=sam.device)
    return boxes


def main():
    assert args.dataset in ['GlaS', 'MoNuSeg', 'CoNSeP', 'MoNuSAC']
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)

    img_dir = args.img_dir
    label_dir = args.label_dir
    if args.mode == 'adpt':
        save_dir = "%s/%s_%s-adpt" % (args.save_dir, args.dataset, args.desc)
    else:
        save_dir = "%s/%s_%s" % (args.save_dir, args.dataset, args.desc)

    if not args.model_path:
        save_dir += '-wo-ft'

    if args.auto_prompt:
        save_dir += '_auto_prop_defdetr'

    model_path = args.model_path
    save_flag = True
    points_batch_size = 8

    # check if it is needed to compute accuracies
    eval_flag = True if label_dir else False

    # set init metrics
    if args.dataset == 'GlaS':
        accumulated_metrics = np.zeros(11)
        args.min_area = 400
    # if args.dataset == 'MoNuSeg':
    else:
        args.min_area = 32
        aji = 0
        nuclei_pq = 0
        pred_dict = {}
        true_dict = {}

    # model init
    if not args.model_path:
        model = sam_model_registry['vit_b'] \
            (checkpoint="/home/data1/my/Project/segment-anything-main/sam_vit_b.pth").cuda()
    elif args.mode == 'orig':
        model = sam_model_registry['vit_b']().cuda()
    elif args.mode == 'adpt':
        model = sam_model_registry['vit_b'](args).cuda()

    if args.model_path:
        epoch = os.path.basename(model_path).split('.')[0].split('_')[-1]
        print("=> loaded model at epoch {}".format(epoch))
        weights = torch.load(args.model_path, map_location='cpu')["state_dict"]
        model.load_state_dict(weights, strict=True)
    else:
        epoch = '0'

    # switch to evaluate mode
    model.eval()
    print("=> Test begins:")
    img_names = os.listdir(img_dir)

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
        # orig_img = Image.open(img_path)
        name = os.path.splitext(img_name)[0]

        ########################### esenbling from multi-scale #################################
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        h, w, _ = image.shape
        res_mask = np.zeros((h, w), np.int32)
        res_cls_mask = np.zeros((h, w), np.int32)
        per_img_mask_list = []
        iou_pred_list = []
        inst_id = 1
        if args.dataset == 'GlaS':
            label_path = '{:s}/{:s}_anno.bmp'.format(label_dir, name)
            inst_label = np.array(Image.open(label_path))
        elif args.dataset  == 'MoNuSeg':
            inst_label = scio.loadmat('{:s}/{:s}.mat'.format(label_dir, name))['inst_map']
        elif args.dataset == 'CoNSeP':
            inst_label = scio.loadmat('{:s}/{:s}.mat'.format(label_dir, name))['inst_map']
            cls_label_ = scio.loadmat('{:s}/{:s}.mat'.format(label_dir, name))['type_map']

            # cls_label[(cls_label == 3) | (cls_label == 4)] = 3
            # cls_label[(cls_label == 5) | (cls_label == 6) | (cls_label == 7)] = 4

            cls_label = np.zeros_like(cls_label_)
            cls_label[cls_label_ == 2] = 1
            cls_label[(cls_label_ == 3) | (cls_label_ == 4)] = 2
            cls_label[(cls_label_ == 5) | (cls_label_ == 6) | (cls_label_ == 7) | (cls_label_ == 1)] = 3

        elif args.dataset == 'MoNuSAC':
            # x = scio.loadmat('{:s}/{:s}.mat'.format(label_dir, name))
            inst_label = scio.loadmat('{:s}/{:s}.mat'.format(label_dir, name))['inst_map']
            cls_label = scio.loadmat('{:s}/{:s}.mat'.format(label_dir, name))['type_map']
            inst_label[cls_label == 5] = 0
            cls_label[cls_label == 5] = 0
            inst_label = remap_label(inst_label)

        inst_label = inst_label.astype(np.int32)

        ''' preprocess '''
        input_image, original_image_size, input_size = img_preprocessing(image, model)
        # load or generate prompt
        if args.auto_prompt:
            # if not prompts were predicted
            if not os.path.exists(os.path.join(args.box_txt, name + '.txt')):
                # boxes_orig_scale = np.empty(shape=(1, 4))
                # boxes = get_scaled_prompt(boxes_orig_scale, model, original_image_size)
                # cls_orig_scale = np.ones(shape=(boxes_orig_scale.shape[0], 1))
                boxes = torch.empty(size=(0, 4))
                cls_orig_scale = torch.empty(size=(0, 1))
            else:
                with open(os.path.join(args.box_txt, name + '.txt'), 'r') as f:
                    lines = f.readlines()
                    boxes_orig_scale = np.zeros(shape=(len(lines), 4))
                    cls_orig_scale = np.zeros(shape=(len(lines), 1))
                    for box_idx, line in enumerate(lines):
                        cls, x, y, box_w, box_h = line.replace('\n', '').split(' ')
                        miny = float(x) * w - 0.5 * float(box_w) * w
                        maxy = float(x) * w + 0.5 * float(box_w) * w
                        minx = float(y) * h - 0.5 * float(box_h) * h
                        maxx = float(y) * h + 0.5 * float(box_h) * h
                        boxes_orig_scale[box_idx] = np.array([miny, minx, maxy, maxx])
                        cls_orig_scale[box_idx] = np.array([int(cls) + 1])
                boxes = get_scaled_prompt(boxes_orig_scale, model, original_image_size)
        else:
            boxes_orig_scale, masks = generate_boxes_prompt_all_inst(inst_label)
            boxes = get_scaled_prompt(boxes_orig_scale, model, original_image_size)
            cls_orig_scale = np.ones(shape=(boxes_orig_scale.shape[0], 1))

        box_num = boxes.shape[0]
        if box_num % points_batch_size == 0:
            batch_num = box_num // points_batch_size
        else:
            batch_num = box_num // points_batch_size + 1

        for i in range(batch_num):
            if i != (box_num // points_batch_size):
                single_boxes = boxes[i * points_batch_size: (i + 1) * points_batch_size]
            else:
                single_boxes = boxes[i * points_batch_size:]
            if single_boxes[0].shape[0] == 0: print('no prompt input')

            ''' forward '''
            # --------------------- get prompt --------------------------#
            with torch.no_grad():
                imge= model.image_encoder(input_image)
                se, de = model.prompt_encoder(
                    points=None,
                    boxes=single_boxes,
                    masks=None,
                )

                output, iou_pred = model.mask_decoder(
                    image_embeddings=imge,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=False,
                )

            ''' postprocess '''
            # msk
            upscaled_masks = model.postprocess_masks(output, input_size, original_image_size).squeeze(1)
            upscaled_masks = (upscaled_masks > 0.5).float().detach().cpu().numpy()

            # for ind in range(upscaled_masks.shape[0]):
            #     res_mask[upscaled_masks[ind] == 1] = inst_id
            #     inst_id += 1
            per_img_mask_list += [upscaled_masks[i] for i in range(upscaled_masks.shape[0])]
            iou_pred_list += [iou_pred.detach().cpu()[:, 0]]

        ############################### post proc ################################################
        # nms
        if batch_num != 0: # if prompts are given
            iou_pred_list = torch.cat(iou_pred_list, dim=0)
            selected_masks, remain_idx = non_max_suppression(per_img_mask_list, iou_pred_list, 0.5)
            cls_orig_scale_nms = cls_orig_scale[remain_idx]
            for id, per_inst_mask in enumerate(selected_masks):
                res_mask[per_inst_mask == 1] = inst_id
                res_cls_mask[per_inst_mask == 1] = cls_orig_scale_nms[id]
                inst_id += 1
            # 根据标记删除小连通域
            filtered_label_image = morph.remove_small_objects(res_mask, min_size=args.min_area)
            # 将标签图像转换回二值图像
            sem_pred = (filtered_label_image > 0).astype(np.uint8)
        else:
            filtered_label_image = np.zeros_like(inst_label)
            sem_pred = np.zeros_like(inst_label)
            res_cls_mask = np.zeros_like(cls_label)

        if args.dataset == 'MoNuSAC':
            filtered_label_image[cls_label == 5] = 0
            filtered_label_image = morph.remove_small_objects(res_mask, min_size=args.min_area)
            sem_pred = (filtered_label_image > 0).astype(np.uint8)
            res_cls_mask[filtered_label_image == 0] = 0

        # TODO: remap the
        inst_pred = utils.remap_label(filtered_label_image)

        pred_dict[name] = generate_cls_info(inst_pred, res_cls_mask)
        true_dict[name] = generate_cls_info(inst_label, cls_label)

        if eval_flag:
            label_img = np.array(inst_label != 0, dtype=np.uint8)
            label_vis = draw_rand_inst_color(inst_label)
            pred_vis = draw_rand_inst_color(inst_pred)
            pred_overlay = draw_inst_overlay(image, inst_pred, res_cls_mask)

            for box in boxes_orig_scale:
                cv2.rectangle(pred_vis,
                              (round(box[0]), round(box[1])),
                              (round(box[2]), round(box[3])), (0, 0, 255), 3)

            img_show = np.concatenate([image,
                                       label_vis,
                                       pred_vis,
                                       pred_overlay],
                                       axis=1)
            cv2.imwrite('{:s}/{}.png'.format(vis_maps_folder, name), img_show)
            np.save('{:s}/{}.npy'.format(prob_maps_folder, name), sem_pred)

            # ------------------------------ metrics measurement ------------------------------- #
            print('\tComputing metrics...')
            # get iou and dice per image
            result = utils.accuracy_pixel_level(np.expand_dims(sem_pred > 0, 0), np.expand_dims(label_img > 0, 0))
            pixel_accu = result[0]
            # single_image_result = utils.gland_accuracy_object_level(pred_labeled, label_img)
            IoU = iou_metrics(sem_pred, label_img)
            Dice = 2 * IoU / (1 + IoU)
            single_image_result = (IoU, Dice)
            all_results[name] = tuple([pixel_accu, *single_image_result])
            # 打印每张test的指标
            print('Pixel Acc: {r[0]:.4f}\n'
                  'IoU: {r[1]:.4f}\n'
                  'Dice: {r[2]:.4f}'.format(r=[pixel_accu, IoU, Dice]))
            # get instance metrics
            if args.dataset == 'GlaS':
                accumulated_metrics += utils.gland_accuracy_object_level_all_images(inst_pred, inst_label)
            if args.dataset in ['MoNuSeg', 'CoNSeP', 'MoNuSAC']:
                # x = np.max(inst_label)
                # y = np.max(inst_pred)
                if np.max(inst_label) != 0 and np.max(inst_pred) == 0:
                    aji_img = 0
                else:
                    aji_img = utils.get_fast_aji(inst_label, inst_pred)
                aji += aji_img
                dq_sq_pq, _ = utils.get_fast_pq(inst_label, inst_pred)
                nuclei_pq += dq_sq_pq[2]
                print('AJI: {r[0]:.4f}\n'
                      'PQ: {r[1]:.4f}'.format(r=[aji_img, dq_sq_pq[2]]))

    over_all_iou, over_all_f1 = get_overall_valid_score('{:s}'.format(prob_maps_folder), args.label_dir,
                                                        dataset_name=args.dataset)

    # semantic
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

    # instance
    if args.dataset == 'GlaS':
        TP_, FP_, FN_, dice_g, dice_s, iou_g, iou_s, hausdorff_g, hausdorff_s, \
            gt_objs_area, pred_objs_area = accumulated_metrics
        # recall = TP / (TP + FN)
        # precision = TP / (TP + FP)
        F1 = 2 * TP_ / (2 * TP_ + FP_ + FN_)
        obj_dice = (dice_g / gt_objs_area + dice_s / pred_objs_area) / 2
        # iou = (iou_g / gt_objs_area + iou_s / pred_objs_area) / 2
        haus = (hausdorff_g / gt_objs_area + hausdorff_s / pred_objs_area) / 2
        header = ['F1', 'objDice', 'objHaus']
        save_results(header, [F1, obj_dice, haus], {},
                     f'{save_dir:s}/test_result_epoch{epoch}_obj_dice_{obj_dice:.4f}_obj_haus_{haus:.4f}.txt')

    elif args.dataset in ['MoNuSeg', 'CoNSeP', 'MoNuSAC']:
        results_list = run_nuclei_type_stat(pred_dict, true_dict, _20x=False)
        print(results_list)
        header = ['aji', 'pq', 'f1', 'f2', 'f3', 'f4']
        save_results(header, [aji / len(img_names), nuclei_pq / len(img_names),
                              results_list[-4], results_list[-3], results_list[-2], results_list[-1]],
                     {},
                     f'{save_dir:s}/test_result_epoch{epoch}_aji_{aji / len(img_names):.4f}_pq_{nuclei_pq / len(img_names):.4f}.txt')

        # header = ['aji', 'pq']
        # save_results(header, [aji / len(img_names), nuclei_pq / len(img_names)],
        #              {},
        #              f'{save_dir:s}/test_result_epoch{epoch}_aji_{aji / len(img_names):.4f}_pq_{nuclei_pq / len(img_names):.4f}.txt')


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


def get_overall_valid_score(pred_image_path, groundtruth_path, dataset_name, num_workers=1, num_class=2):
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

            if dataset_name == 'GlaS':
                groundtruth = np.asarray(Image.open(groundtruth_path + f"/{im_name}_anno.bmp"))
            elif dataset_name in ['MoNuSeg', 'CoNSeP']:
                groundtruth = scio.loadmat(groundtruth_path + f"/{im_name}.mat")['inst_map']
            elif dataset_name == 'MoNuSAC':
                groundtruth = scio.loadmat(groundtruth_path + f"/{im_name}.mat")['inst_map']
                cls_gt = scio.loadmat(groundtruth_path + f"/{im_name}.mat")['type_map']
                groundtruth[cls_gt == 5] = 0

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