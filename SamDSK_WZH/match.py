from utils import draw_rand_inst_overlay
from metrics import iou_metrics
import cv2
from skimage import measure
import skimage.morphology as morph
import os
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Match SwinUnet output and SAM output")
parser.add_argument('--save_dir', type=str, default='experimentsP')
parser.add_argument('--output_dir', type=str, default='experiments')
parser.add_argument('--img_dir', type=str, default='/home/data2/MedImg/GlandSeg')
parser.add_argument('--label_dir', type=str, default='/home/data2/MedImg/GlandSeg')

parser.add_argument('--desc', type=str, default='SAM')
parser.add_argument('--dataset', type=str, choices=['GlaS', 'CRAG'], default='CRAG', help='which dataset be used')

# 后处理参数
parser.add_argument('--iou_threshold', type=float, default=0.2, help='instance wise threshold')
parser.add_argument('--threshold', type=float, default=0.5, help='image wise threshold')
parser.add_argument('--min_area', type=int, default=400, help='minimum area for an object')
parser.add_argument('--radius', type=int, default=4)
parser.add_argument('--mode', type=str, choices=['everything', 'prompt'], default='prompt', help='mode for SAM')
parser.add_argument('--prompt_mode', type=str, choices=['randomPoint', 'point', 'box'], default='box', help='prompt mode for SAM')
parser.add_argument('--round', type=int, default=1, help='number of round for self-training process')
args = parser.parse_args()


def main():
    img_dir = os.path.join(args.img_dir, args.dataset, 'train', 'Images')
    label_dir = os.path.join(args.label_dir, args.dataset, 'train', 'Annotation')

    save_dir = "%s/%s_%s_%s_%s_round%s" % (args.save_dir, args.dataset, args.desc, args.mode, args.prompt_mode, args.round)
    output_dir = "%s/%s_10labeled_round%s" % (args.output_dir, args.dataset, args.round)
    inst_output_dir = os.path.join(output_dir, 'inst_pred')
    overlay_output_dir = os.path.join(output_dir, 'overlay')
    refine_overlay_dir = os.path.join(output_dir, f'refine_overlay_{args.mode}_{args.prompt_mode}')
    refine_inst_dir = os.path.join(output_dir, f'refine_inst_pred_{args.mode}_{args.prompt_mode}')
    save_flag = True

    # check if it is needed to compute accuracies
    eval_flag = True if label_dir else False

    img_names = os.listdir(img_dir)

    if save_flag:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(refine_overlay_dir):
            os.makedirs(refine_overlay_dir)
        if not os.path.exists(refine_inst_dir):
            os.makedirs(refine_inst_dir)

        inst_maps_folder = '{:s}/{:s}'.format(save_dir, 'inst_pred')
    for img_name in img_names:
        # load test image

        img_path = '{:s}/{:s}'.format(img_dir, img_name)
        name = os.path.splitext(img_name)[0]

        ########################### esenbling from multi-scale #################################
        if not os.path.exists(img_path) or not os.path.exists('{:s}/{}.npy'.format(inst_output_dir, name)):
            print(f'{name} not exists! Continue')
            continue
        img = Image.open(img_path)

        ## 试下做一次颜色变换
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        inst_maps = np.load('{:s}/{}.npy'.format(inst_maps_folder, name))
        pred_maps = np.load('{:s}/{}.npy'.format(inst_output_dir, name))

        pred_props = measure.regionprops(pred_maps)
        match_pred_maps = np.zeros_like(pred_maps)
        unknown_maps = np.zeros_like(match_pred_maps)
        avg_iou = []
        for i, pred_prop in enumerate(pred_props):
            pred_map = np.array(pred_maps == pred_prop.label, np.uint8)
            max_iou = -float('inf')
            for j in range(inst_maps.shape[0]):
                sam_inst_map = inst_maps[j]
                iou = iou_metrics(pred_map, sam_inst_map)
                if iou > max_iou:
                    max_iou = iou
            avg_iou.append(max_iou)
            if max_iou > args.iou_threshold:
                match_pred_maps[pred_maps == pred_prop.label] = pred_prop.label
            else:
                unknown_maps[pred_maps == pred_prop.label] = 1
        mean_iou = np.mean(np.array(avg_iou))
        print('=> Processing image {:s} -- Mean IoU: {:.2f}'.format(img_name, mean_iou))
        if mean_iou < args.threshold:
            continue
        match_pred_maps = measure.label(match_pred_maps)
        if eval_flag:
            ori_overlay = cv2.imread('{:s}/{:s}_seg.jpg'.format(overlay_output_dir, name))

            ## draw overlay
            overlay = draw_rand_inst_overlay(img, match_pred_maps, rand_color=False)
            result = np.concatenate([ori_overlay, overlay], axis=1)
            cv2.imwrite('{:s}/{:s}_seg.jpg'.format(refine_overlay_dir, name), result)

            ## save refine pred map
            match_pred_maps[unknown_maps == 1] = -1
            np.save('{:s}/{:s}.npy'.format(refine_inst_dir, name), match_pred_maps)


if __name__ == '__main__':
    main()

