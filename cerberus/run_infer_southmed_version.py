# -*-coding:utf-8-*-
import os
import sys
import glob
import time

import numpy as np
import tqdm
import argparse
import pickle
import threading
import openslide
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import models
from dataset_wqh.WholeSlideSet import WholeSlideSet

def filter_contours(contours, hierarchy, filter_params: dict):
    """
                    Filter contours by: area.
                """
    filtered = []

    # find indices of foreground contours (parent == -1)
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
    all_holes = []

    # loop through foreground contour indices
    for cont_idx in hierarchy_1:
        # actual contour
        cont = contours[cont_idx]
        # indices of holes contained in this contour (children of parent contour)
        holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
        # take contour area (includes holes)
        a = cv2.contourArea(cont)
        # calculate the contour area of each hole
        hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
        # actual area of foreground contour region
        a = a - np.array(hole_areas).sum()
        if a == 0: continue
        if tuple((filter_params['a_t'],)) < tuple((a,)):
            filtered.append(cont_idx)
            all_holes.append(holes)

    foreground_contours = [contours[cont_idx] for cont_idx in filtered]

    hole_contours = []

    for hole_ids in all_holes:
        unfiltered_holes = [contours[idx] for idx in hole_ids]
        unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
        # take max_n_holes largest holes by area
        unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
        filtered_holes = []

        # filter these holes
        for hole in unfilered_holes:
            if cv2.contourArea(hole) > filter_params['a_h']:
                filtered_holes.append(hole)

        hole_contours.append(filtered_holes)

    return foreground_contours, hole_contours

def extract_contour(mask, downsample, color_list, num_classes):
    tissue_contours = []
    colors = []
    types = []
    for class_idx in range(1, num_classes):
        sub_mask = mask.copy()
        sub_mask[sub_mask != class_idx] = 0
        try:
            contours, hierarchy = cv2.findContours(np.uint8(sub_mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
            filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 100}
            foreground_contours, hole_contours = filter_contours(contours, hierarchy, filter_params)
        except:
            continue
        for j in range(len(foreground_contours)):
            if cv2.contourArea(foreground_contours[j]) < (256 / downsample) ** 2:
                continue
            # 轮廓近似
            foreground_contours[j] = cv2.approxPolyDP(foreground_contours[j], 1, True)
            foreground_contours[j] *= downsample
            colors.append(color_list[class_idx])
            tissue_contours.append(np.squeeze(foreground_contours[j]))
            types.append(class_idx)
        for j in range(len(hole_contours)):
            if hole_contours[j]:
                for k in range(len(hole_contours[j])):
                    if cv2.contourArea(hole_contours[j][k]) < (256 / downsample) ** 2:
                        continue
                    hole_contours[j][k] *= downsample
                    colors.append(color_list[class_idx])
                    tissue_contours.append(np.squeeze(hole_contours[j][k]))
                    types.append(class_idx)

    return tissue_contours, colors, types

@torch.no_grad()
def Segment_Tumor(model: torch.nn.Module,
                  num_classes: int,
                  slide: openslide.OpenSlide,
                  slide_name: str,
                  # heatmap_level: int,
                  slide_format: str,
                  batch_size: int,
                  dataset_num_workers: int,
                  post_proc_num_workers: int,
                  patch_size: int = 512,
                  stride: int = 416,
                  patch_downsample: int = 1,
                  mask_level: int = -1,
                  tissue_mask_threshold: float = 0,
                  transforms=None,
                  # device="cuda",
                  results_dir='results'):
    # model.eval()
    start = time.time()
    dataset = WholeSlideSet(slide=slide,
                            patch_size=patch_size,
                            stride=stride,
                            patch_downsample=patch_downsample,
                            mask_level=mask_level,
                            tissue_mask_threshold=tissue_mask_threshold,
                            transforms=transforms,
                            colortrans=args.colortrans)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=dataset_num_workers)

    # 计算heatmap的缩放倍数，heatmap的shape
    heatmap_level = slide.get_best_level_for_downsample(16)

    if slide_format == '.svs' and len(slide.level_downsamples) > heatmap_level + 1:
        heatmap_level += 1
    heatmap_downsample = slide.level_downsamples[heatmap_level]
    heatmap_dimension = slide.level_dimensions[heatmap_level]
    zoomout = int(heatmap_downsample/patch_downsample)

    if zoomout <= 1:
        print(f'can not find correct downsample {slide_name}')
        return

    mask = torch.zeros((num_classes, heatmap_dimension[1], heatmap_dimension[0]), dtype=torch.float16)
    # weight = torch.zeros((heatmap_dimension[1], heatmap_dimension[0]), dtype=torch.int8)

    def post_process(preds, coordinates_str):
        for i in range(preds.shape[0]):
            pred = preds[i]
            coordinate = [int(int(p)/heatmap_downsample) for p in coordinates_str[i].split('_')]
            mask[:, coordinate[1]: coordinate[1] + preds.shape[2],
                 coordinate[0]: coordinate[0] + preds.shape[2]] += pred.cpu()
            # weight[coordinate[1]: coordinate[1] + preds.shape[2],
            #      coordinate[0]: coordinate[0] + preds.shape[2]] += torch.ones((preds.shape[2], preds.shape[2]), dtype=torch.int8)


    # --------------------------- set cerberus model --------------------------- #
    checkpoint_path = "/home/data1/my/Project/GlandSegBenchmark/cerberus/pretrained_weights/resnet34_cerberus/weights.tar"
    import yaml
    with open("/home/data1/my/Project/GlandSegBenchmark/cerberus/pretrained_weights/resnet34_cerberus/settings.yml") as fptr:
        run_paramset = yaml.full_load(fptr)
    from infer.tile_my_version import InferManager
    infer = InferManager(
        checkpoint_path=checkpoint_path,
        decoder_dict=run_paramset["dataset_kwargs"]["req_target_code"],
        model_args=run_paramset["model_kwargs"],
    )
    # -------------------------------------------------------------------------- #

    dataloader = tqdm.tqdm(dataloader, file=sys.stdout)
    for idx, data in enumerate(dataloader):
        # img = data["patch"].to(device)
        coordinate_str = data["coordination"]

        ##### TODO: replace the inference code with cerberus model ######
        preds = cerberus_segmentation(infer, data["patch"][0])
        preds[preds > 0] = 1
        # preds = model(img)
        #################################################################
        # preds = torch.softmax(preds, dim=1)
        # preds = F.interpolate(preds, (int(patch_size/zoomout), int(patch_size/zoomout)), mode='bilinear')
        preds = torch.tensor(preds).unsqueeze(0).type(torch.int64)
        preds = F.one_hot(preds, num_classes=2).type(torch.float32)
        preds = preds.permute(0, 3, 1, 2)
        preds = F.interpolate(preds, (int(patch_size/zoomout), int(patch_size/zoomout)), mode='nearest')

        t = threading.Thread(target=post_process, args=(preds, coordinate_str))
        t.start()
        if threading.active_count() > post_proc_num_workers:
            t.join()
        if idx > len(dataloader) - 5:
            t.join()


    mask = torch.argmax(mask, dim=0).numpy().astype("int8")
    print(f"肿瘤区域分割耗费了{time.time()-start:.3f}秒！")

    color_list = [[0, 0, 0], [0, 0, 255]]
    contour, color, region_class = extract_contour(mask,
                                                   int(heatmap_downsample),
                                                   color_list, num_classes)


    # hierarchy_mask = expand_stroma(slide, mask, heatmap_downsample)
    # if not isinstance(hierarchy_mask, np.ndarray):
    #     print(f'no tumor region {slide_name}, skip!')

    color_map = np.zeros((*mask.shape, 3), dtype=np.uint8)
    # for i in range(1, 2):
    color_map[mask == 1, :] = [0, 0, 255]

    # 存储结果
    start = time.time()
    os.makedirs(f"{results_dir}/{slide_name}", exist_ok=True)

    with open(f"{results_dir}/{slide_name}/{slide_name}_seg.pkl", 'wb') as f:
        pickle.dump({"mask": color_map,
                     "heatmap_downsample": heatmap_downsample,
                     'region_contours': contour,
                     'region_colors': color,
                     'region_types': region_class
                     # 'hierarchy_mask': hierarchy_mask
                     }, f)
        f.close()
    print(f"存储耗费了{time.time() - start:.3f}秒！")
    return

def cerberus_segmentation(infer, img):
    target_list = ['gland'] # ['gland', 'lumen', 'nuclei', 'patch-class']
    run_args = {
        "img": img,
        "nr_inference_workers": 0,
        "nr_post_proc_workers": 0,
        "batch_size": 10,
        "patch_input_shape": 448,
        "patch_output_shape": 144,
        "patch_output_overlap": 0,
        "postproc_list": target_list,
    }

    return infer.process_file_list(run_args)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Res50Unet')
    parser.add_argument('--pretrained', type=str, default='SWAV_es')
    parser.add_argument('--num_classes', type=int, default=2)
    # parser.add_argument('--weight_path', type=str,
    #                     # default="/home/data1/my/Project/SYSTME/SegmenTumor/save_weights/Unet-512-v0630-2023-07-01-17-02/Unet_epoch41_acc0.890_miou0.705.pth")
    #                     # default="/home/data1/my/Project/SYSTME/SegmenTumor/save_weights/Res50Unet-None-512-v0630-2023-07-05-10-09/Res50Unet_epoch39_acc0.897_miou0.723.pth")
    #                     # default="/home/data1/my/Project/SYSTME/SegmenTumor/save_weights/Res50Unet-MoCo-512-v0630-2023-07-05-11-57/Res50Unet_epoch29_acc0.913_miou0.743.pth",
    #                     default="/home/data1/my/Project/SYSTME/SegmenTumor/save_weights/Res50Unet-SWAV-512-v0630-2023-07-07-17-12/Res50Unet_epoch12_acc0.929_miou0.759.pth")
    parser.add_argument('--slide_path', type=str, default='')
    parser.add_argument('--slide_dir', type=str, default="/home/data2/MedImg/TCGA-COAD/Tumor/*/*.svs")
    parser.add_argument('--heatmap_level', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--post_proc_num_workers', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=1024)
    parser.add_argument('--patch_downsample', type=int, default=2)
    parser.add_argument('--mask_level', type=int, default=-1)
    parser.add_argument('--tissue_mask_threshold', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--colortrans', type=bool, default=False)
    parser.add_argument('--dataset_version', type=str, default='v0820')
    parser.add_argument('--results_dir', type=str, default='./TCGA-COAD/')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    args.results_dir = os.path.join(args.results_dir, args.model_name + '-' + args.pretrained + '-' + args.dataset_version)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    # device = torch.device(args.device)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    slide_paths = sorted(glob.glob(args.slide_dir))
    # slide_paths = ["/home/data1/wzh/dataset/SYMH/wsi/SYMH/433160.qptiff"]

    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                      std=[0.229, 0.224, 0.225])])

    for slide_path in slide_paths:
        slide_name, slide_format = os.path.splitext(os.path.basename(slide_path))
        # skip the processed slides
        if os.path.exists(os.path.join(args.results_dir, slide_name)):
            print(slide_name, 'has already inferrenced, skip!')
            continue

        print(f'processing {slide_path}')
        slide = openslide.open_slide(slide_path)
        Segment_Tumor(model=None,
                      num_classes=args.num_classes,
                      slide=slide,
                      slide_name=slide_name,
                      slide_format=slide_format,
                      batch_size=args.batch_size,
                      dataset_num_workers=args.num_workers,
                      post_proc_num_workers=args.post_proc_num_workers,
                      patch_size=args.patch_size,
                      stride=args.stride,
                      patch_downsample=args.patch_downsample,
                      mask_level=args.mask_level,
                      tissue_mask_threshold=args.tissue_mask_threshold,
                      transforms=None, # transform,
                      # device=args.device,
                      results_dir=args.results_dir)


