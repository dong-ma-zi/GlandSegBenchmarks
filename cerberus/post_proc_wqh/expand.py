# import os
#
# import cv2
# import time
# import pickle
# import numpy as np
# import openslide
# import pyclipper
# from skimage import morphology
# from PIL import Image
#
#
# # 提取组织区域
# def get_tissue_region(slide, mask_level=-1, mthresh=7, use_otsu=False, sthresh=20, sthresh_up=255, close=0):
#     if mask_level == -1:
#         mask_level = slide.get_best_level_for_downsample(64)
#     downsample = slide.level_downsamples[mask_level]
#     img = np.array(slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level]).convert('RGB'))
#     img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
#     img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)  # Apply median blurring
#
#     # Thresholding
#     if use_otsu:
#         _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
#     else:
#         _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)
#
#     # Morphological closing
#     if close > 0:
#         kernel = np.ones((close, close), np.uint8)
#         img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)
#
#     img_otsu = morphology.remove_small_objects(
#         img_otsu == 255, min_size=(200 // downsample) ** 2, connectivity=2
#     )
#     img_otsu = morphology.remove_small_holes(img_otsu, area_threshold=(400 // downsample) ** 2)
#
#     return img_otsu, mask_level
#
# def split_region(slide, mask, mask_downsample, mpp, margins, tumor_idx, stroma_idx, exclude_idx, out_interval, in_interval):
#     """
#     取得肿瘤区域以及其空洞
#     对边缘作外扩，对空洞区域作内缩
#     先填充外扩的区域，在填充空洞区域
#     """
#     start = time.time()
#
#     def filter_contours(contours, hierarchy, filter_params: dict):
#         """
#                         Filter contours by: area.
#                     """
#         filtered = []
#         # find indices of foreground contours (parent == -1)
#         hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
#         all_holes = []
#         # loop through foreground contour indices
#         for cont_idx in hierarchy_1:
#             # actual contour
#             cont = contours[cont_idx]
#             # indices of holes contained in this contour (children of parent contour)
#             holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
#             # take contour area (includes holes)
#             a = cv2.contourArea(cont)
#             # calculate the contour area of each hole
#             hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
#             # actual area of foreground contour region
#             a = a - np.array(hole_areas).sum()
#             if a == 0: continue
#             if tuple((filter_params['a_t'],)) < tuple((a,)):
#                 filtered.append(cont_idx)
#                 all_holes.append(holes)
#         foreground_contours = [contours[cont_idx] for cont_idx in filtered]
#         hole_contours = []
#         for hole_ids in all_holes:
#             unfiltered_holes = [contours[idx] for idx in hole_ids]
#             unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
#             # take max_n_holes largest holes by area
#             unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
#             filtered_holes = []
#             # filter these holes
#             for hole in unfilered_holes:
#                 if cv2.contourArea(hole) > filter_params['a_h']:
#                     filtered_holes.append(hole)
#             hole_contours.append(filtered_holes)
#         return foreground_contours, hole_contours
#
#     def equidistant_zoom_contour(contour, margin):
#         """
#         等距离缩放多边形轮廓点
#         :param contour: 一个图形的轮廓格式[[[x1, x2]],...],shape是(-1, 1, 2)
#         :param margin: 轮廓外扩的像素距离，margin正数是外扩，负数是缩小
#         :return: 外扩后的轮廓点
#         """
#         margin = int(margin)
#         pco = pyclipper.PyclipperOffset()
#         ##### 参数限制，默认成2这里设置大一些，主要是用于多边形的尖角是否用圆角代替
#         pco.MiterLimit = 100
#         contour = contour[:, 0, :]
#         pco.AddPath(contour, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
#         solutions = pco.Execute(margin)
#         contour = []
#         try:
#             for i in range(len(solutions)):
#                 contour.append(np.array(solutions[i]).reshape(-1, 1, 2).astype(int))
#         except:
#             contour = np.array([])
#         return contour
#
#     tissue_mask, tissue_mask_level = get_tissue_region(slide)
#     tissue_mask = cv2.resize(np.array(tissue_mask * 1, dtype=np.uint8), (mask.shape[1], mask.shape[0]),
#                              cv2.INTER_NEAREST)
#     tissue_mask = np.array(tissue_mask, dtype=bool)
#
#     # 去除掉不关注的区域
#     for idx in exclude_idx:
#         tissue_mask[mask == idx] = False
#
#     # 获取肿瘤区域
#     mask_tumor = np.uint8((mask == tumor_idx) * 255)
#
#     # Morphological closing
#     kernel = np.ones((21, 21), np.uint8)
#     mask_tumor = cv2.morphologyEx(mask_tumor, cv2.MORPH_CLOSE, kernel)
#     mask_tumor = cv2.GaussianBlur(mask_tumor, (45, 45), 0)
#     mask_tumor = (mask_tumor > 128) * 255
#     # 过滤掉比较小的区域
#     mask_tumor = morphology.remove_small_objects(mask_tumor == 255,
#                                                  min_size=(256 / mask_downsample) ** 2, connectivity=2)
#     mask_tumor = morphology.remove_small_holes(mask_tumor, area_threshold=(256 / mask_downsample) ** 2)
#     mask_tumor = np.array(mask_tumor * 255, dtype=np.uint8)
#
#     # 获取肿瘤边界
#     contours, hierarchy = cv2.findContours(mask_tumor, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#
#     if not isinstance(hierarchy, np.ndarray):
#         return None, None
#
#     hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
#     filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 100}
#     foreground_contours, hole_contours = filter_contours(contours, hierarchy, filter_params)
#
#     # 获取最外围的margin，来确定基质区域
#     max_margin = 0
#     for margin in margins:
#         if max_margin < margin:
#             max_margin = margin
#     # 外扩与内缩的距离，来获取不同区域的mask
#     mask_stroma_regions = {}
#     for margin in margins:
#         mask_stroma_regions[margin] = np.zeros_like(mask_tumor, dtype=np.uint8)
#         for i in range(len(foreground_contours)):
#             contour = equidistant_zoom_contour(foreground_contours[i],
#                                                margin=(margin / mpp / mask_downsample))
#             try:
#                 cv2.drawContours(mask_stroma_regions[margin], contour, -1, 255, -1)
#             except:
#                 pass
#         # 对于空洞区域进行内缩
#         for i in range(len(hole_contours)):
#             if hole_contours[i] == []:
#                 continue
#             for j in range(len(hole_contours[i])):
#                 contour = equidistant_zoom_contour(hole_contours[i][j],
#                                                    margin=-(margin / mpp / mask_downsample))
#                 try:
#                     cv2.drawContours(mask_stroma_regions[margin], contour, -1, 0, -1)
#                 except:
#                     pass
#         # 获得不同层次的区域mask
#         mask_stroma_regions[margin] = np.logical_xor(mask_stroma_regions[margin], mask_tumor)
#         # 获得组织区域内的mask
#         mask_stroma_regions[margin] = np.logical_and(mask_stroma_regions[margin], tissue_mask)
#         mask_stroma_regions[margin] = np.array(mask_stroma_regions[margin] * 255, dtype=np.uint8)
#
#         if margin == max_margin:
#             mask_stroma_regions['基质区域'] = mask_stroma_regions[margin]
#
#     # TODO:目前只能做等间距的外扩与内缩
#     margins.sort(reverse=True)
#     out_margins = margins.copy()
#     out_margins = [x for x in out_margins if x > 0]
#     out_margins = out_margins[:-1]
#     for margin in out_margins:
#         mask_stroma_regions[margin] = np.array(
#             np.logical_xor(mask_stroma_regions[margin], mask_stroma_regions[margin - out_interval]) * 255,
#             dtype=np.uint8)
#
#     margins.sort()
#     in_margins = margins.copy()
#     in_margins = [x for x in in_margins if x < 0]
#     in_margins = in_margins[:-1]
#     for margin in in_margins:
#         mask_stroma_regions[margin] = np.array(
#             np.logical_xor(mask_stroma_regions[margin + in_interval], mask_stroma_regions[margin]) * 255,
#             dtype=np.uint8)
#
#     print(f"划分不同的区域耗费了{time.time() - start:.3f}秒！")
#     return mask_tumor, mask_stroma_regions
#
#
# def expand(slide, mask, heatmap_downsample,
#            margins, tumor_idx, stroma_idx,
#            exclude_idx, color_list,
#            out_interval=200, in_interval=100):
#     """
#     :param slide: WSI对象
#     :param mask: 结果变量
#     :param margins: list type, 外扩的范围
#     :param tumor_idx: int type, 肿瘤索引
#     :param stroma_idx: int type， 基质索引
#     :param exclude_idx: list[int] type, 无关类索引列表
#     :param out_interval: int, 外扩的间隔，
#     :param in_interval: int, 内缩的间隔
#     :param save_path: 保存的路径
#     :param vis: 是否要可视化
#     :return:
#     """
#
#     region = mask
#     downsample = heatmap_downsample
#     try:
#         mpp = float(slide.properties['openslide.mpp-x'])
#     except:
#         mpp = 0.24
#
#     mask_tumor, mask_stroma_regions = split_region(slide, region, downsample, mpp, margins, tumor_idx,
#                                                    stroma_idx, exclude_idx, out_interval, in_interval)
#     if not isinstance(mask_tumor, np.ndarray):
#         return None
#
#     """可视化部分"""
#     # if vis:
#     level = slide.get_best_level_for_downsample(downsample)
#     assert downsample == slide.level_downsamples[level]
#     slide_image = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB'))
#     img_to_draw = np.zeros_like(slide_image)
#     margins.sort()
#     for i, margin in enumerate(margins):
#         mask = mask_stroma_regions[margin]
#         img_to_draw[mask == 255, :] = color_list[i]
#
#     return img_to_draw
#
#
# def expand_stroma(slide, mask, heatmap_downsample):
#     color_list = [
#         [255, 0, 0],
#         [255, 192, 203],
#         [255, 69, 0],
#         [255, 165, 0],
#         [255, 215, 0],
#         [0, 128, 0],
#         [0, 128, 128],
#         [0, 255, 255],
#         [0, 139, 139],
#         [0, 0, 255],
#         [75, 0, 130],
#         [128, 0, 128],
#         [148, 0, 211],
#         [255, 255, 0],
#         [154, 205, 50]
#     ]
#     margins = [2000, 1800, 1600, 1400, 1200, 1000, 800, 600, 400, 200, -100, -200, -300, -400, -500]
#     exclude_idx = [3, 4]
#     stroma_idx = 1
#     tumor_idx = 2
#     out_interval = 200
#     in_interval = 100
#
#     return expand(slide=slide,
#            mask=mask,
#            heatmap_downsample=heatmap_downsample,
#            margins=margins,
#            tumor_idx=tumor_idx,
#            stroma_idx=stroma_idx,
#            exclude_idx=exclude_idx,
#            color_list=color_list,
#            out_interval=out_interval,
#            in_interval=in_interval)

import os

import cv2
import time
import pickle
import numpy as np
import openslide
import pyclipper
from skimage import morphology
from PIL import Image


# 提取组织区域
def get_tissue_region(slide, mask_level=-1, mthresh=7, use_otsu=False, sthresh=20, sthresh_up=255, close=0):
    if mask_level == -1:
        mask_level = slide.get_best_level_for_downsample(64)
    downsample = slide.level_downsamples[mask_level]
    img = np.array(slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level]).convert('RGB'))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)  # Apply median blurring

    # Thresholding
    if use_otsu:
        _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    else:
        _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

    # Morphological closing
    if close > 0:
        kernel = np.ones((close, close), np.uint8)
        img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

    img_otsu = morphology.remove_small_objects(
        img_otsu == 255, min_size=(200 // downsample) ** 2, connectivity=2
    )
    img_otsu = morphology.remove_small_holes(img_otsu, area_threshold=(400 // downsample) ** 2)

    return img_otsu, mask_level

def split_region(slide, mask, mask_downsample, mpp, margins, tumor_idx, stroma_idx, exclude_idx, out_interval, in_interval):
    """
    取得肿瘤区域以及其空洞
    对边缘作外扩，对空洞区域作内缩
    先填充外扩的区域，在填充空洞区域
    """
    start = time.time()

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

    def equidistant_zoom_contour(contour, margin):
        """
        等距离缩放多边形轮廓点
        :param contour: 一个图形的轮廓格式[[[x1, x2]],...],shape是(-1, 1, 2)
        :param margin: 轮廓外扩的像素距离，margin正数是外扩，负数是缩小
        :return: 外扩后的轮廓点
        """
        margin = int(margin)
        pco = pyclipper.PyclipperOffset()
        ##### 参数限制，默认成2这里设置大一些，主要是用于多边形的尖角是否用圆角代替
        pco.MiterLimit = 100
        contour = contour[:, 0, :]
        pco.AddPath(contour, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
        solutions = pco.Execute(margin)
        contour = []
        try:
            for i in range(len(solutions)):
                contour.append(np.array(solutions[i]).reshape(-1, 1, 2).astype(int))
        except:
            contour = np.array([])
        return contour

    # 获取肿瘤区域
    mask_tumor = np.uint8((mask == tumor_idx) * 255)
    mask_tumor_copy = mask_tumor.copy()
    mask_stroma = np.uint8((mask == stroma_idx) * 255)

    # Morphological closing
    kernel = np.ones((21, 21), np.uint8)
    mask_tumor = cv2.morphologyEx(mask_tumor, cv2.MORPH_CLOSE, kernel)
    mask_tumor = cv2.GaussianBlur(mask_tumor, (45, 45), 0)
    mask_tumor = (mask_tumor > 128) * 255
    # 过滤掉比较小的区域
    mask_tumor = morphology.remove_small_objects(mask_tumor == 255,
                                                 min_size=(256 / mask_downsample) ** 2, connectivity=2)
    mask_tumor = morphology.remove_small_holes(mask_tumor, area_threshold=(256 / mask_downsample) ** 2)
    mask_tumor = np.array(mask_tumor * 255, dtype=np.uint8)

    # 获取肿瘤边界
    contours, hierarchy = cv2.findContours(mask_tumor, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if not isinstance(hierarchy, np.ndarray):
        return None, None

    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
    filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 100}
    foreground_contours, hole_contours = filter_contours(contours, hierarchy, filter_params)

    # 获取最外围的margin，来确定基质区域
    max_margin = 0
    for margin in margins:
        if max_margin < margin:
            max_margin = margin
    # 外扩与内缩的距离，来获取不同区域的mask
    mask_stroma_regions = {}
    for margin in margins:
        mask_stroma_regions[margin] = np.zeros_like(mask_tumor, dtype=np.uint8)
        for i in range(len(foreground_contours)):
            contour = equidistant_zoom_contour(foreground_contours[i],
                                               margin=(margin / mpp / mask_downsample))
            try:
                cv2.drawContours(mask_stroma_regions[margin], contour, -1, 255, -1)
            except:
                pass
        # 对于空洞区域进行内缩
        for i in range(len(hole_contours)):
            if hole_contours[i] == []:
                continue
            for j in range(len(hole_contours[i])):
                contour = equidistant_zoom_contour(hole_contours[i][j],
                                                   margin=-(margin / mpp / mask_downsample))
                try:
                    cv2.drawContours(mask_stroma_regions[margin], contour, -1, 0, -1)
                except:
                    pass

        if margin > 0:
            mask_stroma_regions[margin] = np.logical_and(mask_stroma_regions[margin], mask_stroma)
        else:
            mask_stroma_regions[margin] = np.logical_xor(mask_stroma_regions[margin], mask_tumor_copy)
            mask_stroma_regions[margin] = np.logical_and(mask_stroma_regions[margin], mask_tumor_copy)

        mask_stroma_regions[margin] = np.array(mask_stroma_regions[margin] * 255, dtype=np.uint8)

        if margin == max_margin:
            mask_stroma_regions['基质区域'] = mask_stroma_regions[margin]

    # TODO:目前只能做等间距的外扩与内缩
    margins.sort(reverse=True)
    out_margins = margins.copy()
    out_margins = [x for x in out_margins if x > 0]
    out_margins = out_margins[:-1]
    for margin in out_margins:
        mask_stroma_regions[margin] = np.array(
            np.logical_xor(mask_stroma_regions[margin], mask_stroma_regions[margin - out_interval]) * 255,
            dtype=np.uint8)

    margins.sort()
    in_margins = margins.copy()
    in_margins = [x for x in in_margins if x < 0]
    in_margins = in_margins[:-1]
    for margin in in_margins:
        mask_stroma_regions[margin] = np.array(
            np.logical_xor(mask_stroma_regions[margin + in_interval], mask_stroma_regions[margin]) * 255,
            dtype=np.uint8)

    print(f"划分不同的区域耗费了{time.time() - start:.3f}秒！")
    return mask_tumor, mask_stroma_regions


def expand(slide, mask, heatmap_downsample,
           margins, tumor_idx, stroma_idx,
           exclude_idx, color_list,
           out_interval=200, in_interval=100, vis=False):
    """
    :param slide: WSI对象
    :param mask: 结果变量
    :param margins: list type, 外扩的范围
    :param tumor_idx: int type, 肿瘤索引
    :param stroma_idx: int type， 基质索引
    :param exclude_idx: list[int] type, 无关类索引列表
    :param out_interval: int, 外扩的间隔，
    :param in_interval: int, 内缩的间隔
    :param save_path: 保存的路径
    :param vis: 是否要可视化
    :return:
    """

    region = mask
    downsample = heatmap_downsample
    try:
        mpp = float(slide.properties['openslide.mpp-x'])
    except:
        mpp = 0.24

    mask_tumor, mask_stroma_regions = split_region(slide, region, downsample, mpp, margins, tumor_idx,
                                                   stroma_idx, exclude_idx, out_interval, in_interval)

    if not isinstance(mask_tumor, np.ndarray):
        return None

    """可视化部分"""
    # if vis:
    level = slide.get_best_level_for_downsample(downsample)
    assert downsample == slide.level_downsamples[level]
    slide_image = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB'))
    img_to_draw = np.zeros_like(slide_image)
    margins.sort(reverse=True)
    for i, margin in enumerate(margins):
        mask = mask_stroma_regions[margin]
        img_to_draw[mask == 255, :] = color_list[i]
    if vis:
        Image.fromarray(img_to_draw.astype(np.uint8)).show()
        image = 0.5 * slide_image + 0.5 * img_to_draw
        Image.fromarray(image.astype(np.uint8)).show()
    return img_to_draw


def expand_stroma(slide, mask, heatmap_downsample, vis=False):
    color_list = [
        [255, 255, 255],
        [0, 128, 0],
        [0, 255, 0],
        [192, 192, 192],
        [128, 128, 128],
        [128, 0, 128],
        [0, 128, 128],
        [0, 255, 255],
        [0, 0, 128],
        [0, 0, 255],
        [255, 215, 0],
        [139, 0, 0],
        [255, 0, 255],
        [255, 128, 0],
        [255, 0, 0],
    ]

    margins = [2000, 1800, 1600, 1400, 1200, 1000, 800, 600, 400, 200, -100, -200, -300, -400, -500]
    exclude_idx = [3, 4]
    stroma_idx = 1
    tumor_idx = 2
    out_interval = 200
    in_interval = 100

    return expand(slide=slide,
           mask=mask,
           heatmap_downsample=heatmap_downsample,
           margins=margins,
           tumor_idx=tumor_idx,
           stroma_idx=stroma_idx,
           exclude_idx=exclude_idx,
           color_list=color_list,
           out_interval=out_interval,
           in_interval=in_interval,
           vis=vis)

if __name__ == '__main__':
    slide_path = "433375.qptiff"
    mask_file = "../results/433375_seg(1).pkl"
    with open(mask_file, 'rb') as f:
        file = pickle.load(f)
        f.close()
    colormap = file['mask']
    heatmap_downsample = file['heatmap_downsample']
    slide = openslide.open_slide(slide_path)
    mask = np.zeros((colormap.shape[0], colormap.shape[1]))
    colors = [(0, 0, 255), [255, 0, 0], [0, 255, 0], [255, 255, 255]]
    for i, color in enumerate(colors):
        indices = np.where(np.all(colormap == color, axis=-1))
        mask[indices] = i+1
    img_to_draw = expand_stroma(slide, mask, heatmap_downsample, True)
    file['hierarchy_mask'] = img_to_draw
    with open(mask_file, 'wb') as f:
        pickle.dump(file, f)
        f.close()

