import os.path
import numpy as np
import cv2
import glob
import skimage.morphology as morph
from skimage.measure import label
from scipy import ndimage

ckdir = '/home/data1/my/Project/GlandSegBenchmark/OEEM/my_segmentation/glas_seg/'
# os.makedirs('/home/data1/my/Project/GlandSegBenchmark/OEEM/my_segmentation/glas_seg/label_proc/')
img_list = glob.glob('/home/data1/my/Project/GlandSegBenchmark/OEEM/my_segmentation/glas_seg/img_train_256_192/*')
for img_path in img_list:
    img_name = os.path.basename(img_path).split('.')[0]
    img = cv2.imread(img_path)
    label_imag = cv2.imread(os.path.join('/home/data1/my/Project/GlandSegBenchmark/OEEM/my_segmentation/glas_seg/pseudo_train_256_192/',
                                    img_name + '.png'), -1)
    label_imag_raw = np.logical_not(label_imag).astype(np.uint8)

    # 将二值图像转换为标签图像
    label_image = label(label_imag_raw)
    # 根据标记删除小连通域
    filtered_label_image = morph.remove_small_objects(label_image, min_size=300)
    # 将标签图像转换回二值图像
    label_proc = (filtered_label_image > 0).astype(np.uint8)
    # smooth the contour label_prociction
    # label_proc = ndimage.binary_opening(label_proc, structure=morph.disk(5))
    # label_proc = ndimage.binary_closing(label_proc, structure=morph.disk(5))
    # label_proc = ndimage.binary_opening(label_proc, structure=morph.disk(4))
    # label_proc = ndimage.binary_closing(label_proc, structure=morph.disk(4))
    # fill holes
    label_proc = ndimage.binary_fill_holes(label_proc)
    label_proc_save = np.logical_not(label_proc).astype(np.int32)

    cv2.imwrite(f'/home/data1/my/Project/GlandSegBenchmark/OEEM/my_segmentation/glas_seg/pseudo_train_256_192_proc/{img_name}.png',
                label_proc_save)

    label_rgb = np.stack((label_imag_raw * 255, label_imag_raw * 255, label_imag_raw * 255), axis=-1)
    label_proc_rgb = np.stack((label_proc * 255, label_proc * 255, label_proc * 255), axis=-1)
    img_show = np.concatenate([img, label_rgb, label_proc_rgb], axis=1)
    cv2.imwrite(f'/home/data1/my/Project/GlandSegBenchmark/OEEM/my_segmentation/glas_seg/overlay/{img_name}.png',
                img_show)