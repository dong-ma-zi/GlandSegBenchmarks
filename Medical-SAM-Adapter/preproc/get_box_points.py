"""
split the patch 2 diff classes
"""
import copy
import glob
import os
import numpy as np
import cv2
import tqdm
import scipy.io as scio



if __name__ == '__main__':

    # x = scio.loadmat("/home/data2/MedImg/NucleiSeg/MoNuSeg/Test/Prompts/TCGA-2Z-A9J9-01A-01-TS1.mat")

    patch_paths = os.listdir(f'/home/data2/MedImg/NucleiSeg/MoNuSeg/Test/Images/')
    # patch_paths = os.listdir('/home/data1/my/dataset/consep/Test/Images/')
    # monusac
    # save_dir = f'/home/data2/MedImg/NucleiSeg/MoNuSeg/Test/Prompts'

    save_dir = '/home/data2/MedImg/NucleiSeg/MoNuSeg/Test/'
    basePath = '/home/data2/MedImg/NucleiSeg/MoNuSeg/Test/'

    # save_dir = '/home/data1/my/dataset/consep/Test/'
    # basePath = '/home/data1/my/dataset/consep/Test/'

    os.makedirs(save_dir, exist_ok=True)

    center_rad = 1
    contour_th = 1
    os.makedirs(os.path.join(save_dir, f'Centers_{center_rad}_{contour_th}'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, f'Contours_{center_rad}_{contour_th}'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, f'Label_cr_{center_rad}_ct_{contour_th}'), exist_ok=True)

    for patch_path in tqdm.tqdm(patch_paths):
        # monusac
        save_name = patch_path.split('.')[0]

        img = cv2.imread(os.path.join(basePath, 'Images', save_name + '.tif'))
        inst_map = scio.loadmat(os.path.join(basePath, 'Annotation', save_name + '.mat'))["inst_map"]
        # cls_map = scio.loadmat(os.path.join(basePath, 'Labels', save_name + '.mat'))["type_map"]

        # draw contour
        # img_center = np.zeros(shape=(448, 448)).astype(np.uint8)
        # img_contour = np.zeros(shape=(448, 448)).astype(np.uint8)
        img_center = np.zeros(shape=(img.shape[0], img.shape[1])).astype(np.uint8)
        img_contour = np.zeros(shape=(img.shape[0], img.shape[1])).astype(np.uint8)

        # get the nuclei instance
        insts = np.unique(inst_map).tolist()
        insts.remove(0)
        boundaries = []
        for i in insts:
            boundaries += \
            cv2.findContours((inst_map == i).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

        point_list = []
        box_list = []
        for boundary in boundaries:
            M = cv2.moments(boundary)
            try:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
            except:
                continue
            point_list += [np.array([center_x, center_y])[np.newaxis, :]]

            # minx = np.min(boundary[:, :, 0])
            # miny = np.min(boundary[:, :, 1])
            # maxx = np.max(boundary[:, :, 0])
            # maxy = np.max(boundary[:, :, 1])
            # box_list += [np.array([minx, miny, maxx, maxy])[np.newaxis, :]]

            img_center[center_y, center_x] = 1

        img_contour = cv2.drawContours(img_contour, boundaries, -1, (1), contour_th)
        scio.savemat(os.path.join(save_dir, f'Label_cr_{center_rad}_ct_{contour_th}', save_name + ".mat"),
                     {'inst_map': inst_map,
                      # 'cls_map': cls_map,
                      'center_map': img_center,
                      'contour_map': img_contour})

        img_center[img_center != 0] = 255
        img_contour[img_contour != 0] = 255
        cv2.imwrite(os.path.join(save_dir, f'Centers_{center_rad}_{contour_th}', save_name + '.png'), img_center)
        cv2.imwrite(os.path.join(save_dir, f'Contours_{center_rad}_{contour_th}', save_name + '.png'), img_contour)

        # prompt_dict = {'points': np.concatenate(point_list, axis=0),
        #                'boxes': np.concatenate(box_list, axis=0)}
        # scio.savemat(os.path.join(save_dir, save_name + '.mat'), prompt_dict)


# if __name__ == '__main__':
#
#     patch_paths = os.listdir(f'/home/data2/MedImg/GlandSeg/GlaS/my/train/448x448/Images')
#     # monusac
#     # save_dir = f'/home/data2/MedImg/GlandSeg/GlaS/test/Prompts'
#     save_dir = f'/home/data2/MedImg/GlandSeg/GlaS/my/train/448x448/Prompts'
#     # os.makedirs(save_dir, exist_ok=True)
#
#     for patch_path in tqdm.tqdm(patch_paths):
#         # monusac
#         save_name = patch_path.split('.')[0]
#         # basePath = '/home/data2/MedImg/GlandSeg/GlaS/test/'
#         basePath = '/home/data2/MedImg/GlandSeg/GlaS/my/train/448x448/'
#         img = cv2.imread(os.path.join(basePath, 'Images', save_name + '.bmp'))
#
#         inst_map = cv2.imread(os.path.join(basePath, 'Annotation', save_name + '.bmp'))[:, :, 0]
#
#         insts = np.unique(inst_map).tolist()
#
#         # get the nuclei instance
#         insts = np.unique(inst_map).tolist()
#         insts.remove(0)
#         boundaries = []
#         for i in insts:
#             boundaries += \
#             cv2.findContours((inst_map == i).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
#
#         point_list = []
#         box_list = []
#         for boundary in boundaries:
#             M = cv2.moments(boundary)
#             try:
#                 center_x = int(M["m10"] / M["m00"])
#                 center_y = int(M["m01"] / M["m00"])
#             except:
#                 continue
#             point_list += [np.array([center_x, center_y])[np.newaxis, :]]
#
#             minx = np.min(boundary[:, :, 0])
#             miny = np.min(boundary[:, :, 1])
#             maxx = np.max(boundary[:, :, 0])
#             maxy = np.max(boundary[:, :, 1])
#             box_list += [np.array([minx, miny, maxx, maxy])[np.newaxis, :]]
#
#
#         prompt_dict = {'points': np.concatenate(point_list, axis=0),
#                        'boxes': np.concatenate(box_list, axis=0)}
#
#         scio.savemat(os.path.join(save_dir, save_name + '.mat'), prompt_dict)

