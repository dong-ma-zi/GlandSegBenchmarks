import glob
import os
import scipy.io as sio
import cv2
import numpy as np

dataset = 'MoNuSeg' #
save_dir = f'{dataset}_SAM_vit-h_vis'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

root_dir = '/home/data1/my/Project/GlandSegBenchmark/SAM/experimentsP3'
anything_model = f'{dataset}_SAM-vit-h-Anything-mode'
slec_1_point_dir = f'{dataset}_SAM-vit-h-1-point'
slec_20_points_dir = f'{dataset}_SAM-vit-h-20-points'
total_points_dir = f'{dataset}_SAM-vit-h-total-points'
total_boxes_dir = f'{dataset}_SAM-vit-h-total-boxes'
vis_list = [slec_1_point_dir, slec_20_points_dir, total_points_dir, total_boxes_dir, anything_model]

img_dir = '/home/data2/MedImg/NucleiSeg/MoNuSeg/Test/Images'
anno_dir = '/home/data2/MedImg/NucleiSeg/MoNuSeg/Test/Annotation'

# img_dir = '/home/data2/MedImg/GlandSeg/GlaS/test/Images'
# anno_dir = '/home/data2/MedImg/GlandSeg/GlaS/test/Annotation'

img_list = glob.glob(img_dir + '/*')
img_list = [os.path.basename(x).split('.')[0] for x in img_list]

for img_name in img_list:
    img = cv2.imread(os.path.join(img_dir, img_name + '.tif'))
    img = img[512:, 512:, :]
    blank = np.zeros(shape=(5, img.shape[1], 3))

    label = sio.loadmat(os.path.join(anno_dir, img_name + '.mat'))['inst_map'][512:, 512:]
    # label = cv2.imread((os.path.join(anno_dir, img_name + '_anno.bmp')))[:, :, 0]
    mask = np.copy(img)
    mask[label != 0] = (255, 144, 30)
    vis_img = np.concatenate([img, blank, mask], axis=0)
    # cv2.imwrite(os.path.join(save_dir, img_name + '.png'), vis_img)

    for vis_dir in vis_list:
        label = np.load(os.path.join(root_dir, vis_dir, 'mask_pred', img_name + '.npy'))[512:, 512:]
        mask = np.copy(img)
        mask[label != 0] = (255, 144, 30)
        vis_img = np.concatenate([vis_img, blank, mask], axis=0)
    cv2.imwrite(os.path.join(save_dir, img_name + '_3.png'), vis_img)