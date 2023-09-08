"""
Author: my
Since: 2023-9-8
Modifier: wzh
"""
import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def img_loader(path, num_channels):
    if num_channels == 1:
        img = Image.open(path)
    else:
        img = Image.open(path).convert('RGB')

    return img


# get the image list pairs
def get_imgs_list(dir_list):
    """
    :param dir_list: [img1_dir, img2_dir, ...]
    :return: e.g. [(img1.ext, img1_label.png, img1_weight.png), ...]
    """
    img_list = []
    if len(dir_list) == 0:
        return img_list

    img_filename_list = [os.listdir(dir_list[i]) for i in range(len(dir_list))]
    for img in img_filename_list[0]:

        item = [os.path.join(dir_list[0], img), os.path.join(dir_list[1], img)]

        if len(item) == len(dir_list):
            img_list.append(tuple(item))

    return img_list


# dataset that supports one input image, one target image, and one weight map (optional)
class DataFolder(data.Dataset):
    def __init__(self, dir_list, data_transform=None, loader=img_loader):
        super(DataFolder, self).__init__()

        self.img_list = get_imgs_list(dir_list)
        if len(self.img_list) == 0:
            raise(RuntimeError('Found 0 image pairs in given directories.'))

        self.data_transform = data_transform
        self.loader = loader

    def __getitem__(self, index):
        img_paths = self.img_list[index]
        img = np.array(Image.open(img_paths[0]).convert('RGB'))
        label = np.array(Image.open(img_paths[1]))

        indices = np.unique(label)
        indices = indices[indices != 0]
        boundaries = [cv2.findContours((label == i).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0]
                      for i in indices]
        label_contour = np.zeros_like(label).astype(np.uint8)


        # for contour in range(len(boundaries)):
        #     img_contour = cv2.drawContours(label_contour, boundaries, contour, (1), 1)

        label_contour = cv2.drawContours(label_contour, boundaries, -1, (1), 1)
        kernel = np.ones((3, 3), np.uint8)
        label_contour = cv2.dilate(label_contour, kernel, iterations=1)

        cv2.imwrite('test.png', label_contour * 255)

        # if self.data_transform is not None:
        #     sample = self.data_transform(sample)

        return img, label, label_contour

    def __len__(self):
        return len(self.img_list)

