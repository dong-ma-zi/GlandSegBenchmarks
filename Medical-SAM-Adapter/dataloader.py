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
import torchvision.transforms as transforms
import scipy.io as scio

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

transforms = transforms.Compose([transforms.ToTensor()])


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
        # item = [os.path.join(dir_list[0], img), os.path.join(dir_list[1], img.split('.')[0] + '.bmp')]
        # image_name = img.split('.')[0]
        # item = [os.path.join(dir_list[0], image_name + '.tif'), os.path.join(dir_list[1], image_name + '.mat')]

        if len(item) == len(dir_list):
            img_list.append(tuple(item))

    return img_list


# dataset that supports one input image, one target image, and one weight map (optional)
class DataFolder(data.Dataset):
    def __init__(self, dir_list, data_transform=transforms, loader=img_loader):
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
        # label = scio.loadmat(img_paths[1])['inst_map']
        # label = np.array(label != 0, dtype=np.int32)

        if self.data_transform is not None:
            sample = self.data_transform([Image.fromarray(img), Image.fromarray(label)])

        return sample[0], sample[1]

    def __len__(self):
        return len(self.img_list)

