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

        padding_size = 32
        label_padding = cv2.copyMakeBorder(label, padding_size,
                                           padding_size, padding_size, padding_size, cv2.BORDER_REFLECT)
        indices = np.unique(label_padding)
        indices = indices[indices != 0]
        # boundaries = [cv2.findContours((label_padding == i).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        #               for i in indices]

        boundaries = []
        for i in indices:
            boundaries += cv2.findContours((label_padding == i).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]


        label_contour_with_padding = np.zeros_like(label_padding).astype(np.uint8)

        # for contour in range(len(boundaries)):
        #     img_contour = cv2.drawContours(label_contour, boundaries, contour, (1), 1)
        label_contour_with_padding = cv2.drawContours(label_contour_with_padding, boundaries, -1, (1), 1)

        label_contour = label_contour_with_padding[padding_size:-padding_size,
                                                   padding_size:-padding_size]

        kernel = np.ones((3, 3), np.uint8)
        label_contour = cv2.dilate(label_contour, kernel, iterations=1)

        # resized_label_padding = cv2.resize(label_padding, (480, 480), interpolation= cv2.INTER_NEAREST)
        # # resized_contour_padding = cv2.resize(label_contour_with_padding, (480, 480), interpolation=cv2.INTER_AREA)
        # label_show = np.array(label != 0, dtype=np.int32)
        # data_show = np.concatenate([label_show * 255,
        #                             resized_label_padding * 255,
        #                             label_contour * 255],
        #                            axis=1)
        #
        # cv2.imwrite(f'test/test_{index}.png', data_show)

        # if self.data_transform is not None:
        #     sample = self.data_transform(sample)

        # img = self.data_transform(img)
        #cv2.imwrite('test_contour_before.jpg', label_contour * 255)
        if self.data_transform is not None:
            sample = self.data_transform([Image.fromarray(img), Image.fromarray(label), Image.fromarray(label_contour)])
        # test_img = np.array(sample[0] * 255, np.uint8)
        # test_img = np.transpose(test_img, (1, 2, 0))
        # cv2.imwrite('test_img.jpg', test_img)
        # test_label = np.array(sample[1].numpy()>0, np.uint8).reshape(480, 480, 1) * 255
        # cv2.imwrite('test_label.jpg', test_label)
        # test_contour = np.array(sample[2].numpy()>0, np.uint8).reshape(480, 480, 1) * 255
        # cv2.imwrite('test_contour.jpg', test_contour)
        return sample[0], sample[1], sample[2]

    def __len__(self):
        return len(self.img_list)

