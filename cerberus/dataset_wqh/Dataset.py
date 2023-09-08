# -*-coding:utf-8-*-
import json
import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage import color

def get_img_path(img_json_path, train):
    with open(img_json_path, 'r') as f:
        img_path_dict = json.load(f)
        f.close()
    img_path_list = img_path_dict['train'] if train is True else img_path_dict['eval']
    return img_path_list

def ColorDeconv(img: np.uint8):
    img = img / 255.0
    null = np.zeros_like(img[:, :, 0])
    img_hed = color.rgb2hed(img)
    img_h = color.hed2rgb(np.stack((img_hed[:, :, 0], null, null), axis=-1))
    return np.uint8(img_h * 255)

class SegDataset(Dataset):
    def __init__(self, img_json_path, train=True, data_augmentor=None, colortrans=False):
        self.img_path_list = get_img_path(img_json_path, train)
        self.data_augmentor = data_augmentor
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])
        self.colortrans = colortrans

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]['img_path']
        label_path = self.img_path_list[idx]['label_path']
        img = Image.open(img_path).convert('RGB')
        if os.path.exists(label_path):
            label = np.load(label_path)
        else:
            label = np.ones((img.size[0], img.size[1]), dtype=np.uint8) * 255
        if self.colortrans:
            img = np.array(img, dtype=np.uint8)
            img = ColorDeconv(img)

        if self.data_augmentor is not None:
            img, label = self.data_augmentor(img, label)
        img = self.transforms(img)
        label = np.uint8(label)
        label_ = label.copy()
        label_ = torch.from_numpy(label_)
        return {"img": img, "label": label_, "img_path": img_path, "label_path": label_path}

    def __len__(self):
        return len(self.img_path_list)
