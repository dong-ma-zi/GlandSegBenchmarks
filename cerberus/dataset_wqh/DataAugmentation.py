# -*-coding:utf-8-*-
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
import random
from PIL import Image

class SegmentationDataAugmentation:
    def __init__(self, p_flip=0.5, p_scale=0.5, p_rotate=0.5,
                 brightness=0.2, contrast=0.2, saturation=0.2,
                 noise_std=0.005, img_size=256):
        self.p_flip = p_flip
        self.p_scale = p_scale
        self.p_rotate = p_rotate
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.noise_std = noise_std
        self.img_size = img_size

    def __call__(self, img, target):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(np.uint8(img))
        if isinstance(target, np.ndarray):
            target = Image.fromarray(np.uint8(target))

        # Random horizontal flip
        if random.random() < self.p_flip:
            img = F.hflip(img)
            target = F.hflip(target)

        # Random vertical flip
        if random.random() < self.p_flip:
            img = F.vflip(img)
            target = F.vflip(target)

        # Random scale
        if random.random() < self.p_scale:
            scale_factor = random.uniform(0.5, 2.0)
            new_size = [int(round(dim * scale_factor)) for dim in img.size]
            img = F.resize(img, new_size, interpolation=InterpolationMode.BILINEAR)
            target = F.resize(target, new_size, interpolation=InterpolationMode.NEAREST)

        # Random rotation
        if random.random() < self.p_rotate:
            angle = random.randint(-45, 45)
            img = F.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, fill=[255, 255, 255])
            target = F.rotate(target, angle, interpolation=InterpolationMode.NEAREST)

        # Random brightness, contrast, and saturation
        img = F.adjust_brightness(img, random.uniform(1-self.brightness, 1+self.brightness))
        img = F.adjust_contrast(img, random.uniform(1-self.contrast, 1+self.contrast))
        img = F.adjust_saturation(img, random.uniform(1-self.saturation, 1+self.saturation))

        # Add noise
        noise = torch.randn(img.size) * self.noise_std
        img = F.to_tensor(img)
        img += noise
        img = torch.clip(img, 0, 1)
        img = F.to_pil_image(img)

        # Center crop
        w, h = img.size
        if w > self.img_size and h > self.img_size:
            i = (h - self.img_size) // 2
            j = (w - self.img_size) // 2
            img = F.crop(img, i, j, self.img_size, self.img_size)
            target = F.crop(target, i, j, self.img_size, self.img_size)

        # Resize to 256x256
        img = F.resize(img, (self.img_size, self.img_size), interpolation=InterpolationMode.BILINEAR)
        target = F.resize(target, (self.img_size, self.img_size), interpolation=InterpolationMode.NEAREST)
        return img, target

if __name__ == '__main__':
    data_augmentator = SegmentationDataAugmentation()
    img_path = "/mnt/wqh/MY_CODE/Dataset_Zhujiang_Region/Images/2127579-20p/12288_35840.png"
    label_path = "/mnt/wqh/MY_CODE/Dataset_Zhujiang_Region/Annotations/2127579-20p/12288_35840.npy"
    img = Image.open(img_path)
    label = np.load(label_path)
    img.save('img.png')
    Image.fromarray(label*255).save('label.png')
    img, label = data_augmentator(img, label)
    img.save('img1.png')
    Image.fromarray(np.uint8(label)).save('label1.png')
