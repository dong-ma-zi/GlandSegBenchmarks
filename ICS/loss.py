import torch
from skimage import measure
import numpy as np
from PIL import Image

####
def dice_loss(true, pred, smooth=1e-3):
    """`pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC."""
    inse = torch.sum(pred * true)
    l = torch.sum(pred)
    r = torch.sum(true)
    loss = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    loss = torch.sum(loss)
    return loss

def overlap(true, pred):
    return np.sum(np.where(true>0, 1, 0) * np.where(pred>0, 1, 0))


def object_dice_loss(true, pred_ori):
    loss = 0

    pred = torch.gt(pred_ori, 0.5)

    predMatchMap = {}
    trueMatchMap = {}
    true_inst = np.array(true)
    pred_inst = measure.label(pred)
    pred_props = measure.regionprops(pred_inst)
    true_props = measure.regionprops(true_inst)
    for i, pred_prop in enumerate(pred_props):
        max_overlap = -float('inf')
        pred_map = np.array(pred_inst == pred_prop.label, np.uint8)
        for j, true_prop in enumerate(true_props):
            true_map = np.array(true_inst == true_prop.label, np.uint8)
            temp = overlap(true_map, pred_map)
            if temp > max_overlap:
                predMatchMap[i] = j
                max_overlap = temp

    for i, true_prop in enumerate(true_props):
        max_overlap = -float('inf')
        true_map = np.array(true_inst == true_prop.label, np.uint8)
        for j, pred_prop in enumerate(pred_props):
            pred_map = np.array(pred_inst == pred_prop.label, np.uint8)
            temp = overlap(true_map, pred_map)
            if temp > max_overlap:
                trueMatchMap[i] = j
                max_overlap = temp

    pred_loss = 0
    pred_all_slice = torch.gt(pred, 0)
    for i, pred_prop in enumerate(pred_props):
        pred_map = torch.from_numpy(np.array(pred_inst == pred_prop.label, np.uint8))
        true_map = torch.from_numpy(np.array(true_inst == true_props[predMatchMap[i]].label, np.uint8))
        pred_slice = pred_ori * (pred == pred_map)
        pred_loss += dice_loss(true_map, pred_slice) * torch.sum(pred_slice) / torch.sum(pred_all_slice)

    true_loss = 0
    true_all_slice = torch.gt(pred, 0)
    for i, true_prop in enumerate(true_props):
        true_map = torch.from_numpy(np.array(true_inst == true_prop.label, np.uint8))
        pred_map = torch.from_numpy(np.array(pred_inst == pred_props[trueMatchMap[i]].label, np.uint8))
        true_slice = true == true_map
        pred_slice = pred_ori * (pred == pred_map)
        true_loss += dice_loss(true_slice, pred_slice) * torch.sum(true_slice) / torch.sum(true_all_slice)
    return (pred_loss + true_loss) / 2


if __name__ == '__main__':
    anno_path = "/home/data2/MedImg/GlandSeg/GlaS/test/Annotation/testA_10_anno.bmp"
    anno = torch.from_numpy(np.array(Image.open(anno_path)))

    # pred_path = "/home/data2/MedImg/GlandSeg/GlaS/test/Annotation/testA_11_anno.bmp"
    # pred = torch.from_numpy(np.array(Image.open(pred_path)))
    size = anno.size()
    pred = torch.zeros(size)
    pred[10:20, 10:20] = 0.95
    pred[30:40, 30:40] = 0.8

    obj_dice_loss = object_dice_loss(anno, pred)
    print('Object Dice Loss: ', obj_dice_loss)









