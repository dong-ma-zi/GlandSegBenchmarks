import os
import random

import numpy as np
import scipy.io as scio
from conf import settings
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import torch
from transforms import ResizeLongestSide
args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
criterion_G = torch.nn.BCEWithLogitsLoss()
torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []


def get_scaled_prompt(boxes, sam, original_image_size, if_transform: bool = True):
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    # points = transform.apply_coords(points, original_image_size) if if_transform else points
    # points = torch.as_tensor(points).unsqueeze(1)
    # points = (points, torch.ones(points.shape[0], 1))

    boxes = transform.apply_boxes(boxes, original_image_size) if if_transform else boxes
    boxes = torch.as_tensor(boxes, device=sam.device)
    return boxes



def img_preprocessing(image, sam):
    original_image_size = (image.size[1], image.size[0])
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    input_image = transform.apply_image(np.array(image))
    input_image = torch.as_tensor(input_image, dtype=torch.float32, device=sam.device) # set to float32
    input_image = input_image.permute(2, 0, 1).contiguous()[None, :, :, :]
    input_size = input_image.shape[-2:]
    input_image = sam.preprocess(input_image) # do not need padding here
    return input_image, original_image_size, input_size


def train_sam(args, net: nn.Module, optimizer,
              train_img_list,
              train_anno_list,
              epoch,
              shuffle=True):

    # train mode
    net.train()
    boxes_batch_size = 8
    optimizer.zero_grad()

    # epoch_loss = 0
    image_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    lossfunc = criterion_G

    index = list(range(len(train_img_list)))
    # shuffle
    if shuffle:
        random.shuffle(index)

    with tqdm(total=len(index), desc=f'Epoch {epoch}', unit='img') as pbar:

        for ind in index:
            img = Image.open(train_img_list[ind]).convert('RGB')
            # label = np.array(Image.open(train_anno_list[ind]))
            label = scio.loadmat(train_anno_list[ind])['inst_map']

            ''' preprocess '''
            input_image, original_image_size, input_size = img_preprocessing(img, net)
            boxes_orig_scale, masks = generate_boxes_prompt_all_inst(label)
            boxes = get_scaled_prompt(boxes_orig_scale, net, original_image_size)
            masks = torch.as_tensor(masks, dtype=torch.float32)

            # with torch.cuda.amp.autocast(enabled=scaler is not None):

            # for i in range(pts[0].shape[0]): input pt prop with batch
            box_num =boxes.shape[0]
            if box_num % boxes_batch_size == 0:
                batch_num = box_num // boxes_batch_size
            else:
                batch_num = box_num // boxes_batch_size + 1

            for i in range(batch_num):
                if i != (box_num // boxes_batch_size):
                    single_boxes = boxes[i * boxes_batch_size: (i + 1) * boxes_batch_size].to(GPUdevice)

                    batch_masks = masks[i * boxes_batch_size: (i + 1) * boxes_batch_size].to(GPUdevice)
                else:
                    single_boxes = boxes[i * boxes_batch_size:].to(GPUdevice)
                    batch_masks = masks[i * boxes_batch_size:].to(GPUdevice)

                if single_boxes[0].shape[0] == 0: print('no prompt input')
                ''' forward '''
                imge= net.image_encoder(input_image)

                with torch.no_grad():
                    se, de = net.prompt_encoder(
                        points=None, # pts,
                        boxes= single_boxes,
                        masks=None)

                pred, _ = net.mask_decoder(
                    image_embeddings=imge,
                    image_pe=net.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=False)

                ''' postprocess '''
                upscaled_masks = net.postprocess_masks(pred, input_size, original_image_size).squeeze(1)
                # upscaled_masks = torch.nn.functional.sigmoid(upscaled_masks).

                ''' bp '''
                # x = masks[i].unsqueeze(0)
                loss = lossfunc(upscaled_masks, batch_masks)
                image_loss += loss.item()

                optimizer.zero_grad()
                # if scaler is not None:
                #     scaler.scale(loss).backward()
                #     scaler.step(optimizer)
                #     scaler.update()
                # else:
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

            pbar.set_postfix(**{'loss (batch)': image_loss / batch_num})
            # epoch_loss += image_loss
            pbar.update()

    return loss


def validation_sam(args, net: nn.Module,
                   val_img_list,
                   val_anno_list):
    boxes_batch_size = 16

    batch_tot, batch_acc, batch_iou = 0, 0, 0
    tot, tot_acc, tot_iou = 0, 0, 0
    net.eval()

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    lossfunc = criterion_G
    index = list(range(len(val_img_list)))
    with tqdm(total=len(index), desc='Validation round', unit='batch') as pbar:
        for ind in index:
            img = Image.open(val_img_list[ind]).convert('RGB')
            # label = np.array(Image.open(val_anno_list[ind]))
            label = scio.loadmat(val_anno_list[ind])['inst_map']

            ''' preprocess '''
            input_image, original_image_size, input_size = img_preprocessing(img, net)
            boxes_orig_scale, masks = generate_boxes_prompt_all_inst(label)
            boxes = get_scaled_prompt(boxes_orig_scale, net, original_image_size)
            masks = torch.as_tensor(masks, dtype=torch.float32)

            # for i in range(pts[0].shape[0]): input pt prop with batch
            box_num =boxes.shape[0]
            if box_num % boxes_batch_size == 0:
                batch_num = box_num // boxes_batch_size
            else:
                batch_num = box_num // boxes_batch_size + 1

            for i in range(batch_num):
                if i != (box_num // boxes_batch_size):
                    single_boxes = boxes[i * boxes_batch_size: (i + 1) * boxes_batch_size].to(GPUdevice)

                    batch_masks = masks[i * boxes_batch_size: (i + 1) * boxes_batch_size].to(GPUdevice)
                else:
                    single_boxes = boxes[i * boxes_batch_size:].to(GPUdevice)
                    batch_masks = masks[i * boxes_batch_size:].to(GPUdevice)

                if single_boxes.shape[0] == 0: print('no prompt input')

                '''test'''
                with torch.no_grad():
                    imge= net.image_encoder(input_image)
                    se, de = net.prompt_encoder(
                        points=None,
                        boxes=single_boxes,
                        masks=None,
                    )

                    pred, _ = net.mask_decoder(
                        image_embeddings=imge,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,
                    )

                    ''' postprocess '''
                    upscaled_masks = net.postprocess_masks(pred, input_size, original_image_size).squeeze(1)

                    batch_tot += lossfunc(upscaled_masks, batch_masks)

                    upscaled_masks = upscaled_masks.detach().cpu().numpy()
                    upscaled_masks = np.array(upscaled_masks > 0.5, dtype=np.int32)
                    metrics = accuracy_pixel_level(upscaled_masks,
                                                   batch_masks.detach().cpu().numpy())
                    pixel_accu = metrics[0]
                    iou = metrics[1]
                    batch_acc += pixel_accu
                    batch_iou += iou

            tot += batch_tot / batch_num
            tot_acc += batch_acc / batch_num
            tot_iou += batch_iou / batch_num
            torch.cuda.empty_cache()
            del input_image, boxes
            pbar.update()

    return tot/ len(index), tot_acc / len(index), tot_iou / len(index)
