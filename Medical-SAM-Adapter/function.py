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

def get_scaled_prompt(points, sam, original_image_size, if_transform: bool = True):
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    points = transform.apply_coords(points, original_image_size) if if_transform else points
    points = torch.as_tensor(points, device=sam.device).unsqueeze(1)
    points = (points, torch.ones(points.shape[0], 1))
    return points

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
              shuffle=True,
              scaler=None):

    # train mode
    net.train()
    optimizer.zero_grad()

    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    lossfunc = criterion_G

    index = list(range(len(train_img_list)))
    # shuffle
    if shuffle:
        random.shuffle(index)

    with tqdm(total=len(index), desc=f'Epoch {epoch}', unit='img') as pbar:

        for ind in index:
            img = Image.open(train_img_list[ind])
            # label = np.array(Image.open(train_anno_list[ind]))
            label = scio.loadmat(train_anno_list[ind])['inst_map']

            ''' preprocess '''
            input_image, original_image_size, input_size = img_preprocessing(img, net)
            pts, masks = generate_click_prompt_all_inst(label)
            pts = get_scaled_prompt(pts, net, original_image_size)
            masks = torch.as_tensor(masks, dtype=torch.float32, device=GPUdevice)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                ''' forward '''
                imge= net.image_encoder(input_image)

                with torch.no_grad():
                    se, de = net.prompt_encoder(
                        points=pts,
                        boxes=None,
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
                loss = lossfunc(upscaled_masks, masks)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                epoch_loss += loss.item()

            # loss.backward()
            # if ((ind + 1) % 8) == 0:
            #     optimizer.step()  # 反向传播，更新网络参数
            #     optimizer.zero_grad()  # 清空梯度

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            torch.cuda.empty_cache()
            # optimizer.step()
            # optimizer.zero_grad()
            #

            pbar.update()

    return loss

def validation_sam(args, net: nn.Module,
                   val_img_list,
                   val_anno_list):
    tot, tot_acc, tot_iou = 0, 0, 0
    net.eval()

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    lossfunc = criterion_G
    index = list(range(len(val_img_list)))
    with tqdm(total=len(index), desc='Validation round', unit='batch') as pbar:
        for ind in index:
            img = Image.open(val_img_list[ind])
            # label = np.array(Image.open(val_anno_list[ind]))
            label = scio.loadmat(val_anno_list[ind])['inst_map']

            ''' preprocess '''
            input_image, original_image_size, input_size = img_preprocessing(img, net)
            pts, masks = generate_click_prompt_all_inst(label)
            pt = get_scaled_prompt(pts, net, original_image_size)
            masks = torch.as_tensor(masks, dtype=torch.float32, device=GPUdevice)

            '''test'''
            with torch.no_grad():
                imge= net.image_encoder(input_image)
                se, de = net.prompt_encoder(
                    points=pt,
                    boxes=None,
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

                tot += lossfunc(upscaled_masks, masks)

                upscaled_masks = upscaled_masks.detach().cpu().numpy()
                upscaled_masks = np.array(upscaled_masks > 0.5, dtype=np.int32)
                metrics = accuracy_pixel_level(upscaled_masks, masks.detach().cpu().numpy())
                pixel_accu = metrics[0]
                iou = metrics[1]
                tot_acc += pixel_accu
                tot_iou += iou

            torch.cuda.empty_cache()
            pbar.update()

    return tot/ len(index), tot_acc / len(index), tot_iou / len(index)
