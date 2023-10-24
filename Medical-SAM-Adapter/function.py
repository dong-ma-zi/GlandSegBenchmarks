from conf import settings
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import torch
from transforms import ResizeLongestSide
args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# seed = torch.randint(1,11,(args.b,7))
torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []

def img_preprocessing(image, sam):
    original_image_size = (image.size[1], image.size[0])
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    input_image = transform.apply_image(np.array(image))
    input_image = torch.as_tensor(input_image, device=sam.device)
    input_image = input_image.permute(2, 0, 1).contiguous()[None, :, :, :]
    input_size = input_image.shape[-2:]
    input_image = sam.preprocess(input_image)
    return input_image, original_image_size, input_size

def train_sam(args, net: nn.Module, optimizer,
              train_loader,
              epoch):

    # train mode
    net.train()
    optimizer.zero_grad()

    epoch_loss = 0
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    lossfunc = criterion_G

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:

        for sample in train_loader:
            img, label = sample
            imgs = img.to(dtype=torch.float32, device=GPUdevice)
            masks = label.to(dtype=torch.float32, device=GPUdevice)
            imgs, pt, masks = generate_click_prompt_all_inst(imgs, masks)

            mask_type = torch.float32

            point_coords = pt
            labels_torch = torch.as_tensor(np.ones(shape=(point_coords.shape[0], point_coords.shape[1], )), dtype=torch.int, device=GPUdevice)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
            # coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            pt = (coords_torch, labels_torch)

            '''init'''
            imgs = imgs.to(dtype=mask_type, device = GPUdevice)
            
            '''Train'''
            _, _, original_h, original_w = imgs.shape

            imgs = F.interpolate(
                imgs,
                (net.image_encoder.img_size, net.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )

            imge= net.image_encoder(imgs)
            with torch.no_grad():
                se, de = net.prompt_encoder(
                    points=pt,
                    boxes=None,
                    masks=None)

            pred, _ = net.mask_decoder(
                image_embeddings=imge,
                image_pe=net.prompt_encoder.get_dense_pe(), 
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de, 
                multimask_output=False)

            pred = F.interpolate(
                pred,
                (original_h, original_w),
                mode="bilinear",
                align_corners=False,
            )

            loss = lossfunc(pred, masks)
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            pbar.update()

    return loss

def validation_sam(args, val_loader, net: nn.Module):
     # eval mode
    net.eval()

    mask_type = torch.float32
    n_val = len(val_loader)  # the number of batch
    ave_res, mix_res = (0,0,0,0), (0,0,0,0)
    tot = 0
    hard = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))


    lossfunc = criterion_G

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for _, sample in enumerate(val_loader):
            img, label = sample
            imgsw = img.to(dtype=torch.float32, device=GPUdevice)
            masksw = label.to(dtype=torch.float32, device=GPUdevice)
            imgsw, ptw, masksw = generate_click_prompt(imgsw, masksw)

            # pt = ptw
            imgs = imgsw
            masks = masksw
            mask_type = torch.float32

            point_coords = ptw
            labels_torch = torch.as_tensor(np.ones(shape=(point_coords.shape[0], point_coords.shape[1],)),
                                           dtype=torch.int, device=GPUdevice)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
            # coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            pt = (coords_torch, labels_torch)

            '''init'''
            imgs = imgs.to(dtype = mask_type,device = GPUdevice)

            '''test'''
            _, _, original_h, original_w = imgs.shape
            imgs = F.interpolate(
                imgs,
                (net.image_encoder.img_size, net.image_encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )

            with torch.no_grad():
                imge= net.image_encoder(imgs)
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
                pred = F.interpolate(
                    pred,
                    (original_h, original_w),
                    mode="bilinear",
                    align_corners=False,
                )

                tot += lossfunc(pred, masks)
                temp = eval_seg(pred, masks, threshold)
                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            pbar.update()

    return tot/ n_val , tuple([a/n_val for a in mix_res])
