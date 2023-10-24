# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Junde Wu
"""

# from dataset import *
from conf import settings
from utils import *
import function 
from dataloader import DataFolder
from my_transforms import get_transforms
from torch.utils.data import DataLoader

args = cfg.parse_args()
GPUdevice = torch.device('cuda', args.gpu_device)
'''load and load pretrained model'''
net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)

for n, value in net.image_encoder.named_parameters():
    if "Adapter" not in n and "pos_embed" not in n and "patch_embed" not in n:
        value.requires_grad = False

# '''load pretrained model'''
# if args.pretrain:
#     weights = torch.load(args.pretrain)
#     net.load_state_dict(weights,strict=False)

optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad == True],
                       lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay


args.path_helper = set_log_dir('logs', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)


# ----- define augmentation ----- #
# data_transforms = {
#     'train': get_transforms({
#     'horizontal_flip': True,
#     'vertical_flip': True,
#     # 'random_elastic': [6, 15],
#     'random_rotation': 90,
#     'to_tensor': 1,
#     # 'normalize': [[0.787, 0.511, 0.785], [0.167, 0.248, 0.131]],
# }),
#     'val': get_transforms({
#     'to_tensor': 1,
#     # 'normalize': [[0.787, 0.511, 0.785], [0.167, 0.248, 0.131]],
# })}

# ----- load data ----- #
data_path = {'train': '/home/data2/MedImg/GlandSeg/GlaS/my/train/448x448/',
             'val': '/home/data2/MedImg/GlandSeg/GlaS/my/valid/448x448/'}

# data_path = {'train': '/home/data2/MedImg/NucleiSeg/MoNuSeg/extracted_mirror/train/512x512_256x256/',
#              'val': '/home/data2/MedImg/NucleiSeg/MoNuSeg/Test'}

dsets = {}
for x in ['train', 'val']:
    img_dir = os.path.join(data_path[x], 'Images')
    target_dir = os.path.join(data_path[x], 'Annotation')

    dir_list = [img_dir, target_dir]
    dsets[x] = DataFolder(dir_list # , data_transform=data_transforms[x]
                          )

train_loader = DataLoader(dsets['train'], batch_size=1, shuffle=True,
                          num_workers=4)
val_loader = DataLoader(dsets['val'], batch_size=1, shuffle=False,
                        num_workers=4)

'''checkpoint path and tensorboard'''
# iter_per_epoch = len(Glaucoma_training_loader)
checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

# create checkpoint folder to save model
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

'''begain training'''
best_acc = 0.0
best_tol = 1e4
for epoch in range(settings.EPOCH):
    if args.mod == 'sam_adpt':
        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, train_loader, epoch)
        logger.info(f'Train loss: {loss}|| @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        net.eval()
        if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH - 1:
            tol, (eiou, edice) = function.validation_sam(args, val_loader, net)
            logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()

            if tol < best_tol:
                best_tol = tol
                is_best = True

                save_checkpoint({
                'epoch': epoch + 1,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': best_tol,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename=f"best_checkpoint_{epoch}")
            else:
                is_best = False
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': args.net,
                    'state_dict': sd,
                    'optimizer': optimizer.state_dict(),
                    'best_tol': best_tol,
                    'path_helper': args.path_helper,
                }, is_best, args.path_helper['ckpt_path'], filename=f"checkpoint_{epoch}")
