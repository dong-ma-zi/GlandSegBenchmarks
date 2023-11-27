# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    my
"""
import glob

import torch

from conf import settings
from utils import *
import function_boxes_prop
# from dataloader import DataFolder
# from my_transforms import get_transforms
# from torch.utils.data import DataLoader

args = cfg.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in [args.gpu_device])
GPUdevice = torch.device('cuda', args.gpu_device)
'''load and load pretrained model'''
net = get_network(args, args.net, vit_mode='vit_b', gpu_device=GPUdevice)

# load pretrained weights
# net.load_state_dict(torch.load("/home/data1/my/Project/GlandSegBenchmark/Medical-SAM-Adapter/"
#                                "logs_p/monuseg-samOrig-b-1024-16-256_2023_10_29_20_07/Model/"
#                                "checkpoint_50.pth")['state_dict'])

# n_list = [n for n, _ in net.named_parameters()]
for n, value in net.image_encoder.named_parameters():
    if "Adapter" not in n:
        value.requires_grad = False
    else:
        print('training para: ', n)

# # n_list = [n for n, _ in net.named_parameters()]
# for n, value in net.named_parameters():
#     if "mask_decoder" not in n:
#         value.requires_grad = False
#     else:
#         print('training para: ', n)


optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad == True],
                       lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer = optim.SGD([p for p in net.parameters() if p.requires_grad == True],
#                        lr=args.lr, momentum=0.9, weight_decay=0)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay


args.path_helper = set_log_dir('logs_boxes_mod1107', args.exp_name)
logger = create_logger(args.path_helper['log_path'])
logger.info(args)



# ----- load data ----- #

# data_path = {'train': '/home/data2/MedImg/GlandSeg/GlaS/train',
#              'val': '/home/data2/MedImg/GlandSeg/GlaS/test_proc'}

# data_path = {'train': '/home/data2/MedImg/NucleiSeg/MoNuSeg/Train',
#              'val': '/home/data2/MedImg/NucleiSeg/MoNuSeg/Test'}

# data_path = {'train': '/home/data1/my/dataset/consep/Train',
#              'val': '/home/data1/my/dataset/consep/Test'}

data_path = {'train': '/home/data1/my/dataset/monusac/Train',
             'val': '/home/data1/my/dataset/monusac/Test'}

train_img_list = sorted(glob.glob(os.path.join(data_path['train'], 'Images/*')))
train_anno_list = []
for i in train_img_list:
    img_name = os.path.basename(i).split('.')[0]
    # train_anno_list += [os.path.join(data_path['train'], 'Annotation', img_name + '_anno.bmp')]
    # train_anno_list += [os.path.join(data_path['train'], 'Annotation', img_name + '.mat')]
    train_anno_list += [os.path.join(data_path['train'], 'Labels', img_name + '.mat')]

val_img_list = sorted(glob.glob(os.path.join(data_path['val'], 'Images/*')))
val_anno_list = []
for i in val_img_list:
    img_name = os.path.basename(i).split('.')[0]
    # val_anno_list += [os.path.join(data_path['val'], 'Annotation', img_name + '_anno.bmp')]
    # val_anno_list += [os.path.join(data_path['val'], 'Annotation', img_name + '.mat')]
    val_anno_list += [os.path.join(data_path['val'], 'Labels', img_name + '.mat')]

# '''checkpoint path and tensorboard'''
# # iter_per_epoch = len(Glaucoma_training_loader)
# checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
#
# # create checkpoint folder to save model
# if not os.path.exists(checkpoint_path):
#     os.makedirs(checkpoint_path)
# checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

'''begain training'''
scaler = torch.cuda.amp.GradScaler()
best_acc = 0.0
best_tol = 1e4
for epoch in range(1, settings.EPOCH):
    # if args.mod == 'sam_adpt':
    net.train()
    time_start = time.time()
    loss = function_boxes_prop.train_sam(args, net, optimizer,
                                         train_img_list,
                                         train_anno_list,
                                         epoch)
    logger.info(f'Train loss: {loss}|| @ epoch {epoch}.')
    time_end = time.time()
    print('time_for_training ', time_end - time_start)

    net.eval()
    if epoch and epoch % args.val_freq == 0 or epoch == settings.EPOCH - 1:
        tol, eacc, eiou = function_boxes_prop.validation_sam(args,
                                                             net,
                                                             val_img_list,
                                                             val_anno_list
                                                             )
        logger.info(f'Total score: {tol}, ACC: {eacc}, IOU: {eiou} || @ epoch {epoch}.')

        if args.distributed != 'none':
            sd = net.module.state_dict()
        else:
            sd = net.state_dict()

        if tol < best_tol:
            best_tol = tol
            is_best = True

            save_checkpoint({
            'epoch': epoch,
            'model': args.net,
            'state_dict': sd,
            'optimizer': optimizer.state_dict(),
            'best_tol': best_tol,
            'path_helper': args.path_helper,
        }, is_best, args.path_helper['ckpt_path'], filename=f"checkpoint_{epoch}.pth")
        else:
            is_best = False
            save_checkpoint({
                'epoch': epoch,
                'model': args.net,
                'state_dict': sd,
                'optimizer': optimizer.state_dict(),
                'best_tol': best_tol,
                'path_helper': args.path_helper,
            }, is_best, args.path_helper['ckpt_path'], filename=f"checkpoint_{epoch}.pth")
