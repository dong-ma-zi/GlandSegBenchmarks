import argparse
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import dataset
from torch.utils.data import DataLoader
from utils.metric import get_overall_valid_score
from utils.generate_CAM import generate_validation_cam
from utils.pyutils import crop_validation_images
# from utils.torchutils import PolyOptimizer
from torch.optim import Adam
from models.DWS_MIL import DWS_MIL
import yaml
import importlib

import logging
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', default=32, type=int)
    parser.add_argument('-epoch', default=500, type=int)
    parser.add_argument('-lr', default=0.001, type=float)
    parser.add_argument('-test_every', default=5, type=int, help="how often to test a model while training")
    parser.add_argument('-device', default=[0], type=list)
    parser.add_argument('-m', default='vgg_0925', type=str)
    args = parser.parse_args()

    save_dir = os.path.join('./exp', args.m)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    global logger
    import datetime
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    logger = get_logger(os.path.join(save_dir, f'train_log_{cur_time}.txt'))
    logger.info(args)

    batch_size = args.batch
    epochs = args.epoch
    base_lr = args.lr
    test_every = args.test_every
    devices = args.device
    model_name = args.m

    with open('configuration.yml') as f:
        config = yaml.safe_load(f)
    mean = config['mean']
    std = config['std']
    network_image_size = config['network_image_size']
    scales = config['scales']

    if not os.path.exists('weights'):
        os.mkdir('weights')
    if not os.path.exists('result'):
        os.mkdir('result')



    net = DWS_MIL(n_class=2).cuda()
    net.load_state_dict(torch.load("/home/data1/wzh/code/GlandSegBenchmarks/DCAN/DCAN_pretrained_weight.pth"), strict=False)
    #$ net = torch.nn.DataParallel(net, device_ids=devices).cuda()

    # data augmentation
    train_transform = transforms.Compose([
            # transforms.RandomResizedCrop(size=network_image_size, scale=(0.7, 1), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=mean, std=std)
    ])

    # load training dataset
    data_path_name = f'../OEEM/classification/glas_cls/1.training/img/'
    TrainDataset = dataset.OriginPatchesDataset(data_path_name=data_path_name, transform=train_transform)
    print("train Dataset", len(TrainDataset))
    TrainDatasampler = torch.utils.data.RandomSampler(TrainDataset)
    TrainDataloader = DataLoader(TrainDataset, batch_size=batch_size, num_workers=4,
                                 sampler=TrainDatasampler, drop_last=True)

    # optimizer and loss
    # optimizer = PolyOptimizer(net.parameters(), base_lr, weight_decay=1e-4, max_step=epochs, momentum=0.9)
    optimizer = Adam(net.parameters(), lr=base_lr, weight_decay=0.0005, betas=(0.9, 0.99),)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300], gamma=0.1)

    # criteria = torch.nn.BCEWithLogitsLoss(reduction='mean')
    criteria = torch.nn.BCELoss(reduction='mean')
    # regression_criteria = torch.nn.MSELoss(reduction='mean').cuda()
    criteria.cuda()

    # train loop
    loss_t = []
    best_val = 0
    for i in range(epochs):
        count = 0
        running_loss = 0.
        net.train()

        for img, label in tqdm(TrainDataloader):
            count += 1
            img = img.cuda()
            label = label.cuda()

            _, _, _, _, \
                score, score_s1, score_s2, score_s3, = net(img)

            loss_main = criteria(score, label.float())
            loss_s1 = criteria(score_s1, label.float())
            loss_s2 = criteria(score_s2, label.float())
            loss_s3 = criteria(score_s3, label.float())

            loss = 10 * loss_main + 10 * loss_s3 + 5 * loss_s2 + 2.5 * loss_s1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / count
        loss_t.append(train_loss)
        scheduler.step()

        logger.info(f'Epoch [{i+1}/{epochs}], Train Loss: {train_loss:.4f}')
        if test_every != 0 and ((i + 1) % test_every == 0 or (i + 1) == epochs):
            torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(save_dir, f"model_{i+1}.pth"))

    # torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(save_dir, f"_{i+1}.pth"))

    plt.figure(1)
    plt.plot(loss_t)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('train loss')
    plt.savefig(os.path.join(save_dir, 'train_loss.png'))
    plt.close()

    # plt.figure(2)
    # plt.plot(list(range(test_every, epochs + 1, test_every)), iou_v)
    # plt.ylabel('mIoU')
    # plt.xlabel('epochs')
    # plt.title('valid mIoU')
    # plt.savefig('result/valid_iou.png')
