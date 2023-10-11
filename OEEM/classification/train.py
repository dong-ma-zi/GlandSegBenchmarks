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
from utils.torchutils import PolyOptimizer
import yaml
import importlib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', default=32, type=int)
    parser.add_argument('-epoch', default=100, type=int)
    parser.add_argument('-lr', default=0.01, type=float)
    parser.add_argument('-test_every', default=5, type=int, help="how often to test a model while training")
    parser.add_argument('-device', default=[0], type=list)
    parser.add_argument('-m', default='WRes38', type=str)
    args = parser.parse_args()

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
    

    validation_folder_name = 'glas_valid'
    validation_dataset_path = 'glas_cls/2.validation/img'
    validation_mask_path = 'glas_cls/2.validation/mask'
    crop_validation_images(validation_dataset_path, 112, 96, scales,
                           validation_folder_name, ck='_112_96')
    if not os.path.exists(validation_folder_name):
        os.mkdir(validation_folder_name)
        print('crop validation set images ...')
        crop_validation_images(validation_dataset_path, network_image_size, network_image_size, scales, validation_folder_name)
        print('cropping finishes!')

    # load model
    resnet38_path = "weights/res38d.pth"
    
    net = getattr(importlib.import_module("network.wide_resnet"), 'wideResNet')()
    # net = network.wideResNet()
    net.load_state_dict(torch.load(resnet38_path), strict=False)
    net = torch.nn.DataParallel(net, device_ids=devices).cuda()
    
    # data augmentation
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=network_image_size, scale=(0.7, 1), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=mean, std=std)
    ])

    # load training dataset
    data_path_name = f'glas_cls/1.training/img'
    TrainDataset = dataset.OriginPatchesDataset(data_path_name=data_path_name, transform=train_transform)
    print("train Dataset", len(TrainDataset))
    TrainDatasampler = torch.utils.data.RandomSampler(TrainDataset)
    TrainDataloader = DataLoader(TrainDataset, batch_size=batch_size, num_workers=4, sampler=TrainDatasampler, drop_last=True)

    # optimizer and loss
    optimizer = PolyOptimizer(net.parameters(), base_lr, weight_decay=1e-4, max_step=epochs, momentum=0.9)
    criteria = torch.nn.BCEWithLogitsLoss(reduction='mean')
    regression_criteria = torch.nn.MSELoss(reduction='mean').cuda()
    criteria.cuda()

    # train loop
    loss_t = []
    iou_v = []
    best_val = 0
    
    for i in range(epochs):
        count = 0
        running_loss = 0.
        net.train()

        for img, label in tqdm(TrainDataloader):
            count += 1
            img = img.cuda()
            label = label.cuda()
            
            scores = net(img)
            loss = criteria(scores, label.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / count
        loss_t.append(train_loss)

        valid_iou = 0
        if test_every != 0 and ((i + 1) % test_every == 0 or (i + 1) == epochs):
            # net_cam = network.wideResNet_cam()
            net_cam = getattr(importlib.import_module("network.wide_resnet"), 'wideResNet')()
            pretrained = net.state_dict()
            pretrained = {k[7:]: v for k, v in pretrained.items()}
            pretrained['fc_cam.weight'] = pretrained['fc_cls.weight'].unsqueeze(-1).unsqueeze(-1).to(torch.float64)
            pretrained['fc_cam.bias'] = pretrained['fc_cls.bias']
            # del pretrained['fc_cls.weight']
            # del pretrained['fc_cls.bias']

            net_cam.load_state_dict(pretrained)
            net_cam = torch.nn.DataParallel(net_cam, device_ids=devices).cuda()

            # calculate MIOU
            valid_image_path = os.path.join(validation_folder_name, model_name)
            generate_validation_cam(net_cam, config, batch_size, validation_dataset_path, validation_folder_name, model_name)
            valid_iou = get_overall_valid_score(valid_image_path, validation_mask_path, num_workers=8)
            iou_v.append(valid_iou)
            
            if valid_iou > best_val:
                print("Updating the best model..........................................")
                best_val = valid_iou
                torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "result/" + model_name + "_best.pth")
        
        print(f'Epoch [{i+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid mIOU: {valid_iou:.4f}, Valid Dice: {2 * valid_iou / (1 + valid_iou):.4f}')

    torch.save({"model": net.state_dict(), 'optimizer': optimizer.state_dict()}, "result/" + model_name + "_last.pth")

    plt.figure(1)
    plt.plot(loss_t)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('train loss')
    plt.savefig('result/train_loss.png')
    plt.close()

    plt.figure(2)
    plt.plot(list(range(test_every, epochs + 1, test_every)), iou_v)
    plt.ylabel('mIoU')
    plt.xlabel('epochs')
    plt.title('valid mIoU')
    plt.savefig('result/valid_iou.png')
