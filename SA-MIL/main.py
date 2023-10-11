import argparse
# import torch
from torch.utils.data import DataLoader

from models import *
from functions import *
from datasets import *
from utils import *

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

def get_args_parser():
    parser = argparse.ArgumentParser('Multiple Instance Learning for Histopathological Image Segmentation', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--model', default='SA_MIL-2', type=str, choices=['Swin_MIL', 'DWS_MIL', 'SA_MIL'])
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)

    parser.add_argument('--work_path', default='./weights', type=str, help="The path to save model")
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--device_ids', default=[1], type=list)
    parser.add_argument('--r', default=4, type=int, help="The hyperparameter r in Generalized Mean function")

    parser.add_argument('--pretrain', default=True, type=bool, help="Invoke the pretrain method of the model object")
    parser.add_argument('--train', default=True, type=bool, help="Performing training process")
    parser.add_argument('--test', default=False, type=bool, help="Performing testing process")
    parser.add_argument('--checkpoint', default=None, type=str, help="The filename of the checkpoint loaded during testing. If not provided, the best model will be used.")
    parser.add_argument('--save_every', default=5, type=int, help="freq Saving checkpoint during training")
    return parser

def main(args):
    torch.cuda.set_device(1)
    args.work_path = os.path.join(args.work_path, args.model)
    print("Initializing......")
    work_path = args.work_path
    if not os.path.exists(work_path):
        os.makedirs(work_path)
    print("work path: " + work_path)

    import datetime
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    logger = get_logger(os.path.join(args.work_path, f'train_log_{cur_time}.txt'))
    logger.info(args)

    # model = Swin_MIL().cuda()
    # model = DWS_MIL().cuda()
    model = SA_MIL().cuda()

    print('Model: ', args.model)

    if args.train:
        # dataset_train = Dataset_train(args)
        # dataset_valid = Dataset_valid(args)
        dataset_train = OriginPatchesDataset(f'/home/data1/my/Project/GlandSegBenchmark/OEEM/classification/glas_cls/1.training/img_112_56/Train/Images/')
        dataset_valid = ValidPatchesDataset(f'/home/data1/my/Project/GlandSegBenchmark/OEEM/classification/glas_cls/1.training/img_112_56/Valid/')
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        dataloader_valid = DataLoader(dataset_valid, batch_size=1, shuffle=False, num_workers=4)
        print('total train data: ', dataset_train.__len__())
        print('Batch Size: ', args.batch_size)
        # train(model, dataloader_train, args, valid, dataloader_valid)
        train(model, dataloader_train, args, logger, valid_fn=valid, dataloader_valid=dataloader_valid)
        # train(model, dataloader_train, args, logger, valid_fn=None, dataloader_valid=None)


    # if args.test:
    #     print("testing......")
    #     dataset_test = Dataset_test(args)
    #     dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    #     test(model, dataloader_test, args)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
