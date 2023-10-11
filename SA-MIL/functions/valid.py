import torch
import tqdm
from sklearn import metrics
from .gm import gm
import numpy as np

# def valid(model, dataloader, args):
#     num = 0
#     total_f = 0
#
#     pred_list = []
#     true_list = []
#
#     with torch.no_grad():
#         print('validation')
#         for image, label in tqdm.tqdm(dataloader):
#             num += 1
#             pred, _, _, _ = model(image.to(torch.device(args.device)))
#             # pred = ((preds[0] >= 0.5) + 0).to('cpu').numpy()
#             pred = (gm(pred, args.r)>= 0.5 + 0).int().to('cpu').numpy()
#             label = label.int().to("cpu").numpy()
#             pred_list += [pred[0, 0]]
#             true_list += [label[0, 0]]
#
#             # if label.sum().item() > 0:
#             #     f1 = metrics.f1_score(label.reshape(-1), pred.reshape(-1), pos_label=1)
#             # else:
#             #     f1 = metrics.f1_score(label.reshape(-1), pred.reshape(-1), pos_label=0)
#
#         #     total_f += f1
#         # average_f = total_f/num
#         average_f = metrics.f1_score(pred_list, true_list, pos_label=1)
#         return average_f

def iou_metrics(array1, array2):
    # 计算交集
    intersection = np.logical_and(array1, array2)
    # 计算并集
    union = np.logical_or(array1, array2)
    # 计算交并比
    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)

    # 添加分母项来防止出现NaN
    if union_sum == 0:
        iou_score = 0.0
    else:
        iou_score = intersection_sum / union_sum

    return iou_score

def valid(model, dataloader, args):
    # make sure batch size 1
    num = 0
    iou = 0

    with torch.no_grad():
        print('validation')
        for image, label in tqdm.tqdm(dataloader):
            num += 1
            pred, _, _, _ = model(image.to(torch.device(args.device)))
            pred = pred[0].cpu().numpy()
            label = label[0].cpu().numpy()
            pred = np.argmin(pred, axis=0)
            iou += iou_metrics(pred, label)

        return iou / len(dataloader)
