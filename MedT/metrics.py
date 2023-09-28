import torch
from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _WeightedLoss
import numpy as np
from scipy.spatial import cKDTree


EPSILON = 1e-32


class LogNLLLoss(_WeightedLoss):
    __constants__ = ['weight', 'reduction', 'ignore_index']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction=None,
                 ignore_index=-100):
        super(LogNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, y_input, y_target):
        # y_input = torch.log(y_input + EPSILON)
        return cross_entropy(y_input, y_target, weight=self.weight,
                             ignore_index=self.ignore_index)


def classwise_iou(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """
    dims = (0, *range(2, len(output.shape)))
    gt = torch.zeros_like(output).scatter_(1, gt[:, None, :], 1)
    intersection = output*gt
    union = output + gt - intersection
    classwise_iou = (intersection.sum(dim=dims).float() + EPSILON) / (union.sum(dim=dims) + EPSILON)

    return classwise_iou


def classwise_f1(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """

    epsilon = 1e-20
    n_classes = output.shape[1]

    output = torch.argmax(output, dim=1)
    true_positives = torch.tensor([((output == i) * (gt == i)).sum() for i in range(n_classes)]).float()
    selected = torch.tensor([(output == i).sum() for i in range(n_classes)]).float()
    relevant = torch.tensor([(gt == i).sum() for i in range(n_classes)]).float()

    precision = (true_positives + epsilon) / (selected + epsilon)
    recall = (true_positives + epsilon) / (relevant + epsilon)
    classwise_f1 = 2 * (precision * recall) / (precision + recall)

    return classwise_f1


def make_weighted_metric(classwise_metric):
    """
    Args:
        classwise_metric: classwise metric like classwise_IOU or classwise_F1
    """

    def weighted_metric(output, gt, weights=None):

        # dimensions to sum over
        dims = (0, *range(2, len(output.shape)))

        # default weights
        if weights == None:
            weights = torch.ones(output.shape[1]) / output.shape[1]
        else:
            # creating tensor if needed
            if len(weights) != output.shape[1]:
                raise ValueError("The number of weights must match with the number of classes")
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights)
            # normalizing weights
            weights /= torch.sum(weights)

        classwise_scores = classwise_metric(output, gt).cpu()

        return classwise_scores 

    return weighted_metric

def ObjectF1score(S, G):
    S = np.single(S)
    G = np.single(G)

    listS = np.unique(S)
    listS = listS[listS != 0]
    numS = len(listS)

    listG = np.unique(G)
    listG = listG[listG != 0]
    numG = len(listG)

    if numS == 0 and numG == 0:
        score = 1
        return score
    elif numS == 0 or numG == 0:
        score = 0
        return score

    tempMat = np.zeros((numS, 3), dtype=int)
    tempMat[:, 0] = listS

    for iSegmentedObj in range(numS):
        intersectGTObjs = G[S == tempMat[iSegmentedObj, 0]]
        intersectGTObjs = intersectGTObjs[intersectGTObjs != 0]
        if len(intersectGTObjs) > 0:
            listOfIntersectGTObjs, N = np.unique(intersectGTObjs, return_counts=True)
            maxId = np.argmax(N)
            tempMat[iSegmentedObj, 1] = listOfIntersectGTObjs[maxId]

    for iSegmentedObj in range(numS):
        if tempMat[iSegmentedObj, 1] != 0:
            SegObj = S == tempMat[iSegmentedObj, 0]
            GTObj = G == tempMat[iSegmentedObj, 1]
            overlap = np.logical_and(SegObj, GTObj)
            areaOverlap = np.sum(overlap)
            areaGTObj = np.sum(GTObj)
            if areaOverlap / areaGTObj > 0.5:
                tempMat[iSegmentedObj, 2] = 1

    TP = np.sum(tempMat[:, 2] == 1)
    FP = np.sum(tempMat[:, 2] == 0)
    FN = numG - TP

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    score = (2 * precision * recall) / (precision + recall)
    return score, TP, FP, FN


def ObjectHausdorff(S, G):
    S = S.astype(np.float32)
    G = G.astype(np.float32)

    totalAreaS = np.sum(S > 0)
    totalAreaG = np.sum(G > 0)

    listLabelS = np.unique(S)
    listLabelS = listLabelS[listLabelS != 0]

    listLabelG = np.unique(G)
    listLabelG = listLabelG[listLabelG != 0]

    temp1 = 0
    for iLabelS in listLabelS:
        Si = S == iLabelS
        intersectlist = G[Si]
        intersectlist = intersectlist[intersectlist != 0]

        if intersectlist.size > 0:
            indexGi = np.bincount(intersectlist.astype(int)).argmax()
            Gi = G == indexGi
        else:
            tempDist = np.zeros(len(listLabelG))
            for iLabelG in range(len(listLabelG)):
                Gi = G == listLabelG[iLabelG]
                tempDist[iLabelG] = Hausdorff(Gi, Si)
            minIdx = np.argmin(tempDist)
            Gi = G == listLabelG[minIdx]

        omegai = np.sum(Si) / totalAreaS
        temp1 += omegai * Hausdorff(Gi, Si)

    temp2 = 0
    for iLabelG in listLabelG:
        tildeGi = G == iLabelG
        intersectlist = S[tildeGi]
        intersectlist = intersectlist[intersectlist != 0]

        if intersectlist.size > 0:
            indextildeSi = np.bincount(intersectlist.astype(int)).argmax()
            tildeSi = S == indextildeSi
        else:
            tempDist = np.zeros(len(listLabelS))
            for iLabelS in range(len(listLabelS)):
                tildeSi = S == listLabelS[iLabelS]
                tempDist[iLabelS] = Hausdorff(tildeGi, tildeSi)
            minIdx = np.argmin(tempDist)
            tildeSi = S == listLabelS[minIdx]

        tildeOmegai = np.sum(tildeGi) / totalAreaG
        temp2 += tildeOmegai * Hausdorff(tildeGi, tildeSi)

    objHausdorff = (temp1 + temp2) / 2
    return objHausdorff


def Hausdorff(S, G):
    S = S.astype(np.float32)
    G = G.astype(np.float32)

    maskS = S > 0
    maskG = G > 0
    rowInd, colInd = np.indices(S.shape)
    coordinates = np.column_stack((rowInd[maskG], colInd[maskG]))

    x = coordinates
    y = np.column_stack((rowInd[maskS], colInd[maskS]))

    kdtree = cKDTree(y)
    dist, _ = kdtree.query(x)
    dist1 = np.max(dist)

    kdtree = cKDTree(x)
    dist, _ = kdtree.query(y)
    dist2 = np.max(dist)

    hausdorffDistance = max(dist1, dist2)
    return hausdorffDistance

def ObjectDice(S, G):
    def Dice(A, B):
        temp = np.logical_and(A, B)
        dice = 2 * np.sum(temp) / (np.sum(A) + np.sum(B))
        return dice

    S = np.single(S)
    G = np.single(G)

    listLabelS = np.unique(S)
    listLabelS = listLabelS[listLabelS != 0]
    numS = len(listLabelS)

    listLabelG = np.unique(G)
    listLabelG = listLabelG[listLabelG != 0]
    numG = len(listLabelG)

    if numS == 0 and numG == 0:
        objDice = 1
        return objDice
    elif numS == 0 or numG == 0:
        objDice = 0
        return objDice

    totalAreaS = np.sum(S > 0)
    temp1 = 0
    for iLabelS in listLabelS:
        Si = (S == iLabelS)
        intersectlist = G[Si]
        intersectlist = intersectlist[intersectlist != 0]

        if len(intersectlist) != 0:
            indexGi = np.argmax(np.bincount(intersectlist.astype(int)))
            Gi = (G == indexGi)
        else:
            Gi = np.zeros_like(G, dtype=bool)

        omegai = np.sum(Si) / totalAreaS
        temp1 += omegai * Dice(Gi, Si)

    totalAreaG = np.sum(G > 0)
    temp2 = 0
    for iLabelG in listLabelG:
        tildeGi = (G == iLabelG)
        intersectlist = S[tildeGi]
        intersectlist = intersectlist[intersectlist != 0]

        if len(intersectlist) != 0:
            indextildeSi = np.argmax(np.bincount(intersectlist.astype(int)))
            tildeSi = (S == indextildeSi)
        else:
            tildeSi = np.zeros_like(S, dtype=bool)

        tildeOmegai = np.sum(tildeGi) / totalAreaG
        temp2 += tildeOmegai * Dice(tildeGi, tildeSi)

    objDice = (temp1 + temp2) / 2
    return objDice



jaccard_index = make_weighted_metric(classwise_iou)
f1_score = make_weighted_metric(classwise_f1)


if __name__ == '__main__':
    output, gt = torch.zeros(3, 2, 5, 5), torch.zeros(3, 5, 5).long()
    print(classwise_iou(output, gt))
