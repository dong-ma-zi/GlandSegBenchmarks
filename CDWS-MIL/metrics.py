import numpy as np
from scipy.spatial import cKDTree


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

def dice_coefficient(array1, array2):
    # 计算交集的大小
    intersection_size = np.sum(array1 & array2)

    # 计算两个数组的大小
    size_array1 = np.sum(array1)
    size_array2 = np.sum(array2)

    # 计算Dice系数
    dice = (2.0 * intersection_size) / (size_array1 + size_array2)

    return dice


def iou_metrics(array1, array2):
    # 计算交集
    intersection = np.logical_and(array1, array2)

    # 计算并集
    union = np.logical_or(array1, array2)

    # 计算交并比
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score