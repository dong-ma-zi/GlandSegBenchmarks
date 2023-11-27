import numpy as np
from scipy.spatial import cKDTree

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



import cv2
import scipy
from scipy.optimize import linear_sum_assignment

def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

def generate_cls_info(inst, cls):
    inst_id_list = np.unique(inst)[1:]  # exlcude background
    inst_info_dict = {"inst_centroid": [],
                      "inst_type": []}
    for inst_id in inst_id_list:
        inst_map = inst == inst_id
        # TODO: chane format of bbox output
        rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
        inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
        inst_map = inst_map[
            inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
        ]
        inst_map = inst_map.astype(np.uint8)
        inst_moment = cv2.moments(inst_map)
        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid = np.array(inst_centroid)
        inst_centroid[0] += inst_bbox[0][1]  # X
        inst_centroid[1] += inst_bbox[0][0]  # Y

        inst_info_dict["inst_centroid"] += [inst_centroid]

        #### * Get class of each instance id, stored at index id-1
        rmin, cmin, rmax, cmax = (inst_bbox).flatten()
        inst_map_crop = inst[rmin:rmax, cmin:cmax]
        inst_type_crop = cls[rmin:rmax, cmin:cmax]
        inst_map_crop = (
            inst_map_crop == inst_id
        )  # TODO: duplicated operation, may be expensive
        inst_type = inst_type_crop[inst_map_crop]
        type_list, type_pixels = np.unique(inst_type, return_counts=True)
        type_list = list(zip(type_list, type_pixels))
        type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
        inst_type = type_list[0][0]
        if inst_type == 0:  # ! pick the 2nd most dominant if exist
            if len(type_list) > 1:
                inst_type = type_list[1][0]
        inst_info_dict["inst_type"] += [int(inst_type)]

    inst_info_dict["inst_centroid"] = np.array(inst_info_dict["inst_centroid"])
    inst_info_dict["inst_type"] = np.array(inst_info_dict["inst_type"])[..., None]
    return inst_info_dict

#####
def pair_coordinates(setA, setB, radius):
    """Use the Munkres or Kuhn-Munkres algorithm to find the most optimal
    unique pairing (largest possible match) when pairing points in set B
    against points in set A, using distance as cost function.
    Args:
        setA, setB: np.array (float32) of size Nx2 contains the of XY coordinate
                    of N different points
        radius: valid area around a point in setA to consider
                a given coordinate in setB a candidate for match
    Return:
        pairing: pairing is an array of indices
        where point at index pairing[0] in set A paired with point
        in set B at index pairing[1]
        unparedA, unpairedB: remaining poitn in set A and set B unpaired
    """
    # * Euclidean distance as the cost matrix
    pair_distance = scipy.spatial.distance.cdist(setA, setB, metric='euclidean')

    # * Munkres pairing with scipy library
    # the algorithm return (row indices, matched column indices)
    # if there is multiple same cost in a row, index of first occurence
    # is return, thus the unique pairing is ensured
    indicesA, paired_indicesB = linear_sum_assignment(pair_distance)

    # extract the paired cost and remove instances
    # outside of designated radius
    pair_cost = pair_distance[indicesA, paired_indicesB]

    pairedA = indicesA[pair_cost <= radius]
    pairedB = paired_indicesB[pair_cost <= radius]

    pairing = np.concatenate([pairedA[:,None], pairedB[:,None]], axis=-1)
    unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
    unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)
    return pairing, unpairedA, unpairedB

from collections import Counter
def run_nuclei_type_stat(pred_dict, true_dict, type_uid_list=None, exhaustive=True, _20x=False):
    """GT must be exhaustively annotated for instance location (detection).
    Args:
        true_dir, pred_dir: Directory contains .mat annotation for each image.
                            Each .mat must contain:
                    --`inst_centroid`: Nx2, contains N instance centroid
                                       of mass coordinates (X, Y)
                    --`inst_type`    : Nx1: type of each instance at each index
                    `inst_centroid` and `inst_type` must be aligned and each
                    index must be associated to the same instance
        type_uid_list : list of id for nuclei type which the score should be calculated.
                        Default to `None` means available nuclei type in GT.
        exhaustive : Flag to indicate whether GT is exhaustively labelled
                     for instance types

    """

    paired_all = []  # unique matched index pair
    unpaired_true_all = []  # the index must exist in `true_inst_type_all` and unique
    unpaired_pred_all = []  # the index must exist in `pred_inst_type_all` and unique
    true_inst_type_all = []  # each index is 1 independent data point
    pred_inst_type_all = []  # each index is 1 independent data point
    for file_idx, filename in enumerate(pred_dict):

        true_info = true_dict[filename]
        # dont squeeze, may be 1 instance exist
        true_centroid = (true_info["inst_centroid"]).astype("float32")
        true_inst_type = (true_info["inst_type"]).astype("int32")

        if true_centroid.shape[0] != 0:
            true_inst_type = true_inst_type[:, 0]
        else:  # no instance at all
            true_centroid = np.array([[0, 0]])
            true_inst_type = np.array([0])

        pred_info = pred_dict[filename]
        # dont squeeze, may be 1 instance exist
        pred_centroid = (pred_info["inst_centroid"]).astype("float32")
        pred_inst_type = (pred_info["inst_type"]).astype("int32")

        if pred_centroid.shape[0] != 0:
            pred_inst_type = pred_inst_type[:, 0]
        else:  # no instance at all
            pred_centroid = np.array([[0, 0]])
            pred_inst_type = np.array([0])

        # ! if take longer than 1min for 1000 vs 1000 pairing, sthg is wrong with coord
        dist = 12 if not _20x else 6
        paired, unpaired_true, unpaired_pred = pair_coordinates(
            true_centroid, pred_centroid, dist
        )

        # * Aggreate information
        # get the offset as each index represent 1 independent instance
        true_idx_offset = (
            true_idx_offset + true_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )
        pred_idx_offset = (
            pred_idx_offset + pred_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )
        true_inst_type_all.append(true_inst_type)
        pred_inst_type_all.append(pred_inst_type)

        # increment the pairing index statistic
        if paired.shape[0] != 0:  # ! sanity
            paired[:, 0] += true_idx_offset
            paired[:, 1] += pred_idx_offset
            paired_all.append(paired)

        unpaired_true += true_idx_offset
        unpaired_pred += pred_idx_offset
        unpaired_true_all.append(unpaired_true)
        unpaired_pred_all.append(unpaired_pred)

    paired_all = np.concatenate(paired_all, axis=0)
    unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
    unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
    true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
    pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

    # inst_type_count = Counter(true_inst_type_all)

    paired_true_type = true_inst_type_all[paired_all[:, 0]]
    paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
    unpaired_true_type = true_inst_type_all[unpaired_true_all]
    unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

    ###
    def _f1_type(paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
        type_samples = (paired_true == type_id) | (paired_pred == type_id)

        paired_true = paired_true[type_samples]
        paired_pred = paired_pred[type_samples]

        tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
        tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
        fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
        fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

        if not exhaustive:
            ignore = (paired_true == -1).sum()
            fp_dt -= ignore

        fp_d = (unpaired_pred == type_id).sum()
        fn_d = (unpaired_true == type_id).sum()

        f1_type = (2 * (tp_dt + tn_dt)) / (
                2 * (tp_dt + tn_dt)
                + w[0] * fp_dt
                + w[1] * fn_dt
                + w[2] * fp_d
                + w[3] * fn_d
        )
        return f1_type

    # overall
    # * quite meaningless for not exhaustive annotated dataset
    w = [1, 1]
    tp_d = paired_pred_type.shape[0]
    fp_d = unpaired_pred_type.shape[0]
    fn_d = unpaired_true_type.shape[0]

    tp_tn_dt = (paired_pred_type == paired_true_type).sum()
    fp_fn_dt = (paired_pred_type != paired_true_type).sum()

    if not exhaustive:
        ignore = (paired_true_type == -1).sum()
        fp_fn_dt -= ignore

    acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
    f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

    w = [2, 2, 1, 1]

    if type_uid_list is None:
        type_uid_list = np.unique(true_inst_type_all).tolist()

    results_list = [f1_d, acc_type]
    for type_uid in type_uid_list:
        f1_type = _f1_type(
            paired_true_type,
            paired_pred_type,
            unpaired_true_type,
            unpaired_pred_type,
            type_uid,
            w,
        )
        results_list.append(f1_type)

    np.set_printoptions(formatter={"float": "{: 0.5f}".format})
    # print(np.array(results_list))
    return results_list