"""extract_patches.py

Patch extraction script.
"""

import re
import glob
import os
import tqdm
import pathlib
import cv2
import numpy as np
import scipy.io as scio
from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir
import random

# -------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Determines whether to extract type map (only applicable to datasets with class labels).
    type_classification = False
    ## control the class of nuclei, but we only consider the segmentation not classification

    # win_size = [540, 540]
    # step_size = [164, 164]
    win_size = [448, 448]
    step_size = [224, 224]
    # win_size = [300, 300]
    # step_size = [150, 150]

    extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.

    # Name of dataset - use Kumar, CPM17 or CoNSeP.
    # This used to get the specific dataset img and ann loading scheme from dataset.py

    ###################### kumar ######################################
    # dataset_name = "kumar"
    save_root = "/home/data2/MedImg/GlandSeg/GlaS/my/"

    # a dictionary to specify where the dataset path should be
    dataset_info = {
        "train": {
            "img": (".bmp", "/home/data2/MedImg/GlandSeg/GlaS/train/Images"),
            "ann": (".bmp", "/home/data2/MedImg/GlandSeg/GlaS/train/Annotation"),
            #"ann": (".mat", "/mnt/code3/wzh/NuSeg/hover_net/pseudo_mask/MoNuSeg_300x300_20fold2_t1_train")
        },
        "valid": {
            "img": (".bmp", "/home/data2/MedImg/GlandSeg/GlaS/test/Images"),
            "ann": (".bmp", "/home/data2/MedImg/GlandSeg/GlaS/test/Annotation"),
        },
    }
    ###################################################################

    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    xtractor = PatchExtractor(win_size, step_size)
    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc["img"]
        ann_ext, ann_dir = split_desc["ann"]

        img_out_dir = "%s/%s/%dx%d/Images" % (
            save_root,
            split_name,
            win_size[0],
            win_size[1],
        )
        ann_out_dir = "%s/%s/%dx%d/Annotation" % (
            save_root,
            split_name,
            win_size[0],
            win_size[1],
        )
        file_list = glob.glob(patterning("%s/*%s" % (img_dir, img_ext)))

        if split_name == 'train':
            ### split the validation set
            total = len(file_list)
            print('Total data: ', total)
            valid_num = 3
            random.seed(2023)
            random.shuffle(file_list)
            file_list = file_list[3:]
            print('Training data: ', len(file_list))

        file_list.sort()  # ensure same ordering across platform
        if not os.path.exists(img_out_dir):
            os.makedirs(img_out_dir)
        if not os.path.exists(ann_out_dir):
            os.makedirs(ann_out_dir)

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, file_path in enumerate(file_list):
            base_name = pathlib.Path(file_path).stem

            # img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            # ann = parser.load_ann(
            #     "%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification
            # )
            img = cv2.cvtColor(cv2.imread("%s/%s%s" % (img_dir, base_name, img_ext)), cv2.COLOR_BGR2RGB)
            ann = cv2.imread("%s/%s_anno%s" % (ann_dir, base_name, ann_ext)).astype("int32")

            # *
            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            for idx, patch in enumerate(sub_patches):
                # np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
                subImg = patch[:,:,:3]
                subImg = np.array(subImg, np.uint8)
                subImg = cv2.cvtColor(subImg, cv2.COLOR_BGR2RGB)
                subAnno = patch[:,:,3]
                cv2.imwrite("{0}/{1}_{2:03d}.bmp".format(img_out_dir, base_name, idx), subImg)
                cv2.imwrite("{0}/{1}_{2:03d}.bmp".format(ann_out_dir, base_name, idx), subAnno)
                #cv2.imwrite("{0}/Images/{1}_{2:03d}.png".format(out_dir, base_name, idx), img)
                #scio.savemat("{0}/Labels/{1}_{2:03d}.mat".format(out_dir, base_name, idx), {'inst_map': mask})
                pbar.update()
            pbar.close()
            # *
            pbarx.update()
        pbarx.close()
