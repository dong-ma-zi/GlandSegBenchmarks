"""
extract_patches.py
Patch extraction script.
"""

import re
import glob
import os
import tqdm
import pathlib
import scipy.io as scio
import cv2
import numpy as np
from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir
from dataset import get_dataset

# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Determines whether to extract type map (only applicable to datasets with class labels).
    type_classification = False

    win_size = [448, 448]
    step_size = [224, 224] # 128 for consep 256 for monusac
    extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders.
                             # 'valid'- only extract from valid regions.

    # Name of dataset - use Kumar, CPM17 or CoNSeP.
    # This used to get the specific dataset img and ann loading scheme from dataset.py
    dataset_name = "MoNuSeg"
    save_root = f"/home/data2/MedImg/NucleiSeg/{dataset_name}/extracted_{extract_type}"

    # a dictionary to specify where the dataset path should be
    dataset_info = {
        # "train": {
        #     "img": (".tif", f"/home/data2/MedImg/NucleiSeg/{dataset_name}/Train/Images/"),
        #     "ann": (".mat", f"/home/data2/MedImg/NucleiSeg/{dataset_name}/Train/Annotation/"),
        # },

        "test": {
            "img": (".tif", f"/home/data2/MedImg/NucleiSeg/{dataset_name}/Test/Images/"),
            "ann": (".mat", f"/home/data2/MedImg/NucleiSeg/{dataset_name}/Test/Annotation/"),
        },
    }

    # parser = get_dataset(dataset_name)
    parser = get_dataset('kumar')
    xtractor = PatchExtractor(win_size, step_size)
    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc["img"]
        ann_ext, ann_dir = split_desc["ann"]

        out_dir = "%s/%s/%dx%d_%dx%d/" % (save_root,
                                          split_name,
                                          win_size[0],
                                          win_size[1],
                                          step_size[0],
                                          step_size[1])

        # file_list = glob.glob(os.path.join(ann_dir, f'*{ann_ext}'))
        file_list = glob.glob(os.path.join(img_dir, f'*{img_ext}'))
        file_list.sort()  # ensure same ordering across platform

        rm_n_mkdir(os.path.join(out_dir, 'np_files'))
        rm_n_mkdir(os.path.join(out_dir, 'Images'))
        rm_n_mkdir(os.path.join(out_dir, 'Labels'))


        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, file_path in enumerate(file_list):
            base_name = pathlib.Path(file_path).stem

            img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            ann = parser.load_ann(
                "%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification
            )

            # *
            # # # TODO: resize to config the monusac to 20x
            # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            # ann = cv2.resize(ann, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            img = np.concatenate([img, ann], axis=-1)
            # ################### TODO: padding the img to fit the min patch size #######################
            # if img.shape[0] < 256 or img.shape[1] < 256:
            #     # img_padding = np.zeros(shape=(max(256, img.shape[0]), max(256, img.shape[1]), img.shape[2]))
            #     img_padding = np.zeros(shape=(max(256, img.shape[0]), max(256, img.shape[1]), img.shape[2]))
            #     img_padding[:, :, :3] = 255
            #     img_padding[:img.shape[0], :img.shape[1], :] = img
            #     img = img_padding

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
                np.save("{0}/np_files/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
                img = patch[:, :, :3]
                img = np.array(img, np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                inst_mask = patch[:, :, 3]
                # cls_mask = patch[:, :, 4]
                cv2.imwrite("{0}/Images/{1}_{2:03d}.tif".format(out_dir, base_name, idx), img)
                scio.savemat("{0}/Labels/{1}_{2:03d}.mat".format(out_dir, base_name, idx), {'inst_map': inst_mask
                                                                                            # 'cls_map': cls_mask
                                                                                            })

                pbar.update()
            pbar.close()
            # *

            pbarx.update()
        pbarx.close()
