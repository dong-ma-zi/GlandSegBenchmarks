import multiprocessing
from multiprocessing import Lock, Pool

multiprocessing.set_start_method("spawn", True)  # ! must be at top for VScode debugging
import math
import os
import multiprocessing as mp
import pathlib
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, as_completed, wait
from functools import reduce
from importlib import import_module
from multiprocessing import Lock, Pool

import cv2
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import tqdm
from loader.infer_loader import SerializeFileList
from loader.postproc import PostProcInstErodedMap, PostProcInstErodedContourMap, get_inst_info_dict
from misc.utils import (
    log_info,
    recur_find_ext,
    mkdir,
)
from misc.viz_utils import visualize_instances_dict_orig

from . import base

# ! Change this to match with the targets.py gen code
# target key gen code : post proc class
# ! HACK - FIND BETTER WAY TO DO THIS- MAYBE IN A SEPARATE PYTHON SCRIPT
__postproc_func_dict = {
    "IP-ERODED-3": PostProcInstErodedMap,
    "IP-ERODED-11": PostProcInstErodedMap,
    "IP-ERODED-CONTOUR-3": PostProcInstErodedContourMap,
    "IP-ERODED-CONTOUR-11": PostProcInstErodedContourMap,
}


####
def _prepare_patching(img, input_size, output_size, output_overlap_size):
    """Prepare the patch information for tile processing.

    Apply mirror padding to the images and generate patch input output placement.
    """

    win_size = input_size
    msk_size = step_size = output_size

    def get_last_steps(length, msk_size, step_size):
        nr_step = math.ceil((length - msk_size) / step_size)
        last_step = (nr_step + 1) * step_size
        return int(last_step), int(nr_step + 1)

    img_shape = img.shape[:2]
    im_h = img.shape[0]
    im_w = img.shape[1]

    last_h, _ = get_last_steps(im_h, msk_size, output_size)
    last_w, _ = get_last_steps(im_w, msk_size, output_size)

    diff = win_size - step_size
    padt = padl = diff // 2
    padb = last_h + win_size - im_h
    padr = last_w + win_size - im_w

    padded_img = np.lib.pad(img, ((padt, padb), (padl, padr), (0, 0)), "reflect")

    # generating subpatches index from orginal
    input_tl_y = np.arange(0, last_h, step_size, dtype=np.int32)
    input_tl_x = np.arange(0, last_w, step_size, dtype=np.int32)
    input_tl_y, input_tl_x = np.meshgrid(input_tl_y, input_tl_x)
    input_tl = np.stack([input_tl_y.flatten(), input_tl_x.flatten()], axis=-1)
    output_tl = input_tl + diff // 2

    padded_img_shape = padded_img.shape[:2]
    output_br = output_tl + output_size
    input_br = input_tl + input_size
    sel = np.any(input_br > padded_img_shape, axis=-1)
    info_list = np.stack(
        [
            np.stack([input_tl[~sel], input_br[~sel]], axis=1),
            np.stack([output_tl[~sel], output_br[~sel]], axis=1),
        ],
        axis=1,
    )

    if output_overlap_size == 0:
        ovl_output_tl = output_tl + output_overlap_size
        ovl_input_tl = ovl_output_tl - diff // 2
        ovl_output_br = ovl_output_tl + output_size
        ovl_input_br = ovl_input_tl + input_size
        sel = np.any(ovl_input_br > padded_img_shape, axis=-1)
        ovl_info_list = np.stack(
            [
                np.stack([ovl_input_tl[~sel], ovl_input_br[~sel]], axis=1),
                np.stack([ovl_output_tl[~sel], ovl_output_br[~sel]], axis=1),
            ],
            axis=1,
        )
        info_list = np.concatenate([info_list, ovl_info_list], axis=0)

    # !!! output position is currently wrt to the alrd padded src img
    return padded_img, info_list, [padt, padl]


####
def _post_process_patches(patch_info_list, image_info, postproc_code=None, postproc_list=None, model_args=None):
    """Apply post-processing to patches.

    Args:
        patch_info: patch data and associated information
        image_info: input image data and associated information
    """
    src_pos, src_shape = image_info["src_pos"], image_info["src_shape"]

    # get the channel info - used for aggregating the prediction map
    out_ch_info = model_args['decoder_kwargs']
    nr_out_chs = 0
    idx_dict = {}
    for tissue_name, decoder_info in out_ch_info.items():
        for chann_type, nr_chans in decoder_info.items():
            start_idx = nr_out_chs
            if chann_type == 'INST':
                # don't consider background, so subtract 1
                nr_out_chs += (nr_chans - 1)
                idx_dict[tissue_name + '-INST'] = [start_idx, nr_out_chs]
            elif chann_type == 'TYPE':
                nr_out_chs += 1
                idx_dict[tissue_name.split('#')[0] + '-TYPE'] = [start_idx, nr_out_chs]
            else:
                nr_out_chs += 1
                idx_dict[tissue_name] = [start_idx, nr_out_chs]

    # * reassemble the prediction
    ch_code_list = list(patch_info_list[0][0].keys())

    out_br_list = np.array([v[1][1] for v in patch_info_list])
    hw = np.max(out_br_list, axis=0).tolist()
    ovl_canvas = np.zeros(hw + [nr_out_chs], dtype=np.float32)
    raw_canvas = np.zeros(hw + [nr_out_chs], dtype=np.float32)

    # pdata is the return of infer_step but for 1 sample
    for pdata, (patch_tl, patch_br), _ in patch_info_list:
        for ch_code, ch_val in pdata.items():

            if ch_val.ndim == 2:
                ch_val = np.expand_dims(ch_val, -1)

            ch_idx_start = idx_dict[ch_code][0]
            ch_idx_end = idx_dict[ch_code][1]
            raw_canvas[
            patch_tl[0]: patch_br[0], patch_tl[1]: patch_br[1], ch_idx_start: ch_idx_end
            ] += ch_val
            ovl_canvas[
            patch_tl[0]: patch_br[0], patch_tl[1]: patch_br[1], ch_idx_start: ch_idx_end
            ] += np.ones(ch_val.shape)
    # averaging in case of stacking overlap patch
    raw_canvas = raw_canvas / (ovl_canvas + 1.0e-8)
    raw_canvas = raw_canvas[
                 src_pos[0]: src_pos[0] + src_shape[0], src_pos[1]: src_pos[1] + src_shape[1]
                 ]

    # now apply various post-proc func to get output basing on the channel code
    tissue_code_list = [v.split("-")[0] for v in ch_code_list]
    tissue_code_list = set(tissue_code_list)

    pred_inst_map_dict = {}
    pred_type_map_dict = {}
    pred_inst_info_dict = {}
    pclass_map = None
    for tissue_code in postproc_list:
        tissue_code = tissue_code.capitalize()
        if tissue_code + "-INST" in postproc_code.keys():
            inst_info_code = postproc_code[tissue_code + "-INST"]
            proc_func = __postproc_func_dict[inst_info_code]
            pred_inst_map, pred_type_map = proc_func.post_process(
                raw_canvas, idx_dict, tissue_code
            )
            pred_inst_map_dict[tissue_code] = pred_inst_map
            pred_type_map_dict[tissue_code] = pred_type_map
        elif tissue_code == 'Patch-class':
            pclass_map = raw_canvas[..., idx_dict["Patch-Class"][0]]

    # remove lumen predictions not inside glands!
    if 'lumen' in postproc_list and 'gland' in postproc_list:
        binary_gland = pred_inst_map_dict['Gland'].copy()
        binary_gland[binary_gland > 0] = 1
        pred_lumen = binary_gland * pred_inst_map_dict['Lumen']
        pred_inst_map_dict['Lumen'] = pred_lumen  # replace with updated dictionary

    for tissue_code in postproc_list:
        tissue_code = tissue_code.capitalize()
        if tissue_code != 'Patch-class':
            pred_inst_tmp = cv2.resize(pred_inst_map_dict[tissue_code], (0, 0), fx=2, fy=2,
                                       interpolation=cv2.INTER_NEAREST)
            if tissue_code != 'Lumen':
                if pred_type_map_dict[tissue_code] is not None:
                    pred_type_tmp = cv2.resize(pred_type_map_dict[tissue_code], (0, 0), fx=2, fy=2,
                                               interpolation=cv2.INTER_NEAREST)

            # pred_inst_info = get_inst_info_dict(pred_inst_map_dict[tissue_code], pred_type_map_dict[tissue_code])
            pred_inst_info = get_inst_info_dict(pred_inst_tmp, pred_type_tmp)
            pred_inst_info_dict[tissue_code] = pred_inst_info

    return (
        image_info["src_image"],
        pred_inst_map_dict,
        pred_inst_info_dict,
        pred_type_map_dict,
        pclass_map
    )


class InferManager(base.InferManager):
    """Run inference on tiles."""

    def process_file_list(self, run_args):
        """
        Process a single image tile < 5000x5000 in size.
        """
        for variable, value in run_args.items():
            self.__setattr__(variable, value)

        def proc_callback(results):
            """Post processing callback.

            Output format is implicit assumption, taken from `_post_process_patches`

            """
            # base_name, src_image, pred_inst_map_dict, pred_type_map_dict, pclass_map = results
            src_image, pred_inst_map_dict, pred_inst_info_dict, pred_type_map_dict, pclass_map = results
            gland_map = pred_inst_map_dict['Gland']
            return gland_map


        # * caching N files and their raw output into memory
        file_idx = 0
        cache_image_list = []
        cache_patch_info_list = []
        cache_image_info_list = []

        img = self.img
        src_shape = img.shape

        padded_img, patch_info, src_pos = _prepare_patching(
            img,
            self.patch_input_shape,
            self.patch_output_shape,
            self.patch_output_overlap,
        )
        patch_info = np.split(patch_info, patch_info.shape[0], axis=0)
        patch_info = [[np.squeeze(p), file_idx] for p in patch_info]

        file_idx += 1
        cache_image_list.append(padded_img)
        cache_patch_info_list.extend(patch_info)
        cache_image_info_list.append(
            ['blank', src_shape, len(patch_info), src_pos]
        )

        # * apply neural net on cached data
        dataset = SerializeFileList(
            cache_image_list, cache_patch_info_list, self.patch_input_shape
        )

        dataloader = data.DataLoader(
            dataset,
            num_workers=self.nr_inference_workers,
            batch_size=self.batch_size,
            drop_last=False,
        )

        cum_patch_output = []
        for batch_idx, batch_data in enumerate(dataloader):
            pdat_list, pinfo_list, psrc_list = batch_data
            nr_batch = pdat_list.shape[0]
            poutput_list = self.run_step(pdat_list, self.patch_output_shape)
            pinfo_list = np.split(pinfo_list, nr_batch, axis=0)
            psrc_list = np.split(psrc_list, nr_batch, axis=0)
            pinfo_list = [np.squeeze(v.numpy()) for v in pinfo_list]
            psrc_list = [np.squeeze(v.numpy()) for v in psrc_list]
            batch_dat_list = list(zip(poutput_list, pinfo_list, psrc_list))
            cum_patch_output.extend(batch_dat_list)

        # sort patch according to file idx so that later query is faster
        cum_patch_output = sorted(cum_patch_output, key=lambda x: x[-1])
        # * parallely assemble the processed cache data for each file if possible

        for file_idx, file_info in enumerate(cache_image_info_list):
            file_path, src_shape, nr_patch, src_pos = file_info
            # image_info = cache_image_info_list[file_idx]

            file_ouput_data = []
            while len(cum_patch_output) > 0:
                curr_data = cum_patch_output.pop(0)
                file_ouput_data.append(curr_data)
                if len(cum_patch_output) == 0:
                    break
                next_data = cum_patch_output[0]
                # different file tracker so stop as patch has alrd
                # been sorted such that those coming from same file
                # stay sequentially
                if next_data[-1] != curr_data[-1]:
                    break

            # * detach this into func and multiproc dispatch it
            # assert len(file_ouput_data) == nr_batch, '%d vs %d' % (len(file_ouput_data), nr_batch)
            padded_image = cache_image_list[file_idx]
            src_image = padded_image[
                        src_pos[0]: src_pos[0] + src_shape[0],
                        src_pos[1]: src_pos[1] + src_shape[1],
                        ]
            file_info = {
                "src_pos": src_pos,
                "src_shape": src_shape[:2],
                "src_image": src_image,
            }

            func_args = (
                file_ouput_data,
                file_info,
                self.decoder_dict,
                self.postproc_list,
                self.model_args
            )

            # dispatch for parallel post-processing
            proc_output = _post_process_patches(*func_args)

        return proc_callback(proc_output)



