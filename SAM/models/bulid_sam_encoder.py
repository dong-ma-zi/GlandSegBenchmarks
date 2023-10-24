# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from pathlib import Path
import urllib.request
import torch

from .modeling import (
    ImageEncoderViT,
    # MaskDecoder,
    # PromptEncoder,
    # Sam,
    # TwoWayTransformer,
)


def build_sam_vit_h(args=None, checkpoint=None):
    return _build_sam(
        args,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(args, checkpoint=None):
    return _build_sam(
        args,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(args, checkpoint=None):
    return _build_sam(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

sam_encoder_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
        args,
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        checkpoint=None,
):

    prompt_embed_dim = args.out_chans
    image_size = args.image_size
    vit_patch_size = args.patch_size
    encoder_embed_dim = args.embed_dim
    # mage_embedding_size = image_size // vit_patch_size
    sam_encoder = ImageEncoderViT(
            # args=args,
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
    sam_encoder.eval()

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam_encoder.load_state_dict(state_dict, strict=False)

    return sam_encoder
