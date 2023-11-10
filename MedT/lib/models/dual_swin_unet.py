import torch
import torch.nn as nn
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

class DualSwinUnet(nn.Module):
    def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(DualSwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head

        self.teacher = SwinTransformerSys(img_size=img_size,
                                patch_size=4,
                                in_chans=3,
                                num_classes=num_classes,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

        self.student = SwinTransformerSys(img_size=img_size,
                                patch_size=4,
                                in_chans=3,
                                num_classes=num_classes,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)


    def forward(self, x, u_x=None, mode='fully'):
        if mode == 'fully':
            if x.size()[1] == 1:
                x = x.repeat(1,3,1,1)
            logits = self.student(x)
            return logits
        elif mode == 'semi':
            ## 带标注数据
            if x.size()[1] == 1:
                x = x.repeat(1,3,1,1)
            logits = self.student(x)

            ## 无标注数据
            if u_x.size()[1] == 1:
                u_x = u_x.repeat(1,3,1,1)
            u_logits_t = self.teacher(u_x)
            u_logits_s = self.student(u_x)
            return logits, u_logits_t, u_logits_s


    def load_from(self, device, logger, pretrained_path):
        if pretrained_path is not None:
            logger.info("pretrained_path:{}".format(pretrained_path))
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                logger.info("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        logger.info("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            logger.info("---start load pretrained model of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        logger.info(
                            "delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            logger.info("none pretrain")