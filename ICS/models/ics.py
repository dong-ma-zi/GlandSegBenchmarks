import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.ops.boxes import nms as nms_torch

from efficientnet import EfficientNet as EffNet
from utils import MemoryEfficientSwish, Swish
from bifpn import SeparableConvBlock, BiFPN
from utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding

pretrained_path = {'efficientnet-b1' : 'adv-efficientnet-b1-0f3ce85a.pth',
                   'efficientnet-b5' : 'adv-efficientnet-b5-86493f6b.pth'}

def nms(dets, thresh):
    return nms_torch(dets[:, :4], dets[:, 4], thresh)


class Regressor(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_layers, pyramid_levels=5, onnx_export=False):
        super(Regressor, self).__init__()
        self.num_layers = num_layers

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(pyramid_levels)])
        self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)

        return feats


class Classifier(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_classes, num_layers, pyramid_levels=5, onnx_export=False):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(pyramid_levels)])
        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
                                          self.num_classes)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)
        feats = feats.sigmoid()

        return feats


class EfficientNet(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, compound_coef, load_weights=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_name(f'efficientnet-b{compound_coef}')
        model.load_state_dict(torch.load(pretrained_path[f'efficientnet-b{compound_coef}']))
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc

        ## out_channels are set to 88 for b0, 188 for b5
        self.gamma = 88
        self.conv1x1_0 = nn.Conv2d(16, self.gamma, kernel_size=1)
        self.conv1x1_1 = nn.Conv2d(24, self.gamma, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(40, self.gamma, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(112, self.gamma, kernel_size=1)
        self.conv1x1_4 = nn.Conv2d(320, self.gamma, kernel_size=1)
        self.bifpn1 = BiFPN(self.gamma, self.gamma)
        self.bifpn2 = BiFPN(self.gamma, self.gamma)
        self.bifpn3 = BiFPN(self.gamma, self.gamma)

        self.seperable_conv = nn.Sequential(
            SeparableConvBlock(self.gamma, self.gamma),
            SeparableConvBlock(self.gamma, self.gamma),
            SeparableConvBlock(self.gamma, self.gamma),
            SeparableConvBlock(self.gamma, self.gamma)
        )

        self.model = model

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        # TODO: temporarily storing extra tensor last_x and del it later might not be a good idea,
        #  try recording stride changing when creating efficientnet,
        #  and then apply it here.
        last_x = None
        output_idx = [0, 3, 6, 14, 21]
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if idx == 0:
                feat0 = self.conv1x1_0(x)
                _, _, height, width = feat0.size()
            elif idx == 3:
                feat1 = self.conv1x1_1(x)
            elif idx == 6:
                feat2 = self.conv1x1_2(x)
            elif idx == 14:
                feat3 = self.conv1x1_3(x)
            elif idx == 21:
                feat4 = self.conv1x1_4(x)
            if idx in output_idx:
                feature_maps.append(x)
        inputs = (feat0, feat1, feat2, feat3, feat4)
        feat0, feat1, feat2, feat3, feat4 = self.bifpn3(self.bifpn2(self.bifpn1(inputs)))
        feat1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(feat1)
        feat1 = F.interpolate(feat1, size=(height, width), mode='bilinear', align_corners=True)
        feat2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)(feat2)
        feat2 = F.interpolate(feat2, size=(height, width), mode='bilinear', align_corners=True)
        feat3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)(feat3)
        feat3 = F.interpolate(feat3, size=(height, width), mode='bilinear', align_corners=True)
        feat4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)(feat4)
        feat4 = F.interpolate(feat4, size=(height, width), mode='bilinear', align_corners=True)
        feat = feat0 + feat1+ feat2 + feat3 + feat4
        output = self.seperable_conv(feat)
        output = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(output)
        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
        return output,feature_maps

if __name__ == '__main__':
    torch.cuda.set_device(3)
    model = EfficientNet(compound_coef=1).cuda()
    inputs = torch.rand(1, 3, 224, 224).cuda()
    output, feature_maps = model(inputs)
    print('Output size: ', output.size())
    print('Length of feature maps: ', len(feature_maps))
    print(feature_maps[0].shape)  # torch.Size([1, 16, 112, 112])
    print(feature_maps[1].shape)  # torch.Size([1, 24, 56, 56])
    print(feature_maps[2].shape)  # torch.Size([1, 40, 28, 28])
    print(feature_maps[3].shape)  # torch.Size([1, 112, 14, 14])
    print(feature_maps[4].shape)  # torch.Size([1, 320, 7, 7])



