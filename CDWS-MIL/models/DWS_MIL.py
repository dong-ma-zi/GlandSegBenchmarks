"""
Author: wzh
Since: 2023-9-8
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
torch.cuda.set_device(0)
from torchvision.models import alexnet


class GeneralizedMeanPooling(nn.Module):
    def __init__(self, norm=3, output_size=(1, 1), eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

class DWS_MIL(nn.Module):
    def __init__(self, n_class=2):
        # 本质上是FCN, backbone应该是Vgg
        super(DWS_MIL, self).__init__()

        # GMP
        self.GMP = GeneralizedMeanPooling(norm=4, )

        # conv1
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2
        self.sigmoid1 = nn.Sigmoid()

        # conv2
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4
        self.sigmoid2 = nn.Sigmoid()

        # conv3
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8
        self.sigmoid3 = nn.Sigmoid()

        # object
        self.o_classifier1 = nn.Conv2d(64, n_class, 1)
        self.o_classifier2 = nn.Conv2d(128, n_class, 1)
        self.o_classifier3 = nn.Conv2d(256, n_class, 1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)


    def forward(self, x, infer=False):
        _, _, height, width = x.size()
        h = x
        # stage#1
        h = self.relu1(self.conv1(h))
        h = self.pool1(h)  # 1/2
        out_stage_1 = self.sigmoid1((self.o_classifier1(h)))
        out_stage_1 = F.interpolate(out_stage_1, size=(height, width), mode='bilinear', align_corners=True)
        im_out_stage_1 = self.GMP(out_stage_1).squeeze(-1).squeeze(-1)

        # stage#2
        h = self.relu2(self.conv2(h))
        h = self.pool2(h)  # 1/4
        out_stage_2 = self.sigmoid2((self.o_classifier2(h)))
        out_stage_2 = F.interpolate(out_stage_2, size=(height, width), mode='bilinear', align_corners=True)
        im_out_stage_2 = self.GMP(out_stage_2).squeeze(-1).squeeze(-1)

        # stage#3
        h = self.relu3(self.conv3(h))
        h = self.pool3(h)  # 1/8
        out_stage_3 = self.sigmoid2((self.o_classifier3(h)))
        out_stage_3 = F.interpolate(out_stage_3, size=(height, width), mode='bilinear', align_corners=True)
        im_out_stage_3 = self.GMP(out_stage_3).squeeze(-1).squeeze(-1)


        if infer:
            o_output = 0.20 * out_stage_1 + 0.35 * out_stage_2 + 0.45 * out_stage_3
            # o_output = 0.5 * out_stage_1 + 0.5 * out_stage_2
            # im_output = self.GMP(o_output).squeeze(-1).squeeze(-1)
            return o_output
        else:
            o_output = 1 / 3 * out_stage_1 + 1 / 3 * out_stage_2 + 1 / 3 * out_stage_3
            # o_output = 0.5 * out_stage_1 + 0.5 * out_stage_2
            im_output = self.GMP(o_output).squeeze(-1).squeeze(-1)
            return o_output, out_stage_1, out_stage_2, out_stage_3, \
                   im_output, im_out_stage_1, im_out_stage_2, im_out_stage_3
            # return o_output, out_stage_1, out_stage_2, _, \
            #        im_output, im_out_stage_1, im_out_stage_2, _

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


if __name__ == "__main__":
    # 按照DCAN论文输入为480x480
    x = torch.rand((10, 3, 256, 256)).cuda()
    model = DWS_MIL().cuda()
    o_output, o_c1, o_c2, o_c3 = model(x)
    print('Object output size: ', o_output.size())
    print('Object output u1 size: ', o_c1.size())


