"""
Author: wzh
Since: 2023-9-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.aspp import ASPP

def upsample(image, in_channels, out_channels, scale_factor):
    image_upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)(image)
    #image_upsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)(image_upsample)
    return image_upsample

def downsample(image, downsample):
    image_downsample = nn.Upsample(scale_factor=downsample, mode='bilinear', align_corners=True)(image)
    return image_downsample

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.up(x)
        return x


class MILUnit(nn.Module):
    def __init__(self, downsample, in_channels, out_channels):
        super(MILUnit, self).__init__()
        self.downsample = downsample

        ## conv1
        self.conv1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(True)
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        ## downsample conv
        self.downsample_conv = nn.Conv2d(3, in_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(True)

        ## concat conv3
        self.concat_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(True)

        self.relu = nn.ReLU(True)

    def forward(self, x, img):
        h_1 = self.conv1_2(self.relu1(self.conv1_1(x)))

        h_2 = downsample(img, self.downsample)
        h_2 = self.relu2(self.downsample_conv(h_2))

        h_2 = torch.cat([x, h_2], dim=1)
        h_2 = self.relu3(self.concat_conv(h_2))
        return self.relu(h_1 + h_2)


class ResidualUnit(nn.Module):
    def __init__(self, in_channels):
        super(ResidualUnit, self).__init__()
        ## conv1
        self.conv1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(True)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return x + self.conv1_2(self.relu1(self.conv1_1(x)))


class DilatedResidualUnit(nn.Module):
    def __init__(self, in_channels, dilation_rate):
        super(DilatedResidualUnit, self).__init__()
        ## dilated conv1
        self.conv1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate)
        self.relu1 = nn.ReLU(True)
        self.conv1_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate)

    def forward(self, x):
        return x + self.conv1_2(self.relu1(self.conv1_1(x)))


class MILDNet(nn.Module):
    def __init__(self, n_class=2, init_weights=True):
        super(MILDNet, self).__init__()

        ## conv1 maxpooling
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU(True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu1_2 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ## MIL Unit1 and Residual Unit1
        self.mil1 = MILUnit(1/2, 64, 128)
        self.res1 = ResidualUnit(128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ## MIL Unit2 and Residual Unit2
        self.mil2 = MILUnit(1/4, 128, 256)
        self.res2 = ResidualUnit(256)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ## MIL Unit3 and Residual Unit3
        self.mil3 = MILUnit(1/8, 256, 512)
        self.res3 = ResidualUnit(512)
        self.dilated_res1 = DilatedResidualUnit(512, 2)
        self.dilated_res2 = DilatedResidualUnit(512, 2)
        self.dilated_res3 = DilatedResidualUnit(512, 4)
        self.dilated_res4 = DilatedResidualUnit(512, 4)

        ## ASPP Unit
        self.aspp = ASPP(512, 640)

        ## Upsample and conv2_1, conv2_2
        self.upsample1 = Up(256, 640)
        self.conv2_1 = nn.Conv2d(1280, 256, kernel_size=3, stride=1, padding=1)
        self.relu2_1 = nn.ReLU(True)
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu2_2 = nn.ReLU(True)

        ## Upsample and conv3_1, conv3_2
        self.upsample2 = Up(128, 256)
        self.conv3_1 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        self.relu3_1 = nn.ReLU(True)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu3_2 = nn.ReLU(True)
        self.upsample3 = Up(64, 128)

        self.object_classifier = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 2, kernel_size=1, stride=1)
        )

        self.contour_classifier = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 2, kernel_size=1, stride=1)
        )

        ## deep supervision
        self.auxilary_object_classifier = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 2, kernel_size=1, stride=1)
        )

        self.auxilary_contour_classifier = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 2, kernel_size=1, stride=1)
        )


    def forward(self, x):
        _, _, height, width = x.size()
        ori_img = x
        h = x

        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)  # 1/2
        z3 = h

        h = self.mil1(h, ori_img)
        h = self.res1(h)
        h = self.pool2(h)  # 1/4
        z2 = h

        h = self.mil2(h, ori_img)
        h = self.res2(h)
        h = self.pool3(h)  # 1/8
        z1 = h

        h = self.mil3(h, ori_img)
        h = self.res3(h)
        h = self.dilated_res1(h)
        h = self.dilated_res2(h)
        z = h
        h = self.dilated_res3(h)
        h = self.dilated_res4(h)

        h = self.aspp(h)

        h = self.upsample1(z1, h)
        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))

        h = self.upsample2(z2, h)
        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))

        h = self.upsample3(z3, h)
        output_o = self.object_classifier(h)
        output_c = self.contour_classifier(h)

        ## deep supervision
        a_output_o = upsample(z, 512, 512, 8)
        a_output_o = self.auxilary_object_classifier(a_output_o)

        a_output_c = upsample(z, 512, 512, 8)
        a_output_c = self.auxilary_contour_classifier(a_output_c)

        return output_o, a_output_o, output_c, a_output_c


if __name__ == "__main__":
    # 按照MILD-Net论文输入为464x464
    x = torch.rand((4, 3, 480, 480)).cuda()
    model = MILDNet().cuda()
    output_o, a_output_o, output_c, a_output_c = model(x)
    print('Object output size: ', output_o.size())
    print('Contour output size: ', output_c.size())
    print('Auxilary object output size: ', a_output_o.size())





