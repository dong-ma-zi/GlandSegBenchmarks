from torch import nn
import torch
import torch.nn.functional as F


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.atrous_blocks1 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.atrous_blocks2 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.atrous_blocks3 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.atrous_blocks4 = nn.Conv2d(in_channels, out_channels, 1)

        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            nn.ReLU())


    def forward(self, x):
        size = x.shape[2:]
        x = self.initial_conv(x)
        h1 = self.atrous_blocks1(x)
        h2 = self.atrous_blocks2(x)
        h3 = self.atrous_blocks3(x)
        h4 = self.mean(self.atrous_blocks4(x))
        h4 = F.interpolate(h4, size, mode='bilinear', align_corners=False)

        h = self.output_conv(torch.cat([h1, h2, h3, h4], dim=1))
        return h


if __name__ == "__main__":
    aspp = ASPP(256, 512)
    x = torch.rand(2, 256, 13, 13)
    print(aspp(x).shape)