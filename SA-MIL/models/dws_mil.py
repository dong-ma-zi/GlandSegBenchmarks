import torch
import torch.nn as nn
import torchvision


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DWS_MIL(nn.Module):
    def __init__(self, class_num=2):
        super(DWS_MIL, self).__init__()
        self.stage1 = nn.Sequential(
            Conv(3, 64),
            Conv(64, 64)
        )
        self.stage2 = nn.Sequential(
            Conv(64, 128),
            Conv(128, 128)
        )
        self.stage3 = nn.Sequential(
            Conv(128, 256),
            Conv(256, 256),
            Conv(256, 256)
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, class_num, 1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Sigmoid()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, class_num, 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Sigmoid()
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, class_num, 1),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
            nn.Sigmoid()
        )

        self.w = [0.2, 0.35, 0.45]

    def pretrain(self):

        model_pre = torchvision.models.vgg16(pretrained=True)
        self.stage1[0].conv[0] = model_pre.features[0]
        self.stage1[1].conv[0] = model_pre.features[2]
        self.stage2[0].conv[0] = model_pre.features[5]
        self.stage2[1].conv[0] = model_pre.features[7]
        self.stage3[0].conv[0] = model_pre.features[10]
        self.stage3[1].conv[0] = model_pre.features[12]
        self.stage3[2].conv[0] = model_pre.features[14]

    def forward(self, x):
        x = self.stage1(x)
        x = self.pool(x)
        x1 = x

        x = self.stage2(x)
        x = self.pool(x)
        x2 = x

        x = self.stage3(x)
        x = self.pool(x)
        x3 = x

        x1 = self.decoder1(x1)
        x2 = self.decoder2(x2)
        x3 = self.decoder3(x3)

        x = self.w[0] * x1 + self.w[1] * x2 + self.w[2] * x3

        return x, x1, x2, x3


if __name__ == "__main__":
    model = DWS_MIL()
    x = torch.randn(1, 3, 224, 224)
    outs = model(x)[0]

    # t = torch.pow(torch.pow(outs, 4).mean(-1).mean(-1), 1 / 4)
    # g = torch.pow(torch.pow(outs.reshape(outs.shape[0], -1), 4).mean(dim=1).unsqueeze(1), 1 / 4)
    for out in outs:
        print(out.shape)