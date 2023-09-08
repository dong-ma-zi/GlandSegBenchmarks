"""
Author: wzh
Since: 2023-9-8
"""
import torch
import torch.nn as nn
import numpy as np
torch.cuda.set_device(1)


class DCAN(nn.Module):
    def __init__(self, n_class=3):
        # 本质上是FCN, backbone应该是Vgg
        super(DCAN, self).__init__()

        # conv1
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True)  # 1/16

        # conv5
        self.conv5 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 1024, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # object
        self.o_classifier1 = nn.Conv2d(n_class, n_class, 1)
        self.o_classifier2 = nn.Conv2d(n_class, n_class, 1)
        self.o_classifier3 = nn.Conv2d(n_class, n_class, 1)

        self.o_conv1 = nn.Conv2d(n_class, n_class, 1)
        self.o_conv2 = nn.Conv2d(n_class, n_class, 1)
        self.o_conv3 = nn.Conv2d(n_class, n_class, 1)

        self.o_upscore1 = nn.ConvTranspose2d(1024, n_class, 8, stride=8, bias=False)
        self.o_upscore2 = nn.ConvTranspose2d(512, n_class, 8, stride=8, bias=False)
        self.o_upscore3 = nn.ConvTranspose2d(512, n_class, 8, stride=8, bias=False)

        # contour
        self.c_classifier1 = nn.Conv2d(n_class, n_class, 1)
        self.c_classifier2 = nn.Conv2d(n_class, n_class, 1)
        self.c_classifier3 = nn.Conv2d(n_class, n_class, 1)

        self.c_conv1 = nn.Conv2d(n_class, n_class, 1)
        self.c_conv2 = nn.Conv2d(n_class, n_class, 1)
        self.c_conv3 = nn.Conv2d(n_class, n_class, 1)

        self.c_upscore1 = nn.ConvTranspose2d(1024, n_class, 8, stride=8, bias=False)
        self.c_upscore2 = nn.ConvTranspose2d(512, n_class, 8, stride=8, bias=False)
        self.c_upscore3 = nn.ConvTranspose2d(512, n_class, 8, stride=8, bias=False)

        # self._initialize_weights()

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

    def forward(self, x):
        h = x
        h = self.relu1(self.conv1(h))
        h = self.pool1(h)  # 1/2

        h = self.relu2(self.conv2(h))
        h = self.pool2(h)  # 1/4

        h = self.relu3(self.conv3(h))
        h = self.pool3(h)  # 1/8

        h = self.relu4(self.conv4(h))
        z4 = h
        h = self.pool4(h)  # 1/16

        h = self.relu5(self.conv5(h))
        z5 = h
        h = self.pool5(h)  # 1/16

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)
        z6 = h

        # object
        o_u1 = self.o_upscore1(z6)
        o_u2 = self.o_upscore2(z5)
        o_u3 = self.o_upscore3(z4)
        o_c1 = self.o_classifier1(self.o_conv1(o_u1))
        o_c2 = self.o_classifier2(self.o_conv2(o_u2))
        o_c3 = self.o_classifier3(self.o_conv3(o_u3))
        o_output = o_c1 + o_c2 + o_c3

        # contour
        c_u1 = self.c_upscore1(z6)
        c_u2 = self.c_upscore2(z5)
        c_u3 = self.c_upscore3(z4)
        c_c1 = self.c_classifier1(self.c_conv1(c_u1))
        c_c2 = self.c_classifier2(self.c_conv2(c_u2))
        c_c3 = self.c_classifier3(self.c_conv3(c_u3))
        c_output = c_c1 + c_c2 + c_c3

        return o_output, o_c1, o_c2, o_c3, c_output, c_c1, c_c2, c_c3


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
    x = torch.rand((10, 3, 480, 480)).cuda()
    model = DCAN().cuda()
    o_output, o_c1, o_c2, o_c3, c_output, c_c1, c_c2, c_c3 = model(x)
    print('Object output size: ', o_output.size())
    print('Contour output size: ', c_output.size())
    print('Object output u1 size: ', o_c1.size())


