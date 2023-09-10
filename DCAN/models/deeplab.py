import torch
import torch.nn as nn
from torchvision import models
torch.cuda.set_device(1)
class DCAN(nn.Module):
    def __init__(self, n_class=21, split='train', init_weights=True):
        super(DCAN, self).__init__()
        self.split = split

        ### conv1_1 conv1_2 maxpooling
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU(True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu1_2 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ### conv2_1 conv2_2 maxpooling
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2_1 = nn.ReLU(True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu2_2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ### conv3_1 conv3_2 conv3_3 maxpooling
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_1 = nn.ReLU(True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_2 = nn.ReLU(True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu3_3 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ### conv4_1 conv4_2 conv4_3 maxpooling(stride=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_1 = nn.ReLU(True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_2 = nn.ReLU(True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.relu4_3 = nn.ReLU(True)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        ### conv5_1 conv5_2 conv5_3 (dilated convolution dilation=2, padding=2)
        ### maxpooling(stride=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.relu5_1 = nn.ReLU(True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.relu5_2 = nn.ReLU(True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.relu5_3 = nn.ReLU(True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        ### average pooling
        self.avgPool5 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        ### fc6 relu6 drop6
        self.conv6_1 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=12, dilation=12)
        self.relu6_1 = nn.ReLU(True)
        self.drop6 = nn.Dropout2d(0.5)

        ### fc7 relu7 drop7 (kernel_size=1, padding=0)
        self.conv7_1 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)
        self.relu7_1 = nn.ReLU(True)
        self.drop7 = nn.Dropout2d(0.5)

        ### fc8
        # self.conv8 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, padding=0)

        ### DCAN part
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

        if init_weights:
            self._initialize_weights()


    def forward(self, x):
        _, _, height, width = x.size()
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)  # 1/2

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)  # 1/4

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        z4 = h
        h = self.pool4(h)  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        z5 = h
        h = self.pool5(h)  # 1/16
        h = self.avgPool5(h)

        h = self.relu6_1(self.conv6_1(h))
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

        if self.split == 'test':
            o_output = nn.functional.interpolate(o_output, size=(height, width), mode='bilinear',
                                               align_corners=True)
            c_output = nn.functional.interpolate(c_output, size=(height, width), mode='bilinear',
                                               align_corners=True)
        return o_output, o_c1, o_c2, o_c3, c_output, c_c1, c_c2, c_c3

    def _initialize_weights(self):
        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                nn.init.normal_(m[1].weight.data, mean=0, std=0.01)
                nn.init.constant_(m[1].bias.data, 0.0)


if __name__ == "__main__":
    # 按照DCAN论文输入为480x480
    x = torch.rand((4, 3, 480, 480)).cuda()
    model = DCAN().cuda()
    model.load_state_dict(torch.load("../DCAN_pretrained_weight.pth"), strict=False)
    o_output, o_c1, o_c2, o_c3, c_output, c_c1, c_c2, c_c3 = model(x)
    print('Object output size: ', o_output.size())
    print('Contour output size: ', c_output.size())
    print('Object output u1 size: ', o_c1.size())