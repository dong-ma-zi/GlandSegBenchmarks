import torch
import torch.nn as nn
from .swin_transformer import SwinTransformer
import warnings
warnings.filterwarnings("ignore")

class Swin_MIL(nn.Module):
    def __init__(self, class_num=2, w=(0.3, 0.4, 0.3)):
        super(Swin_MIL, self).__init__()
        self.backbone = SwinTransformer(depths=(2, 2, 6, 2), out_indices=(0, 1, 2))

        self.decoder1 = nn.Sequential(
            nn.Conv2d(96, class_num, 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Sigmoid()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(192, class_num, 1),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
            nn.Sigmoid()
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(384, class_num, 1),
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
            nn.Sigmoid()
        )

        self.w = w

    def pretrain(self):
        for w in self.modules():
            if isinstance(w, nn.Conv2d):
                nn.init.xavier_uniform_(w.weight)
        self.backbone.init_weights()
        model_dict = self.state_dict()
        pretrained_dict = torch.load("/home/data1/my/Project/GlandSegBenchmark/SA-MIL/pretrained_model/swin_tiny_patch4_window7_224.pth")['model']
        new_dict = {'backbone.' + k: v for k, v in pretrained_dict.items() if 'backbone.' + k in model_dict.keys()}
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        x1, x2, x3 = self.backbone(x)

        x1 = self.decoder1(x1)
        x2 = self.decoder2(x2)
        x3 = self.decoder3(x3)

        x = self.w[0] * x1 + self.w[1] * x2 + self.w[2] * x3

        return x, x1, x2, x3


if __name__ == "__main__":
    model = Swin_MIL()
    x = torch.randn(1, 3, 224, 224)
    outs = model(x)[0]

    # t = torch.pow(torch.pow(outs, 4).mean(-1).mean(-1), 1 / 4)
    # g = torch.pow(torch.pow(outs.reshape(outs.shape[0], -1), 4).mean(dim=1).unsqueeze(1), 1 / 4)
    for out in outs:
        print(out.shape)