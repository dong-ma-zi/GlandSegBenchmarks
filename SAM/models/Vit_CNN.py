import torch.nn.functional as F
import torch
import torch.nn as nn
from .bulid_sam_encoder import sam_encoder_registry

####
class Vit_CNN(nn.Module):
    """Initialise HoVer-Net."""

    def __init__(
            self,
            args,
            num_types=None,
            vit_mode: str = 'vit_h'
            ) -> None:
        super().__init__()

        self.num_types = num_types

        self.image_encoder = sam_encoder_registry[vit_mode](args)
        self.decoder = nn.Sequential(nn.Conv2d(256, 64, 1, stride=1, padding=0, bias=False),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64, num_types, 1, stride=1, padding=0, bias=True)
                                    )


    def forward(self, x):

        _, _, original_h, original_w = x.shape
        x = F.interpolate(
            x,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )

        image_embedding = self.image_encoder(x)  # [B, 256, 64, 64]
        out = self.decoder(image_embedding)

        out = F.interpolate(
            out,
            (original_h, original_w),
            mode="bilinear",
            align_corners=False,
        )
        return out


if __name__ == '__main__':

    # x = torch.load("/home/data1/my/Project/segment-anything-main/sam_vit_b.pth")
    # y = 1
    net = Vit_CNN(num_types=6)
    x = torch.randn(size=(4, 3, 1000, 1244))
    y = net(x)
    t = 1