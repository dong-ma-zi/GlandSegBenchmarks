import torch
from . import initialization as init
import torch.nn.functional as F

class SegmentationModel(torch.nn.Module):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = 8 # self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            # raise RuntimeError(
            #     f"Wrong input shape height={h}, width={w}. Expected image height and width "
            #     f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            # )
            return new_h, new_w
        return None, None

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        # self.check_input_shape(x)

        # 计算需要填充的大小
        n, c, h, w = x.shape
        target_h, target_w = self.check_input_shape(x)
        if target_h != None or target_w != None:
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=True)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        if target_h != None or target_w != None:
            masks = F.interpolate(masks, size=(h, w), mode='bilinear', align_corners=True)

        return masks

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x
