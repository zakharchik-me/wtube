import torch.nn as nn
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../../../utils")))
from utils.registry import register

@register("detector")
class FlexibleSegmentationModel(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        out_channels: int = 1,
        num_blocks: int = 3,
        base_channels: int = 64,
        use_fc: bool = True,
        final_upsample: bool = False
    ):
        super(FlexibleSegmentationModel, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        in_ch = input_channels

        for i in range(num_blocks):
            out_ch = base_channels * (2 ** i)
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ))
            in_ch = out_ch

        fc_layers = []
        if use_fc:
            fc_ch1 = base_channels * (2 ** num_blocks)
            fc_ch2 = base_channels * (2 ** (num_blocks - 1))
            fc_layers.extend([
                nn.Conv2d(in_ch, fc_ch1, kernel_size=1),
                nn.BatchNorm2d(fc_ch1),
                nn.ReLU(inplace=True),
                nn.Conv2d(fc_ch1, fc_ch2, kernel_size=1),
                nn.BatchNorm2d(fc_ch2),
                nn.ReLU(inplace=True),
            ])
            in_ch = fc_ch2
        self.bottleneck = nn.Sequential(*fc_layers) if fc_layers else nn.Identity()

        for i in reversed(range(num_blocks)):
            out_ch = base_channels * (2 ** i)
            self.decoder.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch

        self.final_upsample = nn.Upsample(scale_factor=2, mode="nearest") if final_upsample else nn.Identity()
        self.detector = nn.Conv2d(in_ch, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for block in self.encoder:
            x = block(x)
        x = self.bottleneck(x)
        for block in self.decoder:
            x = block(x)
        x = self.final_upsample(x)
        x = self.sigmoid(self.detector(x))
        return x
