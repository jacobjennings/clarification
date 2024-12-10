"""Final layer for 1D u-net"""

import torch.nn as nn

from .conv_block import ConvBlock1D

class OutLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_convblocks, device, dtype):
        super(OutLayer, self).__init__()

        self.sequential = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
        )

        for i in range(num_convblocks - 1):
            self.sequential.add_module(f"convblock_{i}", ConvBlock1D(
                in_channels=in_channels, out_channels=in_channels, device=device, dtype=dtype))

        self.sequential.add_module(f"convblock_{i}", ConvBlock1D(
            in_channels=in_channels, out_channels=out_channels, device=device, dtype=dtype))


    def forward(self, x):
        return self.sequential(x)
