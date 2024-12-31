"""Final layer for 1D u-net"""

import logging

logger = logging.getLogger(__name__)
import torch.nn as nn

from .conv_block import ConvBlock1D

class OutLayer(nn.Module):
    def __init__(self, name, in_channels, out_channels, num_convblocks, device, dtype):
        super(OutLayer, self).__init__()

        self.sequential = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=2,
                               stride=2,
                               device=device, dtype=dtype)
        )

        for i in range(num_convblocks - 1):
            self.sequential.add_module(f"convblock_{i}", ConvBlock1D(
                name=f"{name}_convblock_{i}",
                in_channels=out_channels,
                out_channels=out_channels,
                device=device, dtype=dtype))

        self.sequential.add_module(f"convblock_{num_convblocks}", ConvBlock1D(
            name=f"{name}_convblock_{num_convblocks}",
            in_channels=out_channels,
            out_channels=out_channels,
            device=device, dtype=dtype, last_layer=True))


    def forward(self, x):
        x = self.sequential(x)
        # print(f"OutLayer x size: {x.size()}")
        return x
