"""Down layers for 1d u-net"""

import torch.nn as nn

from ..modules import ConvBlock1D


class Down(nn.Module):
    def __init__(self, name, in_channels, out_channels, device, dtype):
        super(Down, self).__init__()

        self.sequential = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            ConvBlock1D(name=f"{name}_conv", in_channels=in_channels, out_channels=out_channels, device=device, dtype=dtype)
        )

    def forward(self, x):
        x = self.sequential(x)
        # print(f"Down out: {x.size()}")
        return x
