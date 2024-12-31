"""1D u-net."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as nnF

from ..modules import ResBlock1D, ConvBlock1D


class ClarificationResNet(nn.Module):
    def __init__(self, name, channel_size, layer_count, device, dtype):
        super(ClarificationResNet, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=channel_size, stride=1, kernel_size=1, device=device, dtype=dtype),
            nn.BatchNorm1d(num_features=channel_size, device=device, dtype=dtype),
            nn.ReLU(inplace=True),
        )

        self.resblocks = nn.ModuleDict()

        for i in range(layer_count):
            self.resblocks.add_module(
                "resblock_" + str(i),
                ResBlock1D(
                    name=f"{name}_resblock_{i}",
                    channel_size=channel_size,
                    device=device,
                    dtype=dtype))

        self.last_layer = nn.Sequential(
            nn.Conv1d(in_channels=channel_size + 1, out_channels=channel_size + 1, stride=1, kernel_size=1, device=device, dtype=dtype),
            nn.BatchNorm1d(num_features=channel_size + 1, device=device, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=channel_size + 1, out_channels=1, stride=1, kernel_size=1, device=device,
                      dtype=dtype),

        )

    def forward(self, initial_x):
        x = initial_x
        outputs = []
        for i in range(self.checkpoint_count()):
            x, initial_x, outputs = self.compute_checkpoint(x, initial_x, outputs, i)

        del outputs
        return x

    def checkpoint_count(self):
        return len(self.resblocks) + 2

    def compute_checkpoint(self, x, initial_x, outputs, checkpoint_index):
        if checkpoint_index == 0:
            x = self.first_layer(x)
            return x, initial_x, None
        if checkpoint_index == self.checkpoint_count() - 1:
            x = torch.cat([x, initial_x], dim=1)
            x = self.last_layer(x)
            return x, None, None
        x = self.resblocks[f"resblock_{checkpoint_index - 1}"](x)
        return x, initial_x, None
