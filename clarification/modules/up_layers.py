"""Up layers for 1d u-net"""
import math
import logging

logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
import torch.nn.functional as nnF

from . import ConvBlock1D
from . import Glue1D


class Up(nn.Module):
    def __init__(self, name, in_channels, out_channels, device, dtype, layer_num=None):
        super(Up, self).__init__()
        self.transpose = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2,
                                            stride=2, device=device, dtype=dtype)
        self.convBlock = ConvBlock1D(name=f"{name}_conv",
                                     in_channels=out_channels * 2,
                                     out_channels=out_channels,
                                     device=device, dtype=dtype)
        self.layer_num = layer_num

    def forward(self, x1, x2):
        # print(f"1 Up layer in: x1: {x1.size()} x2: {x2.size()} layer {self.layer_num}")
        x1 = self.transpose(x1)
        # print(f"2 Up layer: x1: {x1.size()} layer {self.layer_num}")
        diff = x2.size()[2] - x1.size()[2]  # Calculate difference correctly
        # print(f"3 diff: {diff}")
        x1 = nnF.pad(x1, (diff // 2, diff - diff // 2))
        # print(f"4 Up layer:  x1: {x1.size()} x2: {x2.size()} diff: {diff}  layer {self.layer_num}")
        x = torch.cat([x2, x1], dim=1)

        # print(f"5  Up layer:  x: {x.size()} x2: {x2.size()} layer {self.layer_num}")

        x = self.convBlock(x)
        # print(f"6  Up layer out:  x: {x.size()} x2: {x2.size()} diff: {diff} layer {self.layer_num} ")

        return x
    
class UpNoCat(nn.Module):
    def __init__(self, name, in_channels, out_channels, device, dtype, layer_num=None):
        super(UpNoCat, self).__init__()
        self.transpose = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2,
                                            stride=2, device=device, dtype=dtype)
        self.convBlock = ConvBlock1D(name=f"{name}_conv",
                                     in_channels=out_channels,
                                     out_channels=out_channels,
                                     device=device, dtype=dtype)
        self.layer_num = layer_num

    def forward(self, x):
        # print(f"1 Up layer in: x1: {x1.size()} x2: {x2.size()} layer {self.layer_num}")
        x = self.transpose(x)
        # print(f"2 Up layer: x: {x.size()} layer {self.layer_num}")
        x = self.convBlock(x)
        # print(f"6  Up layer out:  x: {x.size()} x2: {x2.size()} diff: {diff} layer {self.layer_num} ")

        return x


class UpWithLSTMInput(nn.Module):
    def __init__(self, in_channels, out_channels, layer_sizes, layer_num, lstm_out_sample_size, samples_per_batch,
                 device, dtype):
        super(UpWithLSTMInput, self).__init__()

        self.layer_num = layer_num

        max_layer_size = 0
        for layer_size in layer_sizes[1:]:
            max_layer_size = max(max_layer_size, layer_size)

        first_up_layer_sample_size = samples_per_batch
        for i in range(len(layer_sizes) - 1):
            first_up_layer_sample_size = first_up_layer_sample_size // 2

        in_sample_size = first_up_layer_sample_size * 2
        for i in range(layer_num):
            in_sample_size = in_sample_size * 2

        # print(f"out_channels: {out_channels}")

        self.transpose = nn.ConvTranspose1d(in_channels=in_channels,
                                            out_channels=in_channels // 2,
                                            kernel_size=2,
                                            stride=2,
                                            device=device, dtype=dtype)
        self.conv_block = ConvBlock1D(in_channels=in_channels, out_channels=out_channels, device=device, dtype=dtype)
        self.lstm_matcher = Glue1D(
            in_channels=max_layer_size,
            out_channels=in_channels,
            in_sample_size=lstm_out_sample_size,
            out_sample_size=in_sample_size,
            device=device, dtype=dtype)

    def forward(self, x, down_output, lstm_output):
        lstm_time_normalized = self.lstm_matcher(lstm_output)
        x = self.transpose(x)
        diff = down_output.size()[1] - x.size()[1]  # Calculate difference correctly
        x = nnF.pad(x, (diff // 2, diff - diff // 2))
        x = torch.cat([down_output, x], dim=1)
        x = x + lstm_time_normalized
        x = self.conv_block(x)

        return x
