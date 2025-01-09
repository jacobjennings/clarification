"""1D u-net."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import logging

logger = logging.getLogger(__name__)
from ..modules import OutLayer, Down, ConvBlock1D, UpNoCat

def input_size_for_layer(layer_num, layer_sizes):
    layer_depths = []
    depth = 0
    for i in range(len(layer_sizes)):
        layer_depths.append(depth)
        if i < len(layer_sizes) // 2:
            depth += 1
        else:
            depth -= 1

    layer_size = 0
    layer_num_depth = layer_depths[layer_num]

    for i in range(layer_num + 1):
        relative_depth = layer_num_depth - layer_depths[i]
        adding_size = int(layer_sizes[i] * 2 ** relative_depth)
        layer_size += adding_size

    return layer_size


class ClarificationDense(nn.Module):
    def __init__(self, name, layer_sizes=None,
                 invert=False, num_output_convblocks=2,
                 device=None, dtype=torch.float32):
        super(ClarificationDense, self).__init__()
        self.layer_sizes = layer_sizes
        if device is None:
            device = torch.get_default_device()

        # if len(layer_sizes) % 2 == 0:
        #     raise ValueError("The number of layers must be odd.")

        layer_sizes_len = len(layer_sizes)
        self.invert = invert

        self.first_layer = ConvBlock1D(name=f"{name}_firstlayer_conv", in_channels=1,
                                       out_channels=layer_sizes[0], device=device, dtype=dtype)
        # print(f"First layer: in_channels: {in_channels} out_channels: {layer_sizes[0]}")

        self.down_layers = nn.ModuleDict()
        for i in range(len(layer_sizes) // 2):
            in_channels = input_size_for_layer(i, layer_sizes)
            out_channels = layer_sizes[i + 1]
            down = Down(name=f"{name}_down_{i}",
                        in_channels=in_channels,
                        out_channels=out_channels,
                        device=device, dtype=dtype)
            self.down_layers.add_module("down_" + str(i), down)

        self.up_layers = nn.ModuleDict()
        for i in range(layer_sizes_len // 2 - 1):
            in_channels = input_size_for_layer(layer_sizes_len // 2 + i, layer_sizes)
            out_channels = layer_sizes[layer_sizes_len // 2 + i + 1]
            up = UpNoCat(name=f"{name}_up_{i}",
                         in_channels=in_channels,
                         out_channels=out_channels,
                         device=device, dtype=dtype,
                         layer_num=i)
            self.up_layers.add_module("up_" + str(i), up)

        self.last_layer = OutLayer(
            name=f"{name}_outlayer",
            in_channels=input_size_for_layer(layer_sizes_len - 2, layer_sizes),
            out_channels=layer_sizes[-1],
            num_convblocks=num_output_convblocks, device=device, dtype=dtype)

    def forward(self, x):
        x = self.first_layer(x)
        outputs = [x]

        for i in range(len(self.down_layers)):
            down_layer = self.down_layers[f"down_{i}"]
            viewed_outputs = [output.view(outputs[-1].size(0), -1, outputs[-1].size(2)) for output in outputs]
            catted_outputs = torch.cat(viewed_outputs, dim=1)
            x = down_layer(catted_outputs)
            outputs.append(x)

        for i in range(len(self.up_layers)):
            viewed_outputs = [output.view(outputs[-1].size(0), -1, outputs[-1].size(2)) for output in outputs]
            catted_outputs = torch.cat(viewed_outputs, dim=1)
            up_layer = self.up_layers[f"up_{i}"]
            x = up_layer(catted_outputs)
            outputs.append(x)

        viewed_outputs = [output.view(outputs[-1].size(0), -1, outputs[-1].size(2)) for output in outputs]
        catted_outputs = torch.cat(viewed_outputs, dim=1)
        x = self.last_layer(catted_outputs)

        return x
