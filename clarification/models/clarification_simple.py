"""1D u-net."""

import torch.nn as nn
import torch.nn.functional as nnF

from ..modules import OutLayer, Down, ConvBlock1D, Up


class ClarificationSimple(nn.Module):
    def __init__(self, in_channels, samples_per_batch, device, dtype, layer_sizes=None):
        super(ClarificationSimple, self).__init__()

        if layer_sizes is None:
            layer_sizes = [64, 128, 256, 512, 1024]

        self.first_layer = ConvBlock1D(in_channels=in_channels, out_channels=layer_sizes[0], device=device, dtype=dtype)

        self.down_layers = [
            Down(in_channels=layer_sizes[i], out_channels=layer_sizes[i + 1], device=device, dtype=dtype)
            for i in range(len(layer_sizes) - 1)
        ]
        self.down_layers_module_list = nn.ModuleList(self.down_layers)

        self.up_layers = []
        for i in range(len(layer_sizes) - 2):
            layer_depth = len(layer_sizes) - 2 - i
            up = Up(in_channels=layer_sizes[-(i + 1)], out_channels=layer_sizes[-(i + 2)],
                    layer_depth=layer_depth, samples_per_batch=samples_per_batch, device=device, dtype=dtype)
            self.up_layers.append(up)

        self.up_layers_module_list = nn.ModuleList(self.up_layers)

        self.last_layer = OutLayer(
            in_channels=layer_sizes[1],
            out_channels=in_channels,
            num_convblocks=2, device=device, dtype=dtype)

    def forward(self, x):
        x = self.first_layer(x)
        down_outputs = []
        for down_layer in self.down_layers:
            x = down_layer(x)
            down_outputs.append(x)

        down_outputs_reversed = list(reversed(down_outputs))

        for (i, up_layer) in enumerate(self.up_layers):
            x = up_layer(x, down_outputs_reversed[i + 1])

        x = self.last_layer(x)
        x = x.squeeze(0).squeeze(0)

        return x

