"""1D u-net."""
import logging

logger = logging.getLogger(__name__)
import torch.nn as nn
import torch.nn.functional as nnF

from ..modules import OutLayer, Down, ConvBlock1D, Up


class ClarificationSimple(nn.Module):
    def __init__(self, name, device, dtype, layer_sizes=None, invert=False, num_output_convblocks=2):
        super(ClarificationSimple, self).__init__()

        if len(layer_sizes) % 2 == 0:
            raise ValueError("The number of layers must be odd.")

        self.invert = invert
        if layer_sizes is None:
            layer_sizes = [64, 128, 256, 512, 1024]

        self.first_layer = ConvBlock1D(name=f"{name}_firstlayer_conv", in_channels=1, out_channels=layer_sizes[0], device=device, dtype=dtype)

        self.down_layers = nn.ModuleList()
        for i in range(len(layer_sizes) // 2):
            down = Down(name=f"{name}_down_{i}", in_channels=layer_sizes[i], out_channels=layer_sizes[i + 1], device=device, dtype=dtype)
            self.down_layers.add_module("down_" + str(i), down)

        layer_sizes_len = len(layer_sizes)
        self.up_layers = nn.ModuleList()
        for i in range(layer_sizes_len // 2 - 1):
            a_in_channels = layer_sizes[layer_sizes_len // 2 + i]
            print(f"layer_sizes {layer_sizes}, layer_sizes_len // 2 + i + 1: {layer_sizes_len // 2 + i + 1}")
            out_channels = layer_sizes[layer_sizes_len // 2 + i + 1]
            up = Up(name=f"{name}_up_{i}",
                    in_channels=a_in_channels,
                    out_channels=out_channels,
                    device=device, dtype=dtype,
                    layer_num=i)
            self.up_layers.add_module("up_" + str(i), up)

        print(f"Out layer: in_channels: {layer_sizes[-1]} out_channels: 1")
        self.last_layer = OutLayer(
            name=f"{name}_outlayer",
            in_channels=layer_sizes[-2],
            out_channels=layer_sizes[-1],
            num_convblocks=num_output_convblocks, device=device, dtype=dtype)

    def forward(self, initial_x):
        x = self.first_layer(initial_x)
        down_outputs = [x]
        for down_layer in self.down_layers:
            x = down_layer(x)
            down_outputs.append(x)

        down_outputs_reversed = list(reversed(down_outputs))

        # print(f"down outputs sizes: {[d.size() for d in down_outputs]}")

        for (i, up_layer) in enumerate(self.up_layers):
            # print(f"up layer {i} x size: {x.size()}")
            x = up_layer(x, down_outputs_reversed[i + 1])

        x = self.last_layer(x)

        # print(f"ClarificationSimple x size: {x.size()}")
        x = x.squeeze(0).squeeze(0)
        # print(f"ClarificationSimple x size after sq: {x.size()}")
        if self.invert:
            x = initial_x - x

        return x

