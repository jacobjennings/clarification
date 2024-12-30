"""1D u-net."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as nnF

from ..modules import OutLayer, Down, ConvBlock1D, UpNoCat


class ClarificationDense(nn.Module):
    def __init__(self, name, in_channels, device, dtype, layer_sizes=None, invert=False, num_output_convblocks=2):
        super(ClarificationDense, self).__init__()

        if len(layer_sizes) % 2 == 0:
            raise ValueError("The number of layers must be odd.")

        layer_sizes_len = len(layer_sizes)
        self.invert = invert

        self.first_layer = ConvBlock1D(name=f"{name}_firstlayer_conv", in_channels=in_channels, out_channels=layer_sizes[0], device=device, dtype=dtype)
        # print(f"First layer: in_channels: {in_channels} out_channels: {layer_sizes[0]}")
        
        output_sizes = [layer_sizes[0]]
        self.down_layers = nn.ModuleList()
        for i in range(len(layer_sizes) // 2):
            input_size = sum(output_sizes)
            out_channels = layer_sizes[i + 1]
            output_sizes.append(out_channels)
            down = Down(name=f"{name}_down_{i}", in_channels=input_size, out_channels=out_channels, device=device, dtype=dtype)
            self.down_layers.add_module("down_" + str(i), down)

        self.up_layers = nn.ModuleList()
        for i in range(layer_sizes_len // 2 - 1):
            input_size = sum(output_sizes)
            out_channels = layer_sizes[layer_sizes_len // 2 + i + 1]
            output_sizes.append(out_channels)
            up = UpNoCat(name=f"{name}_up_{i}",
                        in_channels=input_size,
                        out_channels=out_channels,
                        device=device, dtype=dtype,
                        layer_num=i)
            self.up_layers.add_module("up_" + str(i), up)

        # print(f"Out layer: in_channels: {layer_sizes[-1]} out_channels: 1")
        self.last_layer = OutLayer(
            name=f"{name}_outlayer",
            in_channels=sum(output_sizes),
            out_channels=layer_sizes[-1],
            num_convblocks=num_output_convblocks, device=device, dtype=dtype)

    def forward(self, initial_x):
        x = self.first_layer(initial_x)
        outputs = [x]
        for i, down_layer in enumerate(self.down_layers):
            processed_inputs = [
                nn.functional.interpolate(output, size=x.size()[2], mode='nearest')
                for output in outputs]

            # print(f"cd 1 down layer {i} x.size() {x.size()} extra_inputs_processed sizes: {[d.size() for d in processed_inputs]}")
            x = torch.cat(tuple(processed_inputs), dim=1)
            # print(f"cd 2 down layer {i} x.size() {x.size()}")

            x = down_layer(x)
            outputs.append(x)

        for (i, up_layer) in enumerate(self.up_layers):
            # print(f"outputs sizes: {[d.size() for d in outputs]}")
            # print(f"up layer {i} x size: {x.size()}")
            processed_inputs = [
                nn.functional.interpolate(output, size=x.size()[2], mode='nearest')
                for output in outputs]
            
            x = torch.cat(tuple(processed_inputs), dim=1)

            x = up_layer(x)
            outputs.append(x)

        processed_inputs = [
            nn.functional.interpolate(output, size=x.size()[2], mode='nearest')
            for output in outputs]

        x = torch.cat(tuple(processed_inputs), dim=1)
        # print(f"outputs sizes: {[d.size() for d in outputs]}")

        x = self.last_layer(x)

        # print(f"ClarificationDense x size: {x.size()}")
        x = x.squeeze(0).squeeze(0)
        # print(f"ClarificationDense x size after sq: {x.size()}")
        if self.invert:
            x = initial_x - x

        return x

