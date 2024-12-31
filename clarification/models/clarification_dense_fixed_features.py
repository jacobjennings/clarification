"""1D u-net."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as nnF

from ..modules import OutLayer, Down, ConvBlock1D, UpNoCat


class ClarificationDenseFF(nn.Module):
    def __init__(self, name, in_channels, device, dtype, layer_size, layer_count, num_output_convblocks=2):
        super(ClarificationDenseFF, self).__init__()
        

        self.first_layer = ConvBlock1D(name=f"{name}_firstlayer_conv", in_channels=in_channels, out_channels=layer_sizes[0], device=device, dtype=dtype)
        # print(f"First layer: in_channels: {in_channels} out_channels: {layer_sizes[0]}")
        
        output_sizes = [layer_sizes[0]]
        self.down_layers = nn.ModuleDict()
        for i in range(len(layer_sizes) // 2):
            input_size = sum(output_sizes)
            out_channels = layer_sizes[i + 1]
            output_sizes.append(out_channels)
            down = Down(name=f"{name}_down_{i}", in_channels=input_size, out_channels=out_channels, device=device, dtype=dtype)
            self.down_layers.add_module("down_" + str(i), down)

        self.up_layers = nn.ModuleDict()
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
        x = initial_x
        outputs = []
        for i in range(len(self.layer_sizes)):
            x, initial_x, outputs = self.compute_checkpoint(x, initial_x, outputs, i)

    def checkpoint_count(self):
        return 1 + len(self.down_layers) + len(self.up_layers) + 2

    def compute_checkpoint(self, x, initial_x, outputs, checkpoint_index):
        if checkpoint_index == 0:
            x = self.first_layer(x)
            return x, initial_x, [x]
        elif checkpoint_index < len(self.down_layers) + 1:
            down_layer = self.down_layers[f"down_{checkpoint_index - 1}"]
            processed_inputs = [
                nn.functional.interpolate(output, size=x.size()[2], mode='nearest')
                for output in outputs]
            x = torch.cat(tuple(processed_inputs), dim=1)
            x = down_layer(x)
            return x, initial_x, outputs + [x]
        elif checkpoint_index < len(self.down_layers) + len(self.up_layers) + 1:
            up_layer = self.up_layers[f"up_{checkpoint_index - len(self.down_layers) - 1}"]
            processed_inputs = [
                nn.functional.interpolate(output, size=x.size()[2], mode='nearest')
                for output in outputs]
            x = torch.cat(tuple(processed_inputs), dim=1)
            x = up_layer(x)
            return x, initial_x, outputs + [x]
        elif checkpoint_index == self.checkpoint_count() - 2:
            # This cat is memory hungry so it gets its own checkpoint
            print(f"outputs sizes: {[d.size() for d in outputs]}")
            processed_inputs = [
                nn.functional.interpolate(output, size=x.size()[2], mode='nearest')
                for output in outputs]

            del outputs

            x = torch.cat(tuple(processed_inputs), dim=1)
            del processed_inputs
            return x, None, None

        # last layer    
        x = self.last_layer(x)

        x = x.squeeze(0).squeeze(0)
        if self.invert:
            x = initial_x - x

        return x, None, None
