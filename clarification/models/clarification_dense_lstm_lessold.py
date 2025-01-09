"""1D u-net."""

import math
import logging

logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
import torch.nn.functional as nnF

from ..modules import OutLayer, Down, ConvBlock1D, UpNoCat, Glue1D
# from ..models import ClarificationDense

class ClarificationDenseLSTMLessOld(nn.Module):
    def __init__(self, name, in_channels, samples_per_batch, device=None, dtype=torch.float32, layer_sizes=None, invert=False, num_output_convblocks=2):
        super(ClarificationDenseLSTMLessOld, self).__init__()
        hidden_size_multiplier = 4

        if device is None:
            device = torch.get_default_device()

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

        # self.pre_lstm_reduction = Glue1D(in_channels=layer_sizes[len(layer_sizes) // 2 + 1], out_channels=max_layer_size,
        #                                  in_sample_size=samples_per_batch // 2, out_sample_size=input_size,
        #                                  device=device, dtype=dtype)
        
        # lstm_input_channel_size = layer_sizes[len(layer_sizes) // 2] // 8

        time_dim = samples_per_batch
        for i in range(len(layer_sizes) // 2):
            print(f"Processing time_dim: {time_dim}")
            time_dim = math.ceil(time_dim / 2)

        lstm_input_size = math.ceil(time_dim)
        lstm_kernel_stride_size = 16
        lstm_input_size //= lstm_kernel_stride_size

        down_layer_size_before_lstm = layer_sizes[len(layer_sizes) // 2]

        print(f"lstm_input_size: {lstm_input_size} layer_sizes[len(layer_sizes) // 2]: {layer_sizes[len(layer_sizes) // 2]}")
        self.lstm_conv = nn.Conv1d(in_channels=layer_sizes[len(layer_sizes) // 2],
                                   out_channels=down_layer_size_before_lstm // hidden_size_multiplier, kernel_size=lstm_kernel_stride_size, stride=lstm_kernel_stride_size, padding=0, bias=False)

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size_multiplier * lstm_input_size,
                            num_layers=1, batch_first=False, device=device, dtype=dtype)
        output_sizes.append(down_layer_size_before_lstm)

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
            print(f"Down layer {i} x size: {x.size()}")
            outputs.append(x)

        print(f"x size before lstm preprocess: {x.size()}")
        lstm_input_preprocessed = self.lstm_conv(x)
        print(f"lstm_input_preprocessed size after conv: {lstm_input_preprocessed.size()}")
        lstm_input_preprocessed = lstm_input_preprocessed.view(-1, 1, lstm_input_preprocessed.size()[-1])
        # lstm_input_preprocessed = lstm_input_preprocessed.view(1, -1, lstm_input_preprocessed.shape[-1])
        print(f"lstm_input_preprocessed size after view: {lstm_input_preprocessed.size()}")
        lstm_output = self.lstm(lstm_input_preprocessed)[0]
        print(f"lstm_output size: {lstm_output.size()}")
        lstm_output = lstm_output.view(x.size()[0], x.size()[1], -1)
        print(f"lstm_output size after view: {lstm_output.size()}")
        outputs.append(lstm_output)

        print(f"output sizes after lstm: {[d.size() for d in outputs]}")
        
        for (i, up_layer) in enumerate(self.up_layers):
            # print(f"outputs sizes: {[d.size() for d in outputs]}")
            # print(f"up layer {i} x size: {x.size()}")
            processed_inputs = [
                nn.functional.interpolate(output, size=x.size()[2], mode='nearest')
                for output in outputs]

            print(f"up layer {i} processed_inputs sizes: {[d.size() for d in processed_inputs]}")
            
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

