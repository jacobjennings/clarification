"""1D u-net."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as nnF

from ..modules import OutLayer, Down, ConvBlock1D, UpNoCat, Glue1D
# from ..models import ClarificationDense

class ClarificationDenseLSTM(nn.Module):
    def __init__(self, name, in_channels, samples_per_batch, device, dtype, layer_sizes=None, invert=False, num_output_convblocks=2):
        super(ClarificationDenseLSTM, self).__init__()
        # lstm_input_size = samples_per_batch
        lstm_channel_size = 20
        hidden_size_multiplier = 4

        if len(layer_sizes) % 2 == 0:
            raise ValueError("The number of layers must be odd.")

        layer_sizes_len = len(layer_sizes)
        self.invert = invert

        self.first_layer = ConvBlock1D(name=f"{name}_firstlayer_conv", in_channels=in_channels, out_channels=layer_sizes[0], device=device, dtype=dtype)
        # print(f"First layer: in_channels: {in_channels} out_channels: {layer_sizes[0]}")
        
        output_sizes = [layer_sizes[0]]
        self.down_layers = nn.ModuleList()
        time_dim = samples_per_batch
        for i in range(len(layer_sizes) // 2):
            input_size = sum(output_sizes)
            out_channels = layer_sizes[i + 1]
            output_sizes.append(out_channels)
            down = Down(name=f"{name}_down_{i}", in_channels=input_size, out_channels=out_channels, device=device, dtype=dtype)
            self.down_layers.add_module("down_" + str(i), down)
            time_dim = math.ceil(time_dim / 2)
        
        # self.pre_lstm_reduction = Glue1D(in_channels=layer_sizes[len(layer_sizes) // 2 + 1], out_channels=max_layer_size,
        #                                  in_sample_size=samples_per_batch // 2, out_sample_size=input_size,
        #                                  device=device, dtype=dtype)
        
        # lstm_input_channel_size = layer_sizes[len(layer_sizes) // 2] // 8
        
        print(f"times_dim: {time_dim}")
        self.lstm_conv = nn.Conv1d(in_channels=layer_sizes[len(layer_sizes) // 2], 
                                   out_channels=lstm_channel_size, kernel_size=4, stride=16, padding=0, bias=False)
        lstm_input_size = math.ceil(time_dim / 16)
        print(f"lstm_input_size: {lstm_input_size}")
        
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size_multiplier * lstm_input_size,
                            num_layers=1, batch_first=True, device=device, dtype=dtype)
        output_sizes.append(lstm_channel_size)

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

        print(f"Out layer: in_channels: {layer_sizes[-1]} out_channels: 1")
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

        # print(f"x size before lstm preprocess: {x.size()}")        
        lstm_input_preprocessed = self.lstm_conv(x)
        # print(f"lstm_input_preprocessed size: {lstm_input_preprocessed.size()}")
        lstm_output = self.lstm(lstm_input_preprocessed)[0]
        outputs.append(lstm_output)
        
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

