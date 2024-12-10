"""1D u-net with dense LSTM connections between blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as nnF

import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

from torch.utils.tensorboard import SummaryWriter

from ..modules.conv_block import ConvBlock1D
from ..modules.out_layer import OutLayer
from ..modules.glue import Glue1D

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, device, dtype):
        super(Down, self).__init__()

        self.sequential = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            ConvBlock1D(in_channels=in_channels, out_channels=out_channels, device=device, dtype=dtype)
        )

    def forward(self, x):
        return self.sequential(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, layer_sizes, layer_num, lstm_out_sample_size, samples_per_batch, device, dtype):
        super(Up, self).__init__()

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

        print(f"out_channels: {out_channels}")

        self.transpose = nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels //
                                            2, kernel_size=2, stride=2, device=device, dtype=dtype)
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


class DenseLSTM(nn.Module):
    def __init__(self, layer_sizes, input_size, hidden_size_multiplier, samples_per_batch, device, dtype):
        super(DenseLSTM, self).__init__()

        self.output_sample_size = input_size * hidden_size_multiplier

        down_stacked_size = 0
        max_layer_size = 0
        for layer_size in layer_sizes[1:]:
            down_stacked_size += layer_size
            max_layer_size = max(max_layer_size, layer_size)

        print(f"down_stacked_size: {down_stacked_size}")

        # self.pre_lstm_reduction = nn.Conv1d(in_channels=down_stacked_size, out_channels=max_layer_size, kernel_size=2, stride=2, device=device, dtype=dtype)
        self.pre_lstm_reduction = Glue1D(in_channels=down_stacked_size, out_channels=max_layer_size,
                                         in_sample_size=samples_per_batch // 2, out_sample_size=input_size, device=device, dtype=dtype)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=input_size * hidden_size_multiplier,
                            num_layers=1, batch_first=True, device=device, dtype=dtype)

    def forward(self, down_outputs):
        target_time_dim = down_outputs[0].size()[2]

        upsampled_down_outputs = [
            nnF.interpolate(down_output, size=target_time_dim, mode='linear', align_corners=False)
            for down_output in down_outputs
        ]

        x = torch.cat(upsampled_down_outputs, dim=1)
        x = self.pre_lstm_reduction(x)
        x = self.lstm(x)[0]
        return x


class ClarificationDenseLSTM(nn.Module):
    def __init__(self, in_channels, samples_per_batch, device, dtype):
        super(ClarificationDenseLSTM, self).__init__()

        layer_sizes = [64, 128, 256, 512, 1024]

        self.first_layer = ConvBlock1D(in_channels=in_channels, out_channels=layer_sizes[0], device=device, dtype=dtype)

        self.down_layers = [
            Down(in_channels=layer_sizes[i], out_channels=layer_sizes[i+1], device=device, dtype=dtype)
            for i in range(len(layer_sizes) - 1)
        ]
        self.down_layers_module_list = nn.ModuleList(self.down_layers)

        self.dense_lstm = DenseLSTM(
            layer_sizes=layer_sizes, 
            input_size=400,
            hidden_size_multiplier=3, 
            samples_per_batch=samples_per_batch,
            device=device, 
            dtype=dtype
        )

        self.up_layers = [
            Up(
                in_channels=layer_sizes[-(i+1)],
                out_channels=layer_sizes[-(i+2)],
                layer_sizes=layer_sizes,
                layer_num=i,
                lstm_out_sample_size=self.dense_lstm.output_sample_size,
                samples_per_batch=samples_per_batch,
                device=device,
                dtype=dtype
            )
            for i in range(len(layer_sizes) - 2)
        ]
        self.up_layers_module_list = nn.ModuleList(self.up_layers)

        self.last_layer = OutLayer(
            in_channels=layer_sizes[1], out_channels=in_channels, num_convblocks=4, device=device, dtype=dtype)

    def forward(self, x):
        x = self.first_layer(x)
        down_outputs = []
        for down_layer in self.down_layers:
            x = down_layer(x)
            down_outputs.append(x)

        down_outputs_reversed = list(reversed(down_outputs))
        lstm_out = self.dense_lstm(down_outputs)

        for (i, up_layer) in enumerate(self.up_layers):
            x = up_layer(x, down_outputs_reversed[i + 1], lstm_out)

        x = self.last_layer(x)
        x = x.squeeze(0).squeeze(0)

        return x
