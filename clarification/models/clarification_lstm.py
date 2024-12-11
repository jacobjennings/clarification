"""1D u-net with dense LSTM connections between blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as nnF

from ..modules import OutLayer, Down, ConvBlock1D, Up


class ClarificationLSTM(nn.Module):
    def __init__(self, in_channels, samples_per_batch, device, dtype, layer_sizes=None):
        super(ClarificationLSTM, self).__init__()

        if layer_sizes is None:
            layer_sizes = [64, 128, 256, 512, 1024]

        self.first_layer = ConvBlock1D(in_channels=in_channels, out_channels=layer_sizes[0], device=device, dtype=dtype)

        self.down_layers = [
            Down(in_channels=layer_sizes[i], out_channels=layer_sizes[i + 1], device=device, dtype=dtype)
            for i in range(len(layer_sizes) - 1)
        ]
        self.down_layers_module_list = nn.ModuleList(self.down_layers)

        self.upWithLSTM = UpWithLSTM(
            in_channels=layer_sizes[-1],
            out_channels=layer_sizes[-2],
            layer_sizes=layer_sizes,
            samples_per_batch=samples_per_batch,
            device=device, dtype=dtype)

        self.remaining_up_layers = [
            Up(in_channels=layer_sizes[-(i + 2)], out_channels=layer_sizes[-(i + 3)], device=device, dtype=dtype)
            for i in range(len(layer_sizes) - 3)
        ]
        self.up_layers_module_list = nn.ModuleList(self.remaining_up_layers)

        self.last_layer = OutLayer(in_channels=layer_sizes[1], out_channels=in_channels, device=device, dtype=dtype)

    def forward(self, x):
        x = self.first_layer(x)
        down_outputs = []
        for down_layer in self.down_layers:
            x = down_layer(x)
            down_outputs.append(x)

        down_outputs_reversed = list(reversed(down_outputs))

        for (i, up_layer) in enumerate([self.upWithLSTM] + self.up_layers):
            x = up_layer(x, down_outputs_reversed[i + 1])

        x = self.last_layer(x)
        x = x.squeeze(0).squeeze(0)

        return x



class UpWithLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, layer_sizes, samples_per_batch, device, dtype):
        super(UpWithLSTM, self).__init__()

        lstm_input_size = samples_per_batch
        for i in range(len(layer_sizes) - 1):
            lstm_input_size = lstm_input_size // 2

        self.pre_lstm_reduction = nn.Conv1d(in_channels=in_channels, out_channels=lstm_input_size, kernel_size=2,
                                            stride=2, device=device, dtype=dtype)
        self.lstm = nn.LSTM(input_size=lstm_input_size // 2, hidden_size=lstm_input_size * 4, num_layers=1,
                            batch_first=True, device=device, dtype=dtype)
        # self.lstm_channel_reduction = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, device=device, dtype=dtype)
        self.lstm_downsample = nn.Conv1d(in_channels=lstm_input_size, out_channels=lstm_input_size, kernel_size=2,
                                         stride=2, device=device, dtype=dtype)
        self.lstm_expansion = nn.Conv1d(in_channels=lstm_input_size, out_channels=out_channels, kernel_size=1, stride=1,
                                        device=device, dtype=dtype)

        self.transpose = nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2,
                                            stride=2, device=device, dtype=dtype)
        self.convBlock = ConvBlock1D(in_channels=in_channels, out_channels=out_channels, device=device, dtype=dtype)

    def forward(self, x1, x2):
        # print(f"UPPPPPP x1 size: {x1.size()} x2 size: {x2.size()} x3 size: {x3.size()}")

        # print(f"x1 before LSTM: {x1.size()}")
        lstm_out = self.pre_lstm_reduction(x1)
        # print(f"Pre-LSTM output: {lstm_out.size()}")
        lstm_out = self.lstm(lstm_out)[0]
        # print(f"LSTM output: {lstm_out.size()}")
        # lstm_out = self.lstm_channel_reduction(lstm_out)
        # print(f"LSTM lstm_channel_reduction output: {lstm_out.size()}")
        lstm_out = self.lstm_downsample(lstm_out)
        # print(f"LSTM lstm_downsample output: {lstm_out.size()}")
        lstm_out = self.lstm_expansion(lstm_out)
        # print(f"LSTM lstm_expansion output: {lstm_out.size()}")

        x1 = self.transpose(x1)

        diff = x2.size()[1] - x1.size()[1]  # Calculate difference correctly
        # print(f"Transposed x1: {x1.size()} x2: {x2.size()} diff: {diff}")

        # Pad x1 if necessary
        x1 = nnF.pad(x1, (diff // 2, diff - diff // 2))
        # print(f"Padded x1: {x1.size()}")

        # print(f"x1: {x1}, lstm: {lstm_out}")
        x1 = x1 + lstm_out

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        # print(f"Concatenated x: {x.size()}")

        x = self.convBlock(x)

        # print(f"ConvBlock output: \n{x.size()}")

        return x
