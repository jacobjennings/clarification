"""1D u-net with dense LSTM connections between blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as nnF

from ..modules import ConvBlock1D, OutLayer, Glue1D, Down, UpWithLSTMInput, Up


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

        self.pre_lstm_reduction = Glue1D(in_channels=down_stacked_size, out_channels=max_layer_size,
                                         in_sample_size=samples_per_batch // 2, out_sample_size=input_size,
                                         device=device, dtype=dtype)
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


class HorizontalLSTM(nn.Module):
    def __init__(self, layer_sizes, input_size, hidden_size_multiplier, samples_per_batch, device, dtype):
        super(HorizontalLSTM, self).__init__()

        self.output_sample_size = input_size * hidden_size_multiplier

        down_stacked_size = 0
        max_layer_size = 0
        for layer_size in layer_sizes[1:]:
            down_stacked_size += layer_size
            max_layer_size = max(max_layer_size, layer_size)

        print(f"down_stacked_size: {down_stacked_size}")

        self.pre_lstm_reduction = Glue1D(in_channels=down_stacked_size, out_channels=max_layer_size,
                                         in_sample_size=samples_per_batch // 2, out_sample_size=input_size,
                                         device=device, dtype=dtype)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=input_size * hidden_size_multiplier,
                            num_layers=1, batch_first=True, device=device, dtype=dtype)

    def forward(self, down_output):
        target_time_dim = down_output.size()[2]
        print(f"HorizontalLSTM target_time_dim: {target_time_dim} down_output size: {down_output.size()}")
        upsampled_down_output = nnF.interpolate(down_output, size=target_time_dim, mode='linear', align_corners=False)
        print(f"HorizontalLSTM upsampled_down_output size: {upsampled_down_output.size()}")
        x = torch.cat(upsampled_down_output, dim=1)
        print(f"HorizontalLSTM x cat size: {x.size()}")
        x = self.pre_lstm_reduction(x)
        print(f"HorizontalLSTM x pre_lstm_reduction size: {x.size()}")
        x = self.lstm(x)[0]
        print(f"HorizontalLSTM x lstm size: {x.size()}")
        return x


class ClarificationConfigurableLSTM(nn.Module):
    def __init__(self, in_channels, samples_per_batch, device, dtype, layer_sizes=None, horizontal_lstm_up_layers=None):
        super(ClarificationConfigurableLSTM, self).__init__()

        if layer_sizes is None:
            layer_sizes = [300, 700, 900]

        if horizontal_lstm_up_layers is None:
            horizontal_lstm_up_layers = [1]

        self.first_layer = ConvBlock1D(in_channels=in_channels, out_channels=layer_sizes[0], device=device, dtype=dtype)

        self.down_layers = [
            Down(in_channels=layer_sizes[i], out_channels=layer_sizes[i + 1], device=device, dtype=dtype)
            for i in range(len(layer_sizes) - 1)
        ]
        self.down_layers_module_list = nn.ModuleList(self.down_layers)

        for i in range(len(layer_sizes) - 2):
            layer_depth = len(layer_sizes) - 1 - i

            if i - 1 in horizontal_lstm_up_layers:
                lstm = HorizontalLSTM(layer_sizes=layer_sizes, input_size=123, hidden_size_multiplier=3,
                                      samples_per_batch=samples_per_batch, device=device, dtype=dtype)
                up = UpWithLSTMInput(in_channels=layer_sizes[-(i + 1)], out_channels=layer_sizes[-(i + 2)],
                                     lstm_out_sample_size=layer_sizes[-(i + 2)], layer_sizes=layer_sizes, layer_num=i,
                                     samples_per_batch=samples_per_batch, device=device, dtype=dtype)
            else:
                up = Up(in_channels=layer_sizes[-(i + 1)], out_channels=layer_sizes[-(i + 2)],
                        layer_depth=layer_depth, samples_per_batch=samples_per_batch, device=device, dtype=dtype)
            self.up_layers.append(up)

        self.dense_lstm = DenseLSTM(
            layer_sizes=layer_sizes,
            input_size=400,
            hidden_size_multiplier=3,
            samples_per_batch=samples_per_batch,
            device=device,
            dtype=dtype
        )

        self.up_layers = [
            UpWithLSTMInput(
                in_channels=layer_sizes[-(i + 1)],
                out_channels=layer_sizes[-(i + 2)],
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
