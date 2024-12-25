"""Classifier for distortion detection."""
import math

import torch
import torch.nn as nn

from ..modules import Down
from ..modules import ConvBlock1D

class DistortionDetectorDenseEncoder(nn.Module):
    """Classifier for distortion detection.

    Args:
        in_channels: Number of input channels.
        samples_per_batch: Number of samples per batch.
        device: Device to run on.
        dtype: Data type to use.
        layer_sizes: List of sizes for each convolutional block. Modified due to density.
        downsample_layers: Whether to downsample layers to match input size of next layer.
    """
    def __init__(self, in_channels, samples_per_batch, device, dtype, layer_sizes=None, downsample_layers=False):
        super(DistortionDetectorDenseEncoder, self).__init__()
        self.samples_per_batch = samples_per_batch
        self.downsample_layers = downsample_layers

        self.first_layer = ConvBlock1D(in_channels=in_channels, out_channels=layer_sizes[0], device=device, dtype=dtype)
        # print(f"First layer in_channels: {in_channels} out_channels: {layer_sizes[0]}")
        self.down_layers = nn.ModuleList()

        previous_output_sizes = [1]
        # for i in range(len(layer_sizes) - 1):
        #     input_size = layer_sizes[i] + previous_output_sizes

        for i in range(len(layer_sizes) - 1):
            extra_output_size_total = 0
            for eos_idx, extra_output_size in enumerate(previous_output_sizes):
                if downsample_layers:
                    extra_output_size_total += extra_output_size
                else:
                    extra_output_size_total += math.ceil(
                        extra_output_size * pow(2, (len(previous_output_sizes) - eos_idx - 1)))

            # print(f"extra_output_size_total: {extra_output_size_total}")
            input_size = layer_sizes[i] + extra_output_size_total
            out_channels = layer_sizes[i + 1] if i < len(layer_sizes) - 1 else 1

            previous_output_sizes.append(out_channels)
            # print(f"previous_output_sizes: {previous_output_sizes} extra_output_size_total:{extra_output_size_total}")
            down = Down(in_channels=input_size, out_channels=out_channels, device=device, dtype=dtype)
            # print(f"Down layer {i} in_channels: {input_size} out_channels: {out_channels}")
            self.down_layers.add_module("down_" + str(i), down)

        self.linear_size = samples_per_batch // 2
        self.linear = nn.Linear(in_features=self.linear_size, out_features=1, device=device, dtype=dtype)
        # print(f"Out layer: in_channels: {self.linear_size} out_channels: 1")
        print(self.down_layers.named_modules())


    def forward(self, x):
        down_extra_inputs = [x]
        x = self.first_layer(x)
        # print(f"0 distorion_detector x size: {x.size()}")
        for i, down_layer in enumerate(self.down_layers):
            # print(f"1 distorion_detector down {i} x size: {x.size()}")
            # print(f"1 down_extra_inputs: {[d.size() for d in down_extra_inputs]}")

            if self.downsample_layers:
                down_extra_inputs_processed = [
                    nn.functional.interpolate(down_extra_input, size=x.size()[-1], mode='linear', align_corners=False)
                    for down_extra_input in down_extra_inputs]
            else:
                down_extra_inputs_processed = [
                    down_extra_input.view(x.size()[0], -1, x.size()[2])
                    for down_extra_input in down_extra_inputs]

            # print(f"1.1 distorion_detector down {i} down_extra_inputs_views: {[d.size() for d in down_extra_inputs_views]}")
            x = torch.cat(tuple(down_extra_inputs_processed) + (x,), dim=1)
            # print(f"1.2 distorion_detector down {i} x size: {x.size()}")
            x = down_layer(x)
            # print(f"1.3 distorion_detector down {i} x size: {x.size()}")
            down_extra_inputs.append(x)

        # print(f"1 DistortionDetector x size: {x.size()}")
        x = x.view(x.size(0), -1, self.linear_size)

        # print(f"1.5 distorion_detector x size: {x.size()}")
        x = self.linear(x)
        # print(f"2 DistortionDetector x size: {x.size()}")
        x = x.squeeze(2)
        # print(f"2 DistortionDetector x size: {x.size()}")
        x = torch.mean(x, 1)
        # print(f"4 DistortionDetector x size: {x.size()}")
        return x
