"""Classifier for distortion detection."""

import torch.nn as nn

from ..modules import ConvBlock1D

class DistortionDetector(nn.Module):
    """Classifier for distortion detection.

    Args:
        convblock_sizes: List of sizes for each convolutional block.
        device: Device to run on.
        dtype: Data type to use.
    """

    def calculate_output_size(self, input_size, convblock_sizes):
        size = input_size
        for idx, size in enumerate(convblock_sizes):
            size = (size - 3) // 2 + 1  # Assuming kernel_size=3, stride=2 for MaxPool1d
        return size

    def __init__(self, convblock_sizes, samples_per_batch, batches_per_iteration, device, dtype):
        super(DistortionDetector, self).__init__()

        self.sequential = nn.Sequential()

        for idx, size in enumerate(convblock_sizes):
            self.sequential.add_module(f"conv_block_{idx}", ConvBlock1D(
                in_channels=1 if idx == 0 else convblock_sizes[idx - 1], out_channels=size, device=device, dtype=dtype))
            self.sequential.add_module(f"max_pool_{idx}", nn.MaxPool1d(kernel_size=3, stride=2))

        output_size = self.calculate_output_size(samples_per_batch * batches_per_iteration, convblock_sizes)
        print(f"Output size: {output_size} output_size * convblock_sizes[-1]: {output_size * convblock_sizes[-1]}")
        self.sequential.add_module("flatten", nn.Flatten())
        self.sequential.add_module("linear", nn.Linear(in_features=11225, out_features=1, device=device, dtype=dtype))


    def forward(self, x):
        x = self.sequential(x)
        x = x.squeeze(1)
        return x
