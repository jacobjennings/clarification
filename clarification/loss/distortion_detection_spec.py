"""Classifier for distortion detection."""

import torch.nn as nn
import torchaudio

from ..modules import ConvBlock2D

class DistortionDetectorSpec(nn.Module):
    """Classifier for distortion detection.

    Args:
        convblock_sizes: List of sizes for each convolutional block.
        device: Device to run on.
        dtype: Data type to use.
    """
    def __init__(self, convblock_sizes, samples_per_batch, batches_per_iteration, device, dtype):
        super(DistortionDetectorSpec, self).__init__()

        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_mels=64)

        self.sequential = nn.Sequential()
        for idx, size in enumerate(convblock_sizes):
            self.sequential.add_module(f"conv_block_{idx}", ConvBlock2D(
                in_channels=1 if idx == 0 else convblock_sizes[idx - 1], out_channels=size, device=device, dtype=dtype))
            self.sequential.add_module(f"max_pool_{idx}", nn.MaxPool2d(kernel_size=3, stride=2))

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=75, out_features=1, device=device, dtype=dtype)


    def forward(self, x):
        # print(f"0 DistortionDetector x size: {x.size()}")
        x = self.spec(x)
        # print(f"0.5 DistortionDetector x size: {x.size()}")
        # x = x.squeeze(1)
        # print(f"1 DistortionDetector x size: {x.size()}")
        x = self.sequential(x)
        # print(f"2 DistortionDetector x size: {x.size()}")
        x = self.flatten(x)
        # print(f"3 DistortionDetector x size: {x.size()}")
        x = self.linear(x)
        # print(f"4 DistortionDetector x size: {x.size()}")
        x = x.squeeze(1)
        # print(f"5 DistortionDetector x size: {x.size()}")
        return x
