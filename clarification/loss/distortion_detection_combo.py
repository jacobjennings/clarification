"""Classifier for distortion detection."""
import torch
import torch.nn as nn
import torchaudio

from ..loss import *

class DistortionDetectorCombo(nn.Module):
    """Classifier for distortion detection.

    Args:
        convblock_sizes: List of sizes for each convolutional block.
        device: Device to run on.
        dtype: Data type to use.
    """
    def __init__(self, convblock_sizes, samples_per_batch, batches_per_iteration, device, dtype):
        super(DistortionDetectorCombo, self).__init__()
        self.freq = DistortionDetectorSpec(convblock_sizes, samples_per_batch, batches_per_iteration, device, dtype)
        self.time = DistortionDetector(convblock_sizes, samples_per_batch, batches_per_iteration, device, dtype)



    def forward(self, x):
        freq = self.freq(x)
        time = self.time(x)
        x = torch.stack((freq, time), 1)
        x = torch.mean(x, 1)
        return x
