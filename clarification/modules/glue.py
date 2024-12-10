"""Dynamic glue layer for audio.

Leverages a single Conv1d layer (optionally combined with an Upsampler) to auto-match
input and output tensor shapes. Be wary of data loss with large differences in total size.
"""

from torch import nn
import torch.nn.functional as nnF


class Glue1D(nn.Module):
    """Dynamic glue layer for audio.
    
    Leverages a single Conv1d layer (optionally combined with an Upsampler) to auto-match
    input and output tensor shapes. Be wary of data loss with large differences in total size.

    """

    def __init__(self, in_channels, out_channels, in_sample_size, out_sample_size, device, dtype):
        """Initialize Glue layer.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            in_sample_size: Number of expected time-domain elements in input tensor.
            out_sample_size: Number of expected time-domain elements in output tensor.
            device: Device to run on.
            dtype: Data type to use.
        """
        super(Glue1D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_sample_size = in_sample_size
        self.out_sample_size = out_sample_size
        self.is_downsample = in_sample_size > out_sample_size
        if in_channels == out_channels and in_sample_size == out_sample_size:
            self.is_noop = True
            return

        self.is_noop = False

        self.sequential = nn.Sequential()

        kernel_size = 1
        if self.is_downsample:
            for i in range(10, 1, -1):
                if in_sample_size // i > out_sample_size:
                    kernel_size = i
                    break

        self.sequential.add_module(
            "convReduce", nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=2, device=device, dtype=dtype),
        )

        if not self.is_downsample:
            self.multiplier_factor = 10
            for i in range(10, 1, -1):
                if in_sample_size * i < out_sample_size:
                    self.multiplier_factor = i
                    break
            self.upsample = nn.Upsample(
                scale_factor=self.multiplier_factor,
                mode='linear',
                align_corners=True
            )


    def forward(self, x):
        # pylint: disable=missing-function-docstring
        if self.is_noop:
            return x

        x = self.sequential(x)

        if not self.is_downsample and self.multiplier_factor != 1:
            x = self.upsample(x)

        x = nnF.interpolate(x, size=self.out_sample_size, mode='linear', align_corners=False)

        return x
