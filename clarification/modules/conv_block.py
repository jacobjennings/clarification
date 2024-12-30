"""1d Conv+BN+ReLU block."""

from torch import nn


class ConvBlock1D(nn.Module):
    """1D Conv+BN+ReLU block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        device: Device to run on.
        dtype: Data type to use.
        num_blocks: Number of blocks to stack.
    """

    def __init__(self, name, in_channels, out_channels, device, dtype, num_blocks=2, last_layer=False):
        super(ConvBlock1D, self).__init__()

        # print(f"ConvBlock1D {name} in_channels: {in_channels} out_channels: {out_channels}")

        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, device=device, dtype=dtype),
            nn.BatchNorm1d(num_features=out_channels, device=device, dtype=dtype),
            nn.ReLU(inplace=True),
        )

        for i in range(num_blocks - 1):
            last = i == num_blocks - 2 and last_layer
            self.sequential.add_module(f"conv_{i+2}", nn.Conv1d(
                in_channels=out_channels,
                out_channels=1 if last else out_channels,
                kernel_size=3, stride=1, padding=1, device=device, dtype=dtype))

            if not last:
                self.sequential.add_module(
                    f"batchnorm_{i+2}", nn.BatchNorm1d(num_features=out_channels, device=device, dtype=dtype))
                self.sequential.add_module(f"relu_{i+2}", nn.ReLU(inplace=True))

    def forward(self, x):
        # pylint: disable=missing-function-docstring
        x = self.sequential(x)
        # print(f"ConvBlock1D x size: {x.size()}")
        return x


class ConvBlock2D(nn.Module):
    """2D Conv+BN+ReLU block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        device: Device to run on.
        dtype: Data type to use.
        num_blocks: Number of blocks to stack.
    """

    def __init__(self, in_channels, out_channels, device, dtype, num_blocks=2):
        super(ConvBlock2D, self).__init__()

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, device=device, dtype=dtype),
            nn.BatchNorm2d(num_features=out_channels, device=device, dtype=dtype),
            nn.ReLU(inplace=True),
        )

        for i in range(num_blocks - 1):
            self.sequential.add_module(f"conv_{i+2}", nn.Conv2d(in_channels=out_channels,
                                       out_channels=out_channels, kernel_size=3, stride=1, padding=1, device=device, dtype=dtype))
            self.sequential.add_module(
                f"batchnorm_{i+2}", nn.BatchNorm2d(num_features=out_channels, device=device, dtype=dtype))
            self.sequential.add_module(f"relu_{i+2}", nn.ReLU(inplace=True))

    def forward(self, x):
        # pylint: disable=missing-function-docstring
        x = self.sequential(x)
        # print(f"ConvBlock2D x size: {x.size()}")
        return x
