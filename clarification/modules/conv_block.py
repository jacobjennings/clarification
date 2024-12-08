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

    def __init__(self, in_channels, out_channels, device, dtype, num_blocks=2):
        super(ConvBlock1D, self).__init__()

        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, device=device, dtype=dtype),
            nn.BatchNorm1d(num_features=out_channels, device=device, dtype=dtype),
            nn.ReLU(inplace=True),
        )

        for i in range(num_blocks - 1):
            self.sequential.add_module(f"conv_{i+2}", nn.Conv1d(in_channels=out_channels,
                                       out_channels=out_channels, kernel_size=3, stride=1, padding=1, device=device, dtype=dtype))
            self.sequential.add_module(
                f"batchnorm_{i+2}", nn.BatchNorm1d(num_features=out_channels, device=device, dtype=dtype))
            self.sequential.add_module(f"relu_{i+2}", nn.ReLU(inplace=True))

    def forward(self, x):
        # pylint: disable=missing-function-docstring
        return self.sequential(x)
