from torch import nn


class ResBlock1D(nn.Module):
    """1D Conv+BN+ReLU block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        device: Device to run on.
        dtype: Data type to use.
        num_blocks: Number of blocks to stack.
    """

    def __init__(self, name, channel_size, device, dtype, last_layer=False):
        super(ResBlock1D, self).__init__()

        # print(f"ResBlock1D {name} in_channels: {in_channels} out_channels: {out_channels}")

        self.conv1 = nn.Conv1d(in_channels=channel_size,
                      out_channels=channel_size // 2,
                      kernel_size=1, stride=1, padding=0, device=device, dtype=dtype)
        self.bn2 = nn.BatchNorm1d(num_features=channel_size // 2, device=device, dtype=dtype)
        self.rl3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv1d(
                in_channels=channel_size // 2,
                out_channels=1 if last_layer else channel_size // 2,
                kernel_size=3, stride=1, padding=1, device=device, dtype=dtype)
        self.bn5 = nn.BatchNorm1d(num_features=channel_size // 2, device=device, dtype=dtype)
        self.rl6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv1d(
                in_channels=channel_size // 2,
                out_channels=channel_size,
                kernel_size=1, stride=1, padding=0, device=device, dtype=dtype)

        self.last_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        initial_x = x
        # print(f"ResBlock1D initial x size: {x.size()}")
        # pylint: disable=missing-function-docstring
        x = self.conv1(x)
        # print(f"ResBlock1D 1 x size: {x.size()}")
        x = self.bn2(x)
        # print(f"ResBlock1D 2 x size: {x.size()}")
        x = self.rl3(x)
        # print(f"ResBlock1D 3 x size: {x.size()}")
        x = self.conv4(x)
        # print(f"ResBlock1D 4 x size: {x.size()}")
        x = self.bn5(x)
        # print(f"ResBlock1D 5 x size: {x.size()}")
        x = self.rl6(x)
        # print(f"ResBlock1D 6 x size: {x.size()}")
        x = self.conv7(x)
        # print(f"ResBlock1D 7 x size: {x.size()}")
        x += initial_x
        # print(f"ResBlock1D identity x size: {x.size()}")
        x = self.last_relu(x)
        # print(f"ConvBlock1D x size: {x.size()}")
        return x
