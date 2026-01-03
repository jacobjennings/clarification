import logging

logger = logging.getLogger(__name__)
from torch import nn


class ResBlock1D(nn.Module):
    """Bottleneck residual block for 1D audio.
    
    Uses the standard ResNet bottleneck pattern:
    - 1x1 conv to reduce channels (bottleneck)
    - 3x3 conv for spatial processing
    - 1x1 conv to expand channels back
    - Residual addition (no activation after add)
    
    Pattern: Conv -> BN -> ReLU for each layer, with identity shortcut.
    The final ReLU comes AFTER the residual addition only if needed for
    subsequent processing, but this degrades gradient flow. We follow
    the improved pre-activation pattern where possible.

    Args:
        name: Name for the block (for debugging).
        channel_size: Number of input/output channels.
        device: Device to run on.
        dtype: Data type to use.
    """

    def __init__(self, name, channel_size, device, dtype):
        super(ResBlock1D, self).__init__()
        
        bottleneck_channels = channel_size // 2
        
        # Bottleneck: reduce channels
        self.conv1 = nn.Conv1d(
            in_channels=channel_size,
            out_channels=bottleneck_channels,
            kernel_size=1, stride=1, padding=0, 
            device=device, dtype=dtype)
        self.bn1 = nn.BatchNorm1d(
            num_features=bottleneck_channels, 
            device=device, dtype=dtype)
        
        # Spatial processing at reduced channels
        self.conv2 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
            kernel_size=3, stride=1, padding=1, 
            device=device, dtype=dtype)
        self.bn2 = nn.BatchNorm1d(
            num_features=bottleneck_channels, 
            device=device, dtype=dtype)
        
        # Expand back to original channels
        self.conv3 = nn.Conv1d(
            in_channels=bottleneck_channels,
            out_channels=channel_size,
            kernel_size=1, stride=1, padding=0, 
            device=device, dtype=dtype)
        self.bn3 = nn.BatchNorm1d(
            num_features=channel_size, 
            device=device, dtype=dtype)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        
        # Bottleneck path: conv -> bn -> relu for each layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Residual connection - add identity BEFORE final activation
        # This allows gradients to flow directly through the identity path
        out = out + identity
        
        # Final ReLU after residual addition
        # Note: Some papers suggest removing this for better gradient flow,
        # but we keep it for consistency with original ResNet
        out = self.relu(out)
        
        return out
