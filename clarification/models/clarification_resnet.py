"""1D ResNet for audio processing.

ResNet architecture using bottleneck residual blocks. Each residual block
compresses channels, applies convolution, then expands back, with identity
skip connections for gradient flow.
"""

import logging
from typing import Tuple, Optional, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from ..modules import ResBlock1D


def init_resnet_weights(module: nn.Module) -> None:
    """
    Initialize ResNet with Kaiming init and zero-init for residual BatchNorms.
    
    Zero-initializing the last BatchNorm in each residual block makes the block
    act as identity at initialization, improving gradient flow for deep networks.
    """
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        # Standard init for now - could zero-init last BN per block for deeper networks
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class ClarificationResNet(nn.Module):
    """
    1D ResNet architecture for audio processing.
    
    Uses bottleneck residual blocks with identity skip connections.
    The input is concatenated with the final features before output
    to provide a residual learning path.
    
    Args:
        name: Name for the model (used in layer naming)
        channel_size: Number of channels in residual blocks
        layer_count: Number of residual blocks
        device: Device to create tensors on
        dtype: Data type for weights
    """
    
    def __init__(
        self, 
        name: str, 
        channel_size: int, 
        layer_count: int, 
        device=None, 
        dtype=torch.float32
    ):
        super(ClarificationResNet, self).__init__()
        
        self.name = name
        self.channel_size = channel_size
        self.layer_count = layer_count

        if device is None:
            device = torch.get_default_device()

        # Input projection: 1 -> channel_size
        self.first_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=1, 
                out_channels=channel_size, 
                stride=1, 
                kernel_size=1, 
                device=device, 
                dtype=dtype
            ),
            nn.BatchNorm1d(num_features=channel_size, device=device, dtype=dtype),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(layer_count):
            self.resblocks.append(
                ResBlock1D(
                    name=f"{name}_resblock_{i}",
                    channel_size=channel_size,
                    device=device,
                    dtype=dtype
                )
            )

        # Output projection: channel_size + 1 (concatenated input) -> 1
        self.last_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=channel_size + 1, 
                out_channels=channel_size + 1, 
                stride=1, 
                kernel_size=1, 
                device=device, 
                dtype=dtype
            ),
            nn.BatchNorm1d(num_features=channel_size + 1, device=device, dtype=dtype),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=channel_size + 1, 
                out_channels=1, 
                stride=1, 
                kernel_size=1, 
                device=device,
                dtype=dtype
            ),
        )
        
        # Apply ResNet-specific initialization
        self.apply(init_resnet_weights)

    def forward(self, initial_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNet.
        
        Args:
            initial_x: Input tensor of shape [batch, 1, seq_len]
            
        Returns:
            Output tensor of shape [batch, 1, seq_len]
        """
        # Input projection
        x = self.first_layer(initial_x)
        
        # Residual blocks
        for resblock in self.resblocks:
            x = resblock(x)
        
        # Concatenate with original input for residual learning path
        x = torch.cat([x, initial_x], dim=1)
        
        # Output projection
        x = self.last_layer(x)
        
        return x

    def checkpoint_count(self) -> int:
        """Number of checkpoints for gradient checkpointing."""
        return len(self.resblocks) + 2

    def compute_checkpoint(
        self, 
        x: torch.Tensor, 
        initial_x: torch.Tensor, 
        outputs: Optional[List], 
        checkpoint_index: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List]]:
        """
        Compute a single checkpoint segment for gradient checkpointing.
        
        This allows trading compute for memory by recomputing intermediate
        activations during the backward pass.
        """
        if checkpoint_index == 0:
            x = self.first_layer(x)
            return x, initial_x, None
        if checkpoint_index == self.checkpoint_count() - 1:
            x = torch.cat([x, initial_x], dim=1)
            x = self.last_layer(x)
            return x, None, None
        x = self.resblocks[checkpoint_index - 1](x)
        return x, initial_x, None
