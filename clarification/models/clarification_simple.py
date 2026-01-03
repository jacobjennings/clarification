"""1D U-Net with skip connections.

Standard U-Net architecture for audio processing. Each layer in the encoder
path has a corresponding decoder layer with skip connections for preserving
fine-grained details.
"""
import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from ..modules import OutLayer, Down, ConvBlock1D, Up
from ..util import init_weights_kaiming


class ClarificationSimple(nn.Module):
    """
    Standard 1D U-Net architecture with skip connections.
    
    This is the traditional U-Net design where encoder features are concatenated
    with decoder features at matching resolutions. Requires at least 5 layers
    (2 down, 1 bottleneck, 2 up) for skip connections to be meaningful.
    
    For shallower architectures (3 layers), use ClarificationDense instead.
    
    Args:
        name: Name for the model (used in layer naming)
        device: Device to create tensors on
        dtype: Data type for weights
        layer_sizes: List of channel counts for each layer (must be odd, >= 5)
        invert: If True, output is input - prediction (residual learning)
        num_output_convblocks: Number of conv blocks in the output layer
    """
    
    def __init__(
        self, 
        name: str, 
        device=None, 
        dtype=torch.float32, 
        layer_sizes: List[int] = None, 
        invert: bool = False, 
        num_output_convblocks: int = 2
    ):
        super(ClarificationSimple, self).__init__()

        if device is None:
            device = torch.get_default_device()

        if len(layer_sizes) % 2 == 0:
            raise ValueError("The number of layers must be odd.")
        
        if len(layer_sizes) < 5:
            raise ValueError(
                f"ClarificationSimple requires at least 5 layers for skip connections to work. "
                f"Got {len(layer_sizes)} layers. With 3 layers, there are no skip connections "
                f"and the U-Net architecture provides no benefit. Use ClarificationDense for "
                f"shallow architectures, or increase layer_sizes to at least 5 elements."
            )

        self.invert = invert
        self.layer_sizes = layer_sizes
        
        if layer_sizes is None:
            layer_sizes = [64, 128, 256, 512, 1024]

        # First layer: input -> layer_sizes[0] channels
        self.first_layer = ConvBlock1D(
            name=f"{name}_firstlayer_conv", 
            in_channels=1, 
            out_channels=layer_sizes[0], 
            device=device, 
            dtype=dtype
        )

        # Encoder (down) path
        self.down_layers = nn.ModuleList()
        for i in range(len(layer_sizes) // 2):
            down = Down(
                name=f"{name}_down_{i}", 
                in_channels=layer_sizes[i], 
                out_channels=layer_sizes[i + 1], 
                device=device, 
                dtype=dtype
            )
            self.down_layers.append(down)

        # Decoder (up) path
        layer_sizes_len = len(layer_sizes)
        self.up_layers = nn.ModuleList()
        for i in range(layer_sizes_len // 2 - 1):
            in_channels = layer_sizes[layer_sizes_len // 2 + i]
            out_channels = layer_sizes[layer_sizes_len // 2 + i + 1]
            up = Up(
                name=f"{name}_up_{i}",
                in_channels=in_channels,
                out_channels=out_channels,
                device=device, 
                dtype=dtype,
                layer_num=i
            )
            self.up_layers.append(up)

        # Output layer
        self.last_layer = OutLayer(
            name=f"{name}_outlayer",
            in_channels=layer_sizes[-2],
            out_channels=layer_sizes[-1],
            num_convblocks=num_output_convblocks, 
            device=device, 
            dtype=dtype
        )
        
        # Apply Kaiming initialization for ReLU networks
        self.apply(init_weights_kaiming)

    def forward(self, initial_x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.
        
        Args:
            initial_x: Input tensor of shape [batch, 1, seq_len]
            
        Returns:
            Output tensor of shape [batch, 1, seq_len]
        """
        # Encoder path - save outputs for skip connections
        x = self.first_layer(initial_x)
        down_outputs = [x]
        
        for down_layer in self.down_layers:
            x = down_layer(x)
            down_outputs.append(x)

        # Decoder path - use skip connections from encoder
        down_outputs_reversed = list(reversed(down_outputs))

        for i, up_layer in enumerate(self.up_layers):
            x = up_layer(x, down_outputs_reversed[i + 1])

        # Output layer
        x = self.last_layer(x)
        x = x.squeeze(0).squeeze(0)
        
        # Optional residual learning
        if self.invert:
            x = initial_x - x

        return x

