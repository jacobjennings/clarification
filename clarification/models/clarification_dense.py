"""1D u-net with dense connections.

This module implements a U-Net architecture with DenseNet-style connections,
where each layer has access to all previous layer outputs (resized to match
the current resolution).
"""

import torch
import torch.nn as nn
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)
from ..modules import OutLayer, Down, ConvBlock1D, UpNoCat
from ..util import init_weights_kaiming


def input_size_for_layer(layer_num: int, layer_sizes: List[int]) -> int:
    """
    Compute the total input channels for a layer in the dense network.
    
    In a dense network, each layer receives all previous layer outputs,
    resized to match the current resolution. When viewing a higher-resolution
    feature map at a lower resolution, channels are effectively multiplied.
    
    Args:
        layer_num: Index of the target layer (0-based)
        layer_sizes: List of channel counts for each layer
        
    Returns:
        Total number of input channels for the layer
    """
    layer_depths = []
    depth = 0
    for i in range(len(layer_sizes)):
        layer_depths.append(depth)
        if i < len(layer_sizes) // 2:
            depth += 1
        else:
            depth -= 1

    layer_size = 0
    layer_num_depth = layer_depths[layer_num]

    for i in range(layer_num + 1):
        relative_depth = layer_num_depth - layer_depths[i]
        adding_size = int(layer_sizes[i] * 2**relative_depth)
        layer_size += adding_size

    return layer_size


def compute_dense_metadata(layer_sizes: List[int]) -> Tuple[List[int], List[int], List[List[Tuple[int, int]]]]:
    """
    Pre-compute metadata for efficient dense connections.
    
    Returns:
        total_channels: Total accumulated channels at each layer
        depths: Depth (resolution level) at each layer
        channel_offsets: For each layer, list of (offset, size) for each previous layer's contribution
    """
    num_layers = len(layer_sizes)
    
    # Compute depth at each layer (0 = full resolution, increases going down)
    depths = []
    depth = 0
    for i in range(num_layers):
        depths.append(depth)
        if i < num_layers // 2:
            depth += 1
        else:
            depth -= 1
    
    # Compute total channels and offsets at each layer
    total_channels = []
    channel_offsets = []
    
    for layer_idx in range(num_layers):
        layer_depth = depths[layer_idx]
        offsets = []
        current_offset = 0
        
        for prev_idx in range(layer_idx + 1):
            prev_depth = depths[prev_idx]
            # When viewing from prev_depth to layer_depth, channels multiply by 2^(depth_diff)
            depth_diff = layer_depth - prev_depth
            effective_channels = layer_sizes[prev_idx] * (2 ** depth_diff)
            offsets.append((current_offset, effective_channels))
            current_offset += effective_channels
        
        total_channels.append(current_offset)
        channel_offsets.append(offsets)
    
    return total_channels, depths, channel_offsets


class DenseBuffer:
    """
    Pre-allocated buffer manager for efficient dense connections.
    
    Instead of creating new tensors with torch.cat on every forward pass,
    this class manages pre-allocated buffers and writes layer outputs
    into specific regions.
    """
    
    def __init__(self, layer_sizes: List[int], device, dtype):
        self.layer_sizes = layer_sizes
        self.device = device
        self.dtype = dtype
        self.num_layers = len(layer_sizes)
        
        # Pre-compute metadata
        self.total_channels, self.depths, self.channel_offsets = compute_dense_metadata(layer_sizes)
        
        # Buffers will be allocated on first forward pass (need to know batch size and seq_len)
        self._buffers: Optional[List[torch.Tensor]] = None
        self._cached_batch_size = None
        self._cached_seq_len = None
        
    def _ensure_buffers(self, batch_size: int, seq_len: int) -> None:
        """Allocate or resize buffers if needed."""
        if (self._buffers is not None and 
            self._cached_batch_size == batch_size and 
            self._cached_seq_len == seq_len):
            return
        
        # Allocate buffer for each layer
        self._buffers = []
        for layer_idx in range(self.num_layers):
            depth = self.depths[layer_idx]
            layer_seq_len = seq_len // (2 ** depth)
            total_ch = self.total_channels[layer_idx]
            
            buf = torch.zeros(
                batch_size, total_ch, layer_seq_len,
                device=self.device, dtype=self.dtype
            )
            self._buffers.append(buf)
        
        self._cached_batch_size = batch_size
        self._cached_seq_len = seq_len
    
    def write_output(self, layer_idx: int, output: torch.Tensor, outputs_list: List[torch.Tensor]) -> None:
        """
        Write a layer's output and propagate to all subsequent buffers.
        
        Args:
            layer_idx: Index of the layer that produced this output
            output: The layer's output tensor [batch, channels, seq_len]
            outputs_list: List to store original outputs for viewing
        """
        outputs_list.append(output)
        
        # Write to the current layer's buffer at the appropriate offset
        if layer_idx < len(self._buffers):
            offset, size = self.channel_offsets[layer_idx][layer_idx]
            self._buffers[layer_idx][:, offset:offset+size, :] = output
    
    def get_dense_input(self, layer_idx: int, outputs_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Get the concatenated dense input for a layer.
        
        This builds the dense input by viewing all previous outputs at the
        current layer's resolution and writing them into the pre-allocated buffer.
        
        Args:
            layer_idx: Index of the layer that needs input
            outputs_list: List of all previous layer outputs
            
        Returns:
            Dense input tensor [batch, total_channels, seq_len]
        """
        batch_size = outputs_list[0].size(0)
        seq_len = outputs_list[0].size(2)
        self._ensure_buffers(batch_size, seq_len)
        
        target_depth = self.depths[layer_idx]
        target_seq_len = seq_len // (2 ** target_depth)
        buf = self._buffers[layer_idx]
        
        # Write each previous output into the buffer at the correct offset
        for prev_idx, prev_output in enumerate(outputs_list):
            offset, effective_channels = self.channel_offsets[layer_idx][prev_idx]
            # View the previous output at the current resolution
            viewed = prev_output.view(batch_size, -1, target_seq_len)
            buf[:, offset:offset+effective_channels, :] = viewed
        
        return buf


class ClarificationDense(nn.Module):
    """
    1D U-Net with DenseNet-style connections.
    
    Each layer receives all previous layer outputs (viewed at the current resolution),
    providing rich feature reuse. This uses pre-allocated buffers to avoid memory
    fragmentation from repeated torch.cat operations.
    
    Args:
        name: Name for the model (used in layer naming)
        layer_sizes: List of channel counts for each layer (must be odd length)
        invert: If True, output is input - prediction (residual learning)
        num_output_convblocks: Number of conv blocks in the output layer
        device: Device to create tensors on
        dtype: Data type for weights
    """
    
    def __init__(
        self,
        name: str,
        layer_sizes: List[int] = None,
        invert: bool = False,
        num_output_convblocks: int = 2,
        device=None,
        dtype=torch.float32,
    ):
        super(ClarificationDense, self).__init__()
        self.layer_sizes = layer_sizes
        self.name = name
        
        if device is None:
            device = torch.get_default_device()

        if len(layer_sizes) % 2 == 0:
            raise ValueError("The number of layers must be odd.")

        layer_sizes_len = len(layer_sizes)
        self.invert = invert
        self.num_down = len(layer_sizes) // 2
        self.num_up = layer_sizes_len // 2 - 1

        # First layer: input -> layer_sizes[0] channels
        self.first_layer = ConvBlock1D(
            name=f"{name}_firstlayer_conv",
            in_channels=1,
            out_channels=layer_sizes[0],
            device=device,
            dtype=dtype,
        )

        # Down layers
        self.down_layers = nn.ModuleList()
        for i in range(self.num_down):
            in_channels = input_size_for_layer(i, layer_sizes)
            out_channels = layer_sizes[i + 1]
            down = Down(
                name=f"{name}_down_{i}",
                in_channels=in_channels,
                out_channels=out_channels,
                device=device,
                dtype=dtype,
            )
            self.down_layers.append(down)

        # Up layers
        self.up_layers = nn.ModuleList()
        for i in range(self.num_up):
            in_channels = input_size_for_layer(layer_sizes_len // 2 + i, layer_sizes)
            out_channels = layer_sizes[layer_sizes_len // 2 + i + 1]
            up = UpNoCat(
                name=f"{name}_up_{i}",
                in_channels=in_channels,
                out_channels=out_channels,
                device=device,
                dtype=dtype,
                layer_num=i,
            )
            self.up_layers.append(up)

        # Output layer
        self.last_layer = OutLayer(
            name=f"{name}_outlayer",
            in_channels=input_size_for_layer(layer_sizes_len - 2, layer_sizes),
            out_channels=layer_sizes[-1],
            num_convblocks=num_output_convblocks,
            device=device,
            dtype=dtype,
        )
        
        # Pre-compute dense connection metadata
        self._total_channels, self._depths, self._channel_offsets = compute_dense_metadata(layer_sizes)
        
        # Apply Kaiming initialization
        self.apply(init_weights_kaiming)

    def _build_dense_input(self, outputs: List[torch.Tensor], target_layer_idx: int) -> torch.Tensor:
        """
        Build dense input by viewing all previous outputs at target resolution.
        
        Uses a single pre-allocated tensor to avoid repeated cat allocations.
        
        Args:
            outputs: List of previous layer outputs
            target_layer_idx: Index of the layer that needs input
            
        Returns:
            Concatenated dense input tensor
        """
        batch_size = outputs[-1].size(0)
        target_seq_len = outputs[-1].size(2)
        total_channels = self._total_channels[target_layer_idx]
        
        # Allocate output buffer
        dense_input = torch.empty(
            batch_size, total_channels, target_seq_len,
            device=outputs[0].device, dtype=outputs[0].dtype
        )
        
        # Copy each previous output (viewed at target resolution) into buffer
        for prev_idx, prev_output in enumerate(outputs):
            offset, effective_channels = self._channel_offsets[target_layer_idx][prev_idx]
            # View reshapes to match target resolution (channels increase as resolution decreases)
            viewed = prev_output.view(batch_size, effective_channels, target_seq_len)
            dense_input[:, offset:offset+effective_channels, :] = viewed
        
        return dense_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with efficient dense connections.
        
        Args:
            x: Input tensor of shape [batch, 1, seq_len]
            
        Returns:
            Output tensor of shape [batch, 1, seq_len]
        """
        # First layer
        x = self.first_layer(x)
        outputs = [x]

        # Down path
        for i, down_layer in enumerate(self.down_layers):
            dense_input = self._build_dense_input(outputs, i)
            x = down_layer(dense_input)
            outputs.append(x)

        # Up path
        for i, up_layer in enumerate(self.up_layers):
            layer_idx = self.num_down + i
            dense_input = self._build_dense_input(outputs, layer_idx)
            x = up_layer(dense_input)
            outputs.append(x)

        # Final layer
        final_layer_idx = len(self.layer_sizes) - 2
        dense_input = self._build_dense_input(outputs, final_layer_idx)
        x = self.last_layer(dense_input)

        return x
