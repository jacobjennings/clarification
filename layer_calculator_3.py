"""
Layer calculator for finding viable ClarificationDense architectures.
Computes parameters analytically without creating models for performance.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import itertools


def conv1d_params(in_channels: int, out_channels: int, kernel_size: int = 3) -> int:
    """Parameters for Conv1d: weight + bias."""
    return in_channels * out_channels * kernel_size + out_channels


def batchnorm_params(channels: int) -> int:
    """Parameters for BatchNorm1d: weight + bias."""
    return 2 * channels


def conv_transpose1d_params(in_channels: int, out_channels: int, kernel_size: int = 2) -> int:
    """Parameters for ConvTranspose1d: weight + bias."""
    return in_channels * out_channels * kernel_size + out_channels


def convblock1d_params(in_channels: int, out_channels: int, num_blocks: int = 2, last_layer: bool = False) -> int:
    """
    Parameters for ConvBlock1D.
    
    Structure:
    - Conv1d(in_channels → out_channels) + BatchNorm + ReLU
    - For each additional block:
      - Conv1d(out_channels → out_channels or 1 if last) + BatchNorm + ReLU (unless last)
    """
    total = 0
    
    # First conv + batchnorm
    total += conv1d_params(in_channels, out_channels, kernel_size=3)
    total += batchnorm_params(out_channels)
    
    # Additional blocks
    for i in range(num_blocks - 1):
        is_last = (i == num_blocks - 2) and last_layer
        final_out = 1 if is_last else out_channels
        
        total += conv1d_params(out_channels, final_out, kernel_size=3)
        
        if not is_last:
            total += batchnorm_params(out_channels)
    
    return total


def input_size_for_layer(layer_num: int, layer_sizes: List[int]) -> int:
    """
    Calculate the input channel size for a layer based on dense connections.
    Mirrors the function in clarification_dense.py.
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
        adding_size = int(layer_sizes[i] * (2 ** relative_depth))
        layer_size += adding_size
    
    return layer_size


def calculate_dense_params(layer_sizes: List[int], num_output_convblocks: int = 2) -> int:
    """
    Calculate total parameters for ClarificationDense architecture.
    
    Architecture:
    - first_layer: ConvBlock1D(1, layer_sizes[0])
    - down_layers: len(layer_sizes) // 2 Down layers
    - up_layers: len(layer_sizes) // 2 - 1 UpNoCat layers  
    - last_layer: OutLayer
    """
    if len(layer_sizes) % 2 == 0:
        raise ValueError("layer_sizes must have odd length")
    
    total = 0
    n_layers = len(layer_sizes)
    
    # First layer: ConvBlock1D(in=1, out=layer_sizes[0])
    total += convblock1d_params(1, layer_sizes[0])
    
    # Down layers
    num_down = n_layers // 2
    for i in range(num_down):
        in_ch = input_size_for_layer(i, layer_sizes)
        out_ch = layer_sizes[i + 1]
        # Down = MaxPool1d (no params) + ConvBlock1D
        total += convblock1d_params(in_ch, out_ch)
    
    # Up layers (UpNoCat)
    num_up = num_down - 1
    for i in range(num_up):
        in_ch = input_size_for_layer(n_layers // 2 + i, layer_sizes)
        out_ch = layer_sizes[n_layers // 2 + i + 1]
        # UpNoCat = ConvTranspose1d + ConvBlock1D
        total += conv_transpose1d_params(in_ch, out_ch, kernel_size=2)
        total += convblock1d_params(out_ch, out_ch)
    
    # OutLayer
    out_in_ch = input_size_for_layer(n_layers - 2, layer_sizes)
    out_out_ch = layer_sizes[-1]
    
    # ConvTranspose1d
    total += conv_transpose1d_params(out_in_ch, out_out_ch, kernel_size=2)
    
    # num_convblocks - 1 regular ConvBlock1D
    for i in range(num_output_convblocks - 1):
        total += convblock1d_params(out_out_ch, out_out_ch)
    
    # Final ConvBlock1D with last_layer=True, out_channels=1
    total += convblock1d_params(out_out_ch, 1, last_layer=True)
    
    return total


@dataclass
class ArchitectureResult:
    layer_sizes: Tuple[int, ...]
    params: int
    shape_type: str  # 'bottleneck', 'expansion', 'flat', 'asymmetric'
    
    def __repr__(self):
        return f"[{', '.join(map(str, self.layer_sizes))}] = {self.params:,} params ({self.shape_type})"


def classify_shape(layer_sizes: List[int]) -> str:
    """Classify the shape of the architecture."""
    n = len(layer_sizes)
    mid = n // 2
    
    first = layer_sizes[0]
    middle = layer_sizes[mid]
    last = layer_sizes[-1]
    
    # Check symmetry
    is_symmetric = all(layer_sizes[i] == layer_sizes[n - 1 - i] for i in range(n // 2))
    
    if not is_symmetric:
        return "asymmetric"
    
    if middle < first:
        return "bottleneck"
    elif middle > first:
        return "expansion"
    else:
        return "flat"


def find_architectures(
    target_params: int,
    tolerance: float = 0.15,
    min_layers: int = 3,
    max_layers: int = 9,
    layer_size_step: int = 8,
    min_layer_size: int = 16,
    max_layer_size: int = 256,
    symmetric_only: bool = False,
    shape_filter: Optional[str] = None,  # 'bottleneck', 'expansion', 'flat', None for all
    num_output_convblocks: int = 2,
) -> List[ArchitectureResult]:
    """
    Find dense architectures within the target parameter range.
    
    Args:
        target_params: Target number of parameters
        tolerance: Acceptable range as fraction (0.15 = ±15%)
        min_layers: Minimum number of layers (must be odd)
        max_layers: Maximum number of layers (must be odd)
        layer_size_step: Step size for layer sizes
        min_layer_size: Minimum size per layer
        max_layer_size: Maximum size per layer
        symmetric_only: If True, only generate symmetric architectures
        shape_filter: Filter by shape type ('bottleneck', 'expansion', 'flat', None for all)
        num_output_convblocks: Number of conv blocks in output layer
        
    Returns:
        List of ArchitectureResult sorted by closeness to target
    """
    min_params = int(target_params * (1 - tolerance))
    max_params = int(target_params * (1 + tolerance))
    
    results = []
    
    # Ensure odd layer counts
    layer_counts = [n for n in range(min_layers, max_layers + 1) if n % 2 == 1]
    layer_sizes_range = range(min_layer_size, max_layer_size + 1, layer_size_step)
    
    for num_layers in layer_counts:
        if symmetric_only:
            # Generate only symmetric architectures (much faster)
            half = num_layers // 2 + 1
            for combo in itertools.product(layer_sizes_range, repeat=half):
                # Mirror to create full architecture
                layers = list(combo) + list(reversed(combo[:-1]))
                
                params = calculate_dense_params(layers, num_output_convblocks)
                
                if min_params <= params <= max_params:
                    shape = classify_shape(layers)
                    if shape_filter is None or shape == shape_filter:
                        results.append(ArchitectureResult(
                            layer_sizes=tuple(layers),
                            params=params,
                            shape_type=shape
                        ))
        else:
            # Generate all permutations (slower but finds asymmetric ones)
            # Limit search space based on layer count
            if num_layers <= 3:
                max_range = min(max_layer_size, 160)
            elif num_layers <= 5:
                max_range = min(max_layer_size, 128)
            elif num_layers <= 7:
                max_range = min(max_layer_size, 96)
            else:
                max_range = min(max_layer_size, 72)
            
            limited_range = range(min_layer_size, max_range + 1, layer_size_step)
            
            for combo in itertools.product(limited_range, repeat=num_layers):
                params = calculate_dense_params(list(combo), num_output_convblocks)
                
                if min_params <= params <= max_params:
                    shape = classify_shape(list(combo))
                    if shape_filter is None or shape == shape_filter:
                        results.append(ArchitectureResult(
                            layer_sizes=tuple(combo),
                            params=params,
                            shape_type=shape
                        ))
    
    # Sort by closeness to target
    results.sort(key=lambda x: abs(x.params - target_params))
    
    return results


def find_bottleneck_architectures(
    target_params: int,
    tolerance: float = 0.15,
    min_layers: int = 3,
    max_layers: int = 9,
    **kwargs
) -> List[ArchitectureResult]:
    """Convenience function to find only bottleneck architectures."""
    return find_architectures(
        target_params=target_params,
        tolerance=tolerance,
        min_layers=min_layers,
        max_layers=max_layers,
        symmetric_only=True,
        shape_filter='bottleneck',
        **kwargs
    )


def print_as_config(results: List[ArchitectureResult], name_prefix: str = "dense", limit: int = 50):
    """Print results in the format used by experiment_1.py."""
    for i, r in enumerate(results[:limit]):
        layers_str = ", ".join(map(str, r.layer_sizes))
        print(f'# {r.params:,} params - {r.shape_type}')
        print(f'self.dense_config("{name_prefix}{i+1}", [{layers_str}]),')
        print()


def main():
    # ==================== CONFIGURATION ====================
    TARGET_PARAMS = 250_000
    TOLERANCE = 0.15  # ±15%
    
    MIN_LAYERS = 3
    MAX_LAYERS = 7
    
    # Set to True to only find symmetric architectures (much faster)
    SYMMETRIC_ONLY = True
    
    # Filter by shape: 'bottleneck', 'expansion', 'flat', or None for all
    SHAPE_FILTER = None  # Try 'bottleneck' based on your dense5 findings
    
    # ===========================================================
    
    print(f"Searching for architectures with ~{TARGET_PARAMS:,} params (±{TOLERANCE*100:.0f}%)")
    print(f"Range: {int(TARGET_PARAMS * (1 - TOLERANCE)):,} - {int(TARGET_PARAMS * (1 + TOLERANCE)):,}")
    print(f"Layers: {MIN_LAYERS}-{MAX_LAYERS}, symmetric_only={SYMMETRIC_ONLY}")
    print()
    
    results = find_architectures(
        target_params=TARGET_PARAMS,
        tolerance=TOLERANCE,
        min_layers=MIN_LAYERS,
        max_layers=MAX_LAYERS,
        symmetric_only=SYMMETRIC_ONLY,
        shape_filter=SHAPE_FILTER,
    )
    
    print(f"Found {len(results)} architectures\n")
    
    # Group by shape type
    by_shape = {}
    for r in results:
        by_shape.setdefault(r.shape_type, []).append(r)
    
    for shape, archs in sorted(by_shape.items()):
        print(f"=== {shape.upper()} ({len(archs)} found) ===")
        for r in archs[:10]:  # Top 10 per shape
            print(f"  {r}")
        if len(archs) > 10:
            print(f"  ... and {len(archs) - 10} more")
        print()
    
    # Print configs for best results
    print("\n" + "=" * 60)
    print("CONFIG OUTPUT (top 20 closest to target):")
    print("=" * 60 + "\n")
    print_as_config(results, name_prefix="dense250k_", limit=20)


if __name__ == "__main__":
    main()

