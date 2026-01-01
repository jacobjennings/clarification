#!/bin/bash
# Wrapper to run applications on the RTX 5090 (NVIDIA-0 / PCI:01:00.0)
# Usage: run-on-5090 <command> [args...]

export __NV_PRIME_RENDER_OFFLOAD=1

# Try using the specific provider name from xrandr.
# Usually the sink/offload provider is the one we want.
export __NV_PRIME_RENDER_OFFLOAD_PROVIDER=NVIDIA-G0

export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only

echo "Launching on RTX 5090 (Provider: NVIDIA-G0)..."
exec "$@"

