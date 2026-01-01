#!/bin/bash

# Directory where we store the source configs
CONFIG_SOURCE_DIR="/etc/X11/gpu_configs"
# Target location for Xorg
TARGET_CONFIG="/etc/X11/xorg.conf.d/10-gpu-primary.conf"

# Ensure target directory exists
mkdir -p $(dirname "$TARGET_CONFIG")

# Default to 3090 if nothing specified
GPU_MODE="3090"

# Check kernel command line
if grep -q "gpu_primary=5090" /proc/cmdline; then
    GPU_MODE="5090"
elif grep -q "gpu_primary=3090" /proc/cmdline; then
    GPU_MODE="3090"
fi

echo "Setting primary GPU to $GPU_MODE"

if [ "$GPU_MODE" == "3090" ]; then
    ln -sf "$CONFIG_SOURCE_DIR/xorg-3090-primary.conf" "$TARGET_CONFIG"
else
    ln -sf "$CONFIG_SOURCE_DIR/xorg-5090-primary.conf" "$TARGET_CONFIG"
fi

