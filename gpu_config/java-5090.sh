#!/bin/bash
# Wrapper to force Java to run on the 5090
# Point your Minecraft Launcher "Java Executable" to this script.

export __NV_PRIME_RENDER_OFFLOAD=1
export __NV_PRIME_RENDER_OFFLOAD_PROVIDER=NVIDIA-G0
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only

# Explicitly use the bundled Java found in the process list
REAL_JAVA="/home/jacob/.minecraft/runtime/java-runtime-delta/linux/java-runtime-delta/bin/java"

# Fallback if that specific file doesn't exist
if [ ! -x "$REAL_JAVA" ]; then
    echo "Warning: Specific bundled java not found at $REAL_JAVA"
    # Try finding it in the runtime folder dynamically
    REAL_JAVA=$(find /home/jacob/.minecraft/runtime -name java -type f | head -n 1)
fi

# Fallback to system java if still nothing
if [ ! -x "$REAL_JAVA" ]; then
    REAL_JAVA=$(which java)
fi

echo "Launching: $REAL_JAVA"
exec "$REAL_JAVA" "$@"
