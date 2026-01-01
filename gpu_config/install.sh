#!/bin/bash
set -e

echo "Installing GPU Switching setup..."

# 1. Create config directory
sudo mkdir -p /etc/X11/gpu_configs

# 2. Copy configs
echo "Copying Xorg configs..."
sudo cp xorg-3090-primary.conf /etc/X11/gpu_configs/
sudo cp xorg-5090-primary.conf /etc/X11/gpu_configs/

# 3. Install switching script
echo "Installing switching script..."
sudo cp gpu-switch.sh /usr/local/bin/
sudo chmod +x /usr/local/bin/gpu-switch.sh

# 4. Install systemd service
echo "Installing systemd service..."
sudo cp gpu-switch.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gpu-switch.service

# 5. Install helper scripts
echo "Installing helper scripts..."
sudo cp run-on-5090.sh /usr/local/bin/run-on-5090
sudo chmod +x /usr/local/bin/run-on-5090

echo "--------------------------------------------------------"
echo "Installation complete."
echo "Default behavior: The system will use the RTX 3090 as primary."
echo "To use the RTX 5090, add 'gpu_primary=5090' to your kernel parameters at boot."
echo "--------------------------------------------------------"
echo "USAGE (Run app on 5090 while 3090 is primary):"
echo "  run-on-5090 <application>"
echo "  Example: run-on-5090 minecraft-launcher"
echo "--------------------------------------------------------"
