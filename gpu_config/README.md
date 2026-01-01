# GPU Switching Setup

This setup allows you to switch between using the RTX 3090 and RTX 5090 as your primary display GPU on boot, using kernel parameters.

## Prerequisites
- RTX 5090 at PCI Bus 01:00.0
- RTX 3090 at PCI Bus 07:00.0

## Installation

1. Open a terminal in this directory (`gpu_config`).
2. Run the install script:
   ```bash
   sudo bash install.sh
   ```

## Usage

### Default Mode (RTX 3090)
By default, the system will now configure the **RTX 3090** as the primary GPU. Just boot normally.

### Gaming Mode (RTX 5090)
To use the RTX 5090 as the primary display (e.g., for gaming):
1. Reboot your computer.
2. When the GRUB boot menu appears, verify the selection is on "Ubuntu".
3. Press **`e`** to edit the boot commands.
4. Find the line starting with `linux`.
5. Append `gpu_primary=5090` to the end of that line.
6. Press **F10** to boot.

## Creating a Permanent GRUB Entry

To add a permanent "Gaming Mode" option to your boot menu:

1. Read your current grub configuration:
   ```bash
   sudo cat /boot/grub/grub.cfg
   ```
2. Copy the entire first `menuentry 'Ubuntu ...' { ... }` block (including the closing brace `}`).
3. Open the custom grub configuration file:
   ```bash
   sudo nano /etc/grub.d/40_custom
   ```
4. Paste the menu entry at the end of the file.
5. Edit the pasted entry:
   - Change the name in quotes: `menuentry 'Ubuntu (RTX 5090 Gaming)' ...`
   - Find the line starting with `linux`. Add `gpu_primary=5090` to the end of it.
6. Save and exit (Ctrl+O, Enter, Ctrl+X).
7. Update GRUB:
   ```bash
   sudo update-grub
   ```

