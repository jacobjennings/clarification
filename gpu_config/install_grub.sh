#!/bin/bash
set -e

# Define the custom entries based on your current configuration
# Kernel: 6.8.0-90-generic
# Root UUID: e202a3cf-c381-47dd-8622-b4287110ba62

cat <<EOF | sudo tee -a /etc/grub.d/40_custom

menuentry 'Kubuntu (Primary GPU: RTX 3090)' --class kubuntu --class gnu-linux --class gnu --class os 'gnulinux-primary-3090' {
    recordfail
    load_video
    gfxmode \$linux_gfx_mode
    insmod gzio
    if [ x\$grub_platform = xxen ]; then insmod xzio; insmod lzopio; fi
    insmod part_gpt
    insmod ext2
    search --no-floppy --fs-uuid --set=root e202a3cf-c381-47dd-8622-b4287110ba62
    linux   /boot/vmlinuz-6.8.0-90-generic root=UUID=e202a3cf-c381-47dd-8622-b4287110ba62 ro  quiet splash gpu_primary=3090 \$vt_handoff
    initrd  /boot/initrd.img-6.8.0-90-generic
}

menuentry 'Kubuntu (Primary GPU: RTX 5090)' --class kubuntu --class gnu-linux --class gnu --class os 'gnulinux-primary-5090' {
    recordfail
    load_video
    gfxmode \$linux_gfx_mode
    insmod gzio
    if [ x\$grub_platform = xxen ]; then insmod xzio; insmod lzopio; fi
    insmod part_gpt
    insmod ext2
    search --no-floppy --fs-uuid --set=root e202a3cf-c381-47dd-8622-b4287110ba62
    linux   /boot/vmlinuz-6.8.0-90-generic root=UUID=e202a3cf-c381-47dd-8622-b4287110ba62 ro  quiet splash gpu_primary=5090 \$vt_handoff
    initrd  /boot/initrd.img-6.8.0-90-generic
}
EOF

echo "Added custom entries to /etc/grub.d/40_custom"
echo "Updating GRUB configuration..."
sudo update-grub
echo "Done. You should now see these options in your boot menu."



