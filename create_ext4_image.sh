#!/bin/bash +x

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: create_ext4_image 47G output.img"
    exit 1
fi

truncate -s "$1" "$2"
mkfs.ext4 "$2"
tune2fs -c0 -i0 "$2"

echo "To mount the image, run:"
echo "mkdir foo"
echo "sudo mount $2 foo"
