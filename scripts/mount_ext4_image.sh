#!/bin/bash +x

if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sudo mount_ext4_image my.img mount_dir"
    exit 1
fi

mkdir -p "$2"
mount "$1" "$2"
chmod og+rw "$2"

echo "Mounted $1 at $2."

