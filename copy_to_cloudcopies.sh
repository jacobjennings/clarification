#!/bin/bash +x

cd /workspace || exit 1
rm -r cloudcopies/clarification
rsync -av --exclude=".*" --exclude='runs/' --exclude='venv' clarification cloudcopies/

INSTANCE=$(/workspace/clarification/venv/bin/vastai show instances | grep -v 'SSH Addr' | awk '{print $1}')
export INSTANCE

/workspace/clarification/venv/bin/vastai copy -i /home/jacob/.ssh/id_ed25519_vast1 /workspace/cloudcopies/clarification "$INSTANCE":/workspace/
