#!/bin/bash +x

cd /workspace || exit 1

echo "Copying wheelcache to cloud"

INSTANCE=$(/workspace/clarification/venv/bin/vastai show instances | grep -v 'SSH Addr' | awk '{print $1}')
export INSTANCE

/workspace/clarification/venv/bin/vastai copy -i /home/jacob/.ssh/id_ed25519_vast1 /workspace/wheelcache "$INSTANCE":/workspace/
