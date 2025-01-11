#!/bin/bash +x

cd /workspace || exit 1

echo "Copying wheelcache to cloud"

source /workspace/clarification/scripts/source_me_for_exports.sh

# Check if /workspace/wheelcache on the remote host has more than 1 file before continuing
if ssh -p "$VSSH_PORT" -i /home/jacob/.ssh/id_ed25519_vast1 "$VSSH_TARGET" "[ \$(ls -1 /workspace/wheelcache | wc -l) -gt 1 ]"; then
  echo "wheelcache directory exists on remote host, skipping copy"
  exit 0
else
  echo "wheelcache directory does not exist on remote host, syncing"
fi

/workspace/clarification/venv/bin/vastai copy -i /home/jacob/.ssh/id_ed25519_vast1 /workspace/wheelcache "$INSTANCE":/workspace/
