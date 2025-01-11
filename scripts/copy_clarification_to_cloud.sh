#!/bin/bash +x

cd /workspace || exit 1

echo "Copying clarification to cloud"

source /workspace/clarification/scripts/source_me_for_exports.sh

# Check if the clarification directory exists on the remote host
if ssh -p "$VSSH_PORT" -i /home/jacob/.ssh/id_ed25519_vast1 "$VSSH_TARGET" "[ -d /workspace/clarification ]"; then
  echo "Clarification directory exists on remote host"
  exit 0
else
  echo "Clarification directory does not exist on remote host, syncing"
  rsync -rvz -e "ssh -p $VSSH_PORT -i /home/jacob/.ssh/id_ed25519_vast1" --exclude='venv' /workspace/clarification "$VSSH_TARGET":/workspace/
fi

# /root/.cache/pip/wheels/89/77/2b/763542583cafc50c5f7396f7ef1295678a6f560e337e37fa98

