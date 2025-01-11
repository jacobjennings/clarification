#!/bin/bash +x

cd /workspace || exit 1

echo "Copying clarification to cloud"

INSTANCE=$(/workspace/clarification/venv/bin/vastai show instances | grep -v 'SSH Addr' | awk '{print $1}')
export INSTANCE

VSSH_URL=$(/workspace/clarification/venv/bin/vastai ssh-url "$INSTANCE")

# Remove ssh:// prefix
VSSH_TARGET=${VSSH_URL#ssh://}

# Extract the port from the end (looks like :12345)
VSSH_PORT=${VSSH_TARGET##*:}

# Remove the port from the end of VSSH_TARGET
VSSH_TARGET=${VSSH_TARGET%:*}

rsync -vz -e "ssh -p $VSSH_PORT -i /home/jacob/.ssh/id_ed25519_vast1" --exclude={'venv'} /workspace/clarification "$VSSH_TARGET":/workspace/clarification
