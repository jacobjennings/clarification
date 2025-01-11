#!/bin/bash

# Creates an SSH tunnel from remote 6006 to local 6007 (to avoid stomping on any local tensorboards)

INSTANCE=$(/workspace/clarification/venv/bin/vastai show instances | grep -v 'SSH Addr' | awk '{print $1}')
export INSTANCE

VSSH_URL=$(/workspace/clarification/venv/bin/vastai ssh-url "$INSTANCE")

# Remove ssh:// prefix
VSSH_TARGET=${VSSH_URL#ssh://}

# Extract the port from the end (looks like :12345)
VSSH_PORT=${VSSH_TARGET##*:}

# Remove the port from the end of VSSH_TARGET
VSSH_TARGET=${VSSH_TARGET%:*}

echo "Connecting to $VSSH_TARGET on port $VSSH_PORT"

if [[ $# -ne 1 ]]; then
  CERT_LOCATION="~/.ssh/id_ed25519_vast1"
else
  CERT_LOCATION="$1"
fi

set -x

ssh -p "$VSSH_PORT" -i "$CERT_LOCATION" "$VSSH_TARGET" -L 6007:localhost:6006
