#!/bin/bash +x

if [[ $# -ne 1 ]]; then
  echo "Error: This script requires exactly one parameter, the session name."
  exit 1
fi

INSTANCE=$(/workspace/clarification/venv/bin/vastai show instances | grep -v 'SSH Addr' | awk '{print $1}')
export INSTANCE

VSSH_URL=$(/workspace/clarification/venv/bin/vastai ssh-url "$INSTANCE")

ssh-tmux "$VSSH_URL" "$1" -i /home/jacob/.ssh/id_ed25519_vast1
