#!/bin/bash

cd /workspace || exit 1

INSTANCE=$(/workspace/clarification/venv/bin/vastai show instances | grep -v 'SSH Addr' | awk '{print $1}')
export INSTANCE

VSSH_URL=$(/workspace/clarification/venv/bin/vastai ssh-url "$INSTANCE")

# Remove ssh:// prefix
VSSH_TARGET=${VSSH_URL#ssh://}

# Extract the port from the end (looks like :12345)
VSSH_PORT=${VSSH_TARGET##*:}

# Remove the port from the end of VSSH_TARGET
VSSH_TARGET=${VSSH_TARGET%:*}

/workspace/clarification/scripts/populate_ssh_agent_forwarding.sh

/workspace/clarification/scripts/copy_clarification_to_cloud.sh

/workspace/clarification/scripts/copy_wheelcache_to_cloud.sh

ssh -p "$VSSH_PORT" -i /home/jacob/.ssh/id_ed25519_vast1 "$VSSH_TARGET" "zsh /workspace/clarification/scripts/vast_cloud_setup_1.sh"
return_code=$?

# Check the return code
if [ $return_code -ne 0 ]; then
  echo "Error: vast_cloud_setup_1.sh failed with return code $return_code"
  exit $return_code
fi

ssh -p "$VSSH_PORT" -i /home/jacob/.ssh/id_ed25519_vast1 "$VSSH_TARGET" "zsh /workspace/clarification/scripts/vast_cloud_setup_2.sh"
return_code=$?

# Check the return code
if [ $return_code -ne 0 ]; then
  echo "Error: vast_cloud_setup_2.sh failed with return code $return_code"
  exit $return_code
fi

