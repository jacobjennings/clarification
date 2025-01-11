#!/bin/bash

cd /workspace || exit 1

source /workspace/clarification/scripts/source_me_for_exports.sh

/workspace/clarification/scripts/populate_ssh_agent_forwarding.sh || exit 1

/workspace/clarification/scripts/copy_dataset.sh || exit 1

/workspace/clarification/scripts/copy_clarification_to_cloud.sh || exit 1

/workspace/clarification/scripts/copy_wheelcache_to_cloud.sh || exit 1

echo "Running vast_cloud_setup_1.sh on remote"
ssh -p "$VSSH_PORT" -i /home/jacob/.ssh/id_ed25519_vast1 "$VSSH_TARGET" "bash /workspace/clarification/scripts/vast_cloud_setup_1.sh"
return_code=$?

# Check the return code
if [ $return_code -ne 0 ]; then
  echo "Error: vast_cloud_setup_1.sh failed with return code $return_code"
  exit $return_code
fi

echo "Running vast_cloud_setup_2.sh on remote"
ssh -p "$VSSH_PORT" -i /home/jacob/.ssh/id_ed25519_vast1 "$VSSH_TARGET" "zsh /workspace/clarification/scripts/vast_cloud_setup_2.sh"
return_code=$?

# Check the return code
if [ $return_code -ne 0 ]; then
  echo "Error: vast_cloud_setup_2.sh failed with return code $return_code"
  exit $return_code
fi
