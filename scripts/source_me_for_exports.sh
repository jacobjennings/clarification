#!/bin/bash

# use this to populate useful vars in env:
# source /workspace/clarification/scripts/source_me_for_exports.sh

INSTANCE=$(/workspace/clarification/venv/bin/vastai show instances | grep -v 'SSH Addr' | awk '{print $1}')
export INSTANCE

VSSH_URL=$(/workspace/clarification/venv/bin/vastai ssh-url "$INSTANCE")
export VSSH_URL

# Remove ssh:// prefix
VSSH_TARGET=${VSSH_URL#ssh://}
export VSSH_TARGET

# Extract the port from the end (looks like :12345)
VSSH_PORT=${VSSH_TARGET##*:}
export VSSH_PORT

# Remove the port from the end of VSSH_TARGET
VSSH_TARGET=${VSSH_TARGET%:*}
export VSSH_TARGET

# Remove the user, just leaving the ip address
VSSH_IP=${VSSH_TARGET#*@}
export VSSH_IP

alias vast=/workspace/clarification/venv/bin/vastai
