#!/bin/bash +x

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

# Remove the user, just leaving the ip address
VSSH_IP=${VSSH_TARGET#*@}

# Check if ssh config contains this instance's IP
if grep -q "$VSSH_IP" /home/jacob/.ssh/config; then
  echo "SSH config already contains $VSSH_IP"
  exit 0
else
  echo "Adding $VSSH_IP to SSH config"
fi

echo "
Host $VSSH_IP
  HostName $VSSH_IP
  Port $VSSH_PORT
  User root
  IdentityFile /home/jacob/.ssh/id_ed25519_vast1
  ForwardAgent yes
" >> /home/jacob/.ssh/config
