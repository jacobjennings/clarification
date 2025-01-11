#!/bin/bash

source /workspace/clarification/scripts/source_me_for_exports.sh

# Check if /workspace/clarification-ds1.tar.46 exists on the remote host
if ssh -p "$VSSH_PORT" -i /home/jacob/.ssh/id_ed25519_vast1 "$VSSH_TARGET" "[ -f /workspace/clarification-ds1.tar.46 ]"; then
    echo "Clarification dataset exists on remote host"
    exit 0
else
    echo "Clarification dataset does not exist on remote host, copying"
fi

/workspace/clarification/venv/bin/vastai cloud copy --instance "$INSTANCE" --connection 20461 --src clarification-ds1-tarballs --dst /workspace/ --transfer "Cloud To Instance" --explain

while true; do
    # Fetch logs
    LOGS=$(vast logs --daemon-logs "$INSTANCE" 2>&1)

    # Check for failure in logs
    if echo "$LOGS" | grep -q "Cloud.*fail"; then
        echo "Error: Cloud operation failed."
        exit 1
    fi

    # Extract the latest progress details
    PROGRESS=$(echo "$LOGS" | grep -Eo '[0-9.]+ GiB / [0-9.]+ GiB, [0-9]+%, [0-9.]+ MiB/s, ETA [0-9m]+s' | tail -n 1)

    if [[ -n "$PROGRESS" ]]; then
        echo "Status: $PROGRESS"
    fi

    # Check if operation is complete
    if echo "$LOGS" | grep -q "Cloud Copy Operation Complete"; then
        echo "Operation complete. Continuing..."
        break
    fi

    # Wait for 15 seconds before checking again
    sleep 15
done
