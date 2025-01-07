#!/bin/bash +x

INSTANCE=$(/workspace/clarification/venv/bin/vastai show instances | grep -v 'SSH Addr' | awk '{print $1}')
export INSTANCE

/workspace/clarification/venv/bin/vastai cloud copy --instance "$INSTANCE" --connection 20482 --src jjdatasets/clarification --dst /workspace/images --transfer "Cloud To Instance" --explain
