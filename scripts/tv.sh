#!/bin/bash

name="${TAR_ARCHIVE%.[0-9]*}"

m_one=$((TAR_VOLUME - 1))

echo "$name.$m_one" >&$TAR_FD
