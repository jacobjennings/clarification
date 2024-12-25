#!/bin/bash +x

rsync -av --exclude=".*" --exclude='runs/' --exclude='venv' clarification cloudcopies/clarification
