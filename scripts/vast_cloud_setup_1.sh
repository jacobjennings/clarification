#!/bin/bash -x

apt install -y vim ack tree zsh nvtop progress

apt install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl git \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

chsh root -s /bin/zsh
echo "set-option -g default-shell /bin/zsh" >> ~/.tmux.conf
echo "set -g mouse on" >> ~/.tmux.conf
echo "bind -n WheelDownPane if-shell -F -t = '#{mouse_any_flag}' 'send-keys -M' 'if -Ft= \"#{pane_in_mode}\" \"send-keys -M\" \"copy-mode -e\"'" >> ~/.tmux.conf

echo "zsh set as default. call exit() and ssh back in, then run vast_cloud_setup_2.sh"
