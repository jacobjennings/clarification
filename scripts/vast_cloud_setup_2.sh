#!/bin/zsh -x

cd /workspace || exit 1

# Check if /workspace/clarification has been copied to the cloud, otherwise print and exit
if [ ! -d "/workspace/clarification" ]; then
  echo "Error: /workspace/clarification has not been copied to the cloud."
  exit 1
fi

# Check if clarification-ds1.tar.0 has been copied to the cloud, otherwise print and exit
if [ ! -f "/workspace/clarification-ds1.tar.0" ]; then
  echo "Error: /workspace/clarification-ds1.tar.0 has not been copied to the cloud."
  exit 1
fi

echo "Installing ohmyzsh"
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" || exit 1

echo "cd /workspace" >> ~/.zshrc
echo "export PYTHONUNBUFFERED=1" >> ~/.zshrc
echo "export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.7" >> ~/.zshrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init - bash)"' >> ~/.zshrc

curl -fsSL https://pyenv.run | bash || exit 1

source ~/.zshrc

echo "Extracting dataset"
tar -x -M -F /workspace/clarification/scripts/tv.sh -f clarification-ds1.tar.0 & progress -mp $! || exit 1

mkdir mounted_image
mv noisy-commonvoice-24k-300ms-5ms-opus mounted_image/

echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen
locale-gen || exit 1

pyenv install 3.12.3 || exit 1
pyenv global 3.12.3 || exit 1

cd clarification || exit 1
python3 -m venv venv || exit 1
venv/bin/pip install --find-links /workspace/wheelcache -r requirements.txt || exit 1

