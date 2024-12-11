cd /workspace

apt install zip zstd vim -y
mkdir model-weights
unzip clarification.zip
unzip noisy-commonvoice-24k-300ms-10ms.zip

cd clarification-cloud
pip install -r requirements.txt
