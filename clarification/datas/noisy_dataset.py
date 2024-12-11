import csv

from torch.utils.data import Dataset

import torchaudio


class NoisyCommonsDataset(Dataset):
    def __init__(self, base_dir='/home/jacob/noisy-commonvoice-24k-300ms-10ms/en'):
        self.base_dir = base_dir
        with open(f"{base_dir}/info.csv") as f:
            self.chunk_datas = list(csv.reader(f))

    def __len__(self):
        return len(self.chunk_datas)

    def __getitem__(self, idx):
        data = self.chunk_datas[idx]

        noisy_chunks = []
        clear_chunks = []
        for idx in range(int(data[2])):
            noisy_path = self.base_dir + "/clips/" + data[0] + "_" + str(idx) + ".mp3"
            clear_path = self.base_dir + "/clips/" + data[1] + "_" + str(idx) + ".mp3"
            noisy, _ = torchaudio.load(noisy_path)
            clear, _ = torchaudio.load(clear_path)
            noisy_chunks.append(noisy)
            clear_chunks.append(clear)

        return noisy_chunks, clear_chunks, data
