import csv

import torch
from torch.utils.data import Dataset
import h5py

class NoisyCommonsDataset(Dataset):
    def __init__(self, batch_size, base_dir='/workspace/noisy-commonvoice-24k-300ms-10ms-h5py/en'):
        self.base_dir = base_dir
        bigfile = h5py.File(base_dir + "/clarification-dataset.hdf5", 'r')
        self.dataset = bigfile['noisy_clear_samples_300ms_10ms_overlap']
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, batch_idx):
        idx = batch_idx * self.batch_size

        datas = [
            torch.from_numpy(self.dataset[idx])
            for idx in range(idx, idx + self.batch_size)
        ]

        stacked = torch.stack(datas)

        return stacked
