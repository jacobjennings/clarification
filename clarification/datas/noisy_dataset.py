import csv

import torch
from torch.utils.data import Dataset

import torchaudio

class NoisyCommonsDataset(Dataset):
    def __init__(self, batch_size, device, base_dir, csv_filename):
        super().__init__()
        self.base_dir = base_dir
        self.device = device
        with open(f"{base_dir}/info.csv") as csvfile:
            fieldnames = ['sample_rate', 'sample_size', 'overlap_size', 'consumption_batch_size']
            reader = csv.DictReader(csvfile, fieldnames=fieldnames)
            _ = next(reader) # header
            row = next(reader)
            self.sample_rate = int(row['sample_rate'])
            self.sample_size = int(row['sample_size'])
            self.overlap_size = int(row['overlap_size'])
            self.consumption_batch_size = int(row['consumption_batch_size'])
            self.batch_size = batch_size
            if self.batch_size % self.consumption_batch_size != 0:
                raise ValueError(f"batch_size {self.batch_size} must be a multiple of consumption_batch_size {self.consumption_batch_size}")
            self.consumption_batches_multiplier = self.batch_size // self.consumption_batch_size
            # Opus always decodes at 48khz, bc torchaudio doesn't support passing input sample rate to decoder in ffmpeg
            self.resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=self.sample_rate).to(device)

        with open(f"{base_dir}/{csv_filename}") as csvfile:
            fieldnames = ['path', 'sentence_id']
            samples_reader = csv.DictReader(csvfile, fieldnames=fieldnames)
            self.sample_infos = list(samples_reader)

    def __len__(self):
        return len(self.sample_infos) // self.consumption_batches_multiplier

    def __getitem__(self, batch_idx):
        audio_aggregated = None
        for i in range(self.consumption_batches_multiplier):
            consumption_batch_idx = batch_idx * self.consumption_batches_multiplier + i
            sample_info = self.sample_infos[consumption_batch_idx]
            path = sample_info['path']
            absolute_path = f"{self.base_dir}/{path}"
            audio, _ = torchaudio.load(absolute_path)
            audio = audio.to(self.device)
            audio = self.resampler(audio)
            if audio_aggregated is None:
                audio_aggregated = audio
            else:
                audio_aggregated = torch.cat((audio_aggregated, audio), dim=1)

        audio_samples = torch.stack(audio_aggregated.split(self.sample_size, dim=1))

        return audio_samples
