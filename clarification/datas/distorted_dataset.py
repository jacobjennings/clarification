import logging

logger = logging.getLogger(__name__)
import torch
from torch.utils.data import Dataset

import torchaudio

from .noisy_dataset import *


def noise(audio, stddev=0.1, mean=0):
    noisy = torch.randn_like(audio) * stddev + mean

    # Scale noise to -1, 1
    noisy = noisy / (torch.max(torch.abs(noisy)))

    # Add noise to the tensor
    audio_noisy = audio + noisy

    return audio_noisy


class DistortedCommonsDataset(Dataset):
    def __init__(self, batch_size, device, base_dir='/workspace/distorted-commonvoice-24k-300ms-10ms-opus/en'):
        super().__init__()
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
            self.device = device
            self.base_dir = base_dir
            if self.batch_size % self.consumption_batch_size != 0:
                raise ValueError(f"batch_size {self.batch_size} must be a multiple of consumption_batch_size {self.consumption_batch_size}")
            self.consumption_batches_multiplier = self.batch_size // self.consumption_batch_size
            # Opus always decodes at 48khz, bc torchaudio doesn't support passing input sample rate to decoder in ffmpeg
            self.resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=self.sample_rate).to(device)

        with open(f"{base_dir}/samples.csv") as csvfile:
            fieldnames = ['path', 'gain']
            samples_reader = csv.DictReader(csvfile, fieldnames=fieldnames)
            self.sample_infos = list(samples_reader)

    def __len__(self):
        return len(self.sample_infos) // self.consumption_batches_multiplier

    def __getitem__(self, batch_idx):
        grad_enabled = torch.is_grad_enabled()
        torch.no_grad()

        audio_aggregated = None
        gains = []
        for i in range(self.consumption_batches_multiplier):
            consumption_batch_idx = batch_idx * self.consumption_batches_multiplier + i
            sample_info = self.sample_infos[consumption_batch_idx]
            path = sample_info['path']
            gain = sample_info['gain']
            gains.append(gain)
            absolute_path = f"{self.base_dir}/{path}"
            audio, _ = torchaudio.load(absolute_path)
            audio = audio.to(self.device)
            audio = self.resampler(audio)
            if audio_aggregated is None:
                audio_aggregated = audio
            else:
                audio_aggregated = torch.cat((audio_aggregated, audio), dim=1)

        audio_samples = torch.stack(audio_aggregated.split(self.sample_size, dim=1))

        golden_values = torch.zeros([audio_samples.size(0)], device=self.device)

        for idx in range(len(audio_samples)):
            golden_values[idx] = float(gains[idx // self.consumption_batch_size])

        torch.set_grad_enabled(grad_enabled)

        return audio_samples, golden_values
