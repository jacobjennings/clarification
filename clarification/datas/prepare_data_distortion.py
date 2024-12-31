import argparse
import copy
import os
import logging
import secrets
import numpy
import copy
import threading
import concurrent.futures
import multiprocessing
import itertools
import csv
import pathlib
import logging

logger = logging.getLogger(__name__)
import h5py

from ipywidgets import IntProgress
from IPython.display import display
from IPython.display import Audio
import time

from more_itertools import chunked
from multiprocessing import Pool, TimeoutError, Value

# PyTorch model and training necessities
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
from torch.utils.data import DataLoader

# Audio
import torchaudio
import torchaudio.transforms as TAT

from torio.io import CodecConfig

# Image datasets and image manipulation
import torchvision
import torchvision.transforms as transforms

# Image display
import matplotlib.pyplot as plt
import numpy as np

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

base_dataset_directory = '/home/jacob/cv-corpus-17.0-2024-03-15/en'

# Uncomment these. Safety measure to avoid accidental use.
# out_dataset_directory = '/workspace/distorted-commonvoice-24k-300ms-10ms-opus2/en'

num_processed = Value("l", 0)

sample_ms = 300
overlap_ms = 10

megachunk_size = 1000

consumption_batch_size = 8

resample_lock = threading.Lock()

gain_values = [0., 20., 40.]

def discard_remainder(tensors, split_size):
    if tensors[-1].size(0) != split_size:
        tensors = tensors[:-1]
    return tensors

def better_split_discard_remainder(tensor, split_size):
    splits = torch.split(tensor, split_size)

    # Check the size of the last group and discard if necessary
    if splits[-1].size(0) != split_size:
        splits = splits[:-1]

    return splits

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def overlapping_samples(audio, sample_size, overlap_size):
    chunks = audio.unfold(dimension=0, size=sample_size, step=sample_size - overlap_size)
    return chunks


def process_file(data):
    add_noise = AddGaussianNoise(std=0.1)

    (data_batch, batch_idx), megachunk_idx, (data_loader_len, t0, sample_size, overlap_size, resample_rate) = data

    gain_to_batches = {}
    for gain in gain_values:
        noisy_subsamples_aggregate = None
        for data in data_batch:

            if num_processed.value % 500 == 0 and num_processed.value != 0:
                t1 = time.time()
                elapsed = t1 - t0
                files_per_second = num_processed.value / elapsed
                print(
                    f"Processed {num_processed.value}/{data_loader_len} files "
                    f"({num_processed.value / data_loader_len * 100:.0f}%). "
                    f"Elapsed: {elapsed:.0f}s. "
                    f"Files/second: {files_per_second:.0f}. "
                    f"Remaining estimated hours: {(data_loader_len - num_processed.value) / files_per_second / 60 / 60:.2f}")

            num_processed.value += 1

            original_waveform = data[0].squeeze()
            sample_rate = data[1][0].item()
            resampler = TAT.Resample(sample_rate, resample_rate, dtype=torch.float32)
            # with resample_lock:
            resampled_clear = resampler(original_waveform)
            # noisy_waveform = add_noise(noisy_waveform)
            noisy_subsamples = overlapping_samples(resampled_clear, sample_size=sample_size, overlap_size=overlap_size)
            if noisy_subsamples_aggregate is not None:
                noisy_subsamples_aggregate = torch.cat([noisy_subsamples_aggregate, noisy_subsamples], dim=0)
            else:
                noisy_subsamples_aggregate = noisy_subsamples

        noisy_batches = better_split_discard_remainder(noisy_subsamples_aggregate, consumption_batch_size)
        noisy_batches = [
            torchaudio.functional.overdrive(batch, gain=gain)
            for batch in noisy_batches
        ]
        gain_to_batches[gain] = noisy_batches

    clips_dir = f"{out_dataset_directory}/{megachunk_idx}/clips"
    pathlib.Path(clips_dir).mkdir(parents=True, exist_ok=True)

    write_dicts = []
    for gain, noisy_batches in gain_to_batches.items():
        for idx, noisy_batch in enumerate(noisy_batches):
            relative_path = f"{megachunk_idx}/clips/{batch_idx}_{idx}_{gain}.opus"

            path_absolute = f"{out_dataset_directory}/{megachunk_idx}/clips/{batch_idx}_{idx}_{gain}.opus"

            noisy_full = noisy_batch.reshape(-1)
            noisy_full = noisy_full.to("cpu").unsqueeze(1)

            torchaudio.save(
                uri=path_absolute,
                src=noisy_full,
                sample_rate=resample_rate,
                format="opus",
                channels_first=False,
            )

            write_dicts.append({
                "path": relative_path,
                "gain": gain
            })

    return write_dicts


class ChunkIterable:
    def __init__(self, chunk_size, length):
        self.chunk_size = chunk_size
        self.length = length
        self.lock = threading.Lock()
        self.count = 0
        self.chunk_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        chunk_count = 0
        with self.lock:
            self.count += 1
            if self.count == self.chunk_size:
                self.chunk_count += 1
                self.count = 0
            chunk_count = self.chunk_count

        return chunk_count

class EnumeratedIter:
    def __init__(self, it):
        self.it = it
        self.index = 0

    def __next__(self):
        retval = (next(self.it), self.index)
        self.index += 1
        return retval

    def __iter__(self):
        return self

def process_data():
    resample_rate = 24000
    sample_size = int((sample_ms / 1000) * resample_rate)
    overlap_size = int((overlap_ms / 1000) * resample_rate)

    print(torch.__version__)
    print(torchaudio.__version__)

    pathlib.Path(out_dataset_directory).mkdir(parents=True, exist_ok=True)

    training_speech_dataset = torchaudio.datasets.COMMONVOICE(root=base_dataset_directory)

    training_speech_dataset_noisy = copy.deepcopy(training_speech_dataset)

    data_loader = DataLoader(
        training_speech_dataset_noisy,
        batch_size=1,
        num_workers=16)

    print(torchaudio.utils.ffmpeg_utils.get_audio_encoders())
    print(torchaudio.list_audio_backends)

    print(f"Detected {os.cpu_count()} cpus.")

    data_loader_len = len(data_loader)

    t0 = time.time()

    process_count = 2

    chunk_iter = ChunkIterable(megachunk_size, data_loader_len)

    with open(f"{out_dataset_directory}/info.csv", 'w', newline='\n') as csvfile:
        fieldnames = ['sample_rate', 'sample_size', 'overlap_size', 'consumption_batch_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "sample_rate": resample_rate,
            "sample_size": sample_size,
            "overlap_size": overlap_size,
            "consumption_batch_size": consumption_batch_size
        })

    enumerated_batched_iter = EnumeratedIter(itertools.batched(data_loader, 32))
    zipped_dataset = zip(
        enumerated_batched_iter, chunk_iter, itertools.repeat((data_loader_len, t0, sample_size, overlap_size, resample_rate)))

    write_path = f"{out_dataset_directory}/samples.csv"
    with open(write_path, 'w', newline='\n') as csvfile:
        fieldnames = ['path', 'gain']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for x in zipped_dataset:
            write_dicts = process_file(x)
            print(".", sep="", end="")
            for write_dict in write_dicts:
                writer.writerow(write_dict)

        # with Pool(processes=process_count) as pool:
        #     for write_dicts in pool.imap_unordered(process_file, zipped_dataset, chunksize=8):
        #         print(".", sep="", end="")
        #         for write_dict in write_dicts:
        #             writer.writerow(write_dict)

if __name__ == '__main__':
    process_data()
