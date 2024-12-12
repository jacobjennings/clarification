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
out_dataset_directory = '/workspace/noisy-commonvoice-24k-300ms-10ms-h5py-opus/en'

num_processed = Value("l", 0)

sample_ms = 300
overlap_ms = 10

megachunk_size = 1000

consumption_batch_size = 8

resample_lock = threading.Lock()

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
        return tensor + torch.randn(tensor.size(), device="cuda") * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def overlapping_samples(audio, sample_size, overlap_size):
    chunks = audio.unfold(dimension=0, size=sample_size, step=sample_size - overlap_size)
    return chunks


def process_file(data):
    add_noise = AddGaussianNoise(std=0.1)

    (data_batch, batch_idx), megachunk_idx, (data_loader_len, t0, sample_size, overlap_size, resample_rate) = data

    resampled_clear_subsamples_aggregate = None
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
        resampler = TAT.Resample(sample_rate, resample_rate, dtype=torch.float32).to("cuda")
        # with resample_lock:
        resampled_clear = resampler(original_waveform.to("cuda"))
        noisy_waveform = add_noise(resampled_clear)
        resampled_clear_subsamples = overlapping_samples(resampled_clear, sample_size=sample_size, overlap_size=overlap_size)
        noisy_subsamples = overlapping_samples(noisy_waveform, sample_size=sample_size, overlap_size=overlap_size)
        if resampled_clear_subsamples_aggregate is not None:
            resampled_clear_subsamples_aggregate = torch.cat([resampled_clear_subsamples_aggregate, resampled_clear_subsamples], dim=0)
            noisy_subsamples_aggregate = torch.cat([noisy_subsamples_aggregate, noisy_subsamples], dim=0)
        else:
            resampled_clear_subsamples_aggregate = resampled_clear_subsamples
            noisy_subsamples_aggregate = noisy_subsamples

    noisy_batches = better_split_discard_remainder(noisy_subsamples_aggregate, consumption_batch_size)
    clear_batches = better_split_discard_remainder(resampled_clear_subsamples_aggregate, consumption_batch_size)

    clips_dir = f"{out_dataset_directory}/{megachunk_idx}/clips"
    pathlib.Path(clips_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{out_dataset_directory}/{megachunk_idx}/info.csv", 'a', newline='\n') as csvfile:
        for idx, (noisy_batch, clear_batch) in enumerate(zip(noisy_batches, clear_batches)):
            relative_path = f"{batch_idx}_{idx}.opus"

            path_absolute = f"{out_dataset_directory}/{megachunk_idx}/clips/{batch_idx}_{idx}.opus"

            noisy_full = noisy_batch.reshape(-1)
            clear_full = clear_batch.reshape(-1)

            two_channels = torch.stack([noisy_full, clear_full], dim=0).permute(1, 0).to("cpu")

            torchaudio.save(
                uri=path_absolute,
                src=two_channels,
                sample_rate=resample_rate,
                format="opus",
                channels_first=False,
            )

            fieldnames = ['path', 'sentence_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writerow({
                "path": relative_path,
                "sentence_id": data[2]["sentence_id"]
            })

    return 0


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

    process_count = 32

    num_megachunks = 0

    with open(f"{out_dataset_directory}/info.csv", 'w', newline='\n') as csvfile:
        fieldnames = ['num_megachunks', 'sample_rate', 'sample_size', 'overlap_size', 'consumption_batch_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "num_megachunks": num_megachunks,
            "sample_rate": resample_rate,
            "sample_size": sample_size,
            "overlap_size": overlap_size,
            "consumption_batch_size": consumption_batch_size
        })

    chunk_iter = ChunkIterable(megachunk_size, data_loader_len)

    enumerated_batched_iter = EnumeratedIter(itertools.batched(data_loader, 32))
    zipped_dataset = zip(
        enumerated_batched_iter, chunk_iter, itertools.repeat((data_loader_len, t0, sample_size, overlap_size, resample_rate)))

    with Pool(processes=process_count) as pool:
        for i in pool.imap_unordered(process_file, zipped_dataset, chunksize=8):
            pass

if __name__ == '__main__':
    process_data()
