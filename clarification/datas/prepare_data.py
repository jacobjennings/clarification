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
out_dataset_directory = '/workspace/noisy-commonvoice-24k-300ms-10ms-h5py/en'

num_processed = Value("l", 0)

sample_ms = 300
overlap_ms = 10

megachunk_size = 10000

resample_lock = threading.Lock()

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

    data_batch, megachunk_idx, (data_loader_len, t0, sample_size, overlap_size, resample_rate) = data

    resampled_clear_chunks_aggregate = None
    noisy_chunks_aggregate = None
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
        resampled_clear_chunks = overlapping_samples(resampled_clear, sample_size=sample_size, overlap_size=overlap_size)
        noisy_chunks = overlapping_samples(noisy_waveform, sample_size=sample_size, overlap_size=overlap_size)
        if resampled_clear_chunks_aggregate is not None:
            resampled_clear_chunks_aggregate = torch.cat([resampled_clear_chunks_aggregate, resampled_clear_chunks], dim=0)
            noisy_chunks_aggregate = torch.cat([noisy_chunks_aggregate, noisy_chunks], dim=0)
        else:
            resampled_clear_chunks_aggregate = resampled_clear_chunks
            noisy_chunks_aggregate = noisy_chunks

    pairs = torch.stack([noisy_chunks_aggregate, resampled_clear_chunks_aggregate], dim=1).to("cpu")

    return pairs


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

    outfile = h5py.File(f"{out_dataset_directory}/clarification-dataset.hdf5", 'w')
    outfile_dataset = outfile.create_dataset(
        "noisy_clear_samples_300ms_10ms_overlap",
        shape=(20000000, 2, sample_size), #  20000000
        maxshape=(None, 2, sample_size),
        chunks=(5, 2, sample_size),
        dtype=numpy.float32
    )

    chunk_iter = ChunkIterable(megachunk_size, data_loader_len)
    zipped_dataset = zip(
        itertools.batched(data_loader, 16), chunk_iter, itertools.repeat((data_loader_len, t0, sample_size, overlap_size, resample_rate)))

    write_index = 0

    with Pool(processes=process_count) as pool:
        for pairs in pool.imap_unordered(process_file, zipped_dataset, chunksize=8):
            outfile_size = outfile_dataset.shape[0]
            pair_size = pairs.shape[0]

            if write_index + pair_size > outfile_size:
                outfile_dataset.resize(outfile_size + pair_size, axis=0)

            outfile_dataset[write_index:write_index + pair_size] = pairs
            write_index += pair_size


if __name__ == '__main__':
    process_data()
