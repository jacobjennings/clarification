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
out_dataset_directory = '/workspace/noisy-commonvoice-24k-300ms-10ms/en'

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

    data, megachunk_idx, (data_loader_len, t0, sample_size, overlap_size, resample_rate) = data

    clips_dir = f"{out_dataset_directory}/{megachunk_idx}/clips"
    pathlib.Path(clips_dir).mkdir(parents=True, exist_ok=True)

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
    filename = data[2]["path"][0]
    original_waveform = data[0].squeeze()
    sample_rate = data[1][0].item()
    resampler = TAT.Resample(sample_rate, resample_rate, dtype=torch.float32).to("cuda")
    # with resample_lock:
    resampled_clear = resampler(original_waveform.to("cuda"))
    noisy_waveform = add_noise(resampled_clear)
    resampled_clear_chunks = overlapping_samples(resampled_clear, sample_size=sample_size, overlap_size=overlap_size)
    noisy_chunks = overlapping_samples(noisy_waveform, sample_size=sample_size, overlap_size=overlap_size)
    with open(f"{out_dataset_directory}/{megachunk_idx}/info.csv", 'a', newline='\n') as csvfile:
        fieldnames = ['noisy_path', 'clear_path', 'chunk_count', 'sentence_id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        filename_no_ext = filename.removesuffix(".mp3")

        for idx, (noisy_chunk, clear_chunk) in enumerate(zip(noisy_chunks.to("cpu"), resampled_clear_chunks.to("cpu"))):
            noisy_path = f"{filename_no_ext}_noisy_{idx}.wav"
            clear_path = f"{filename_no_ext}_clear_{idx}.wav"

            noisy_path_absolute = f"{out_dataset_directory}/{megachunk_idx}/clips/{filename_no_ext}_noisy_{idx}.wav"
            clear_path_absolute = f"{out_dataset_directory}/{megachunk_idx}/clips/{filename_no_ext}_clear_{idx}.wav"

            torchaudio.save(
                uri=noisy_path_absolute,
                src=noisy_chunk.unsqueeze(1),
                sample_rate=resample_rate,
                format="wav",
                channels_first=False
            )

            torchaudio.save(
                uri=clear_path_absolute,
                src=clear_chunk.unsqueeze(1),
                sample_rate=resample_rate,
                format="wav",
                channels_first=False
            )

            writer.writerow({
                "noisy_path": noisy_path,
                "clear_path": clear_path,
                "sentence_id": data[2]["sentence_id"]
            })

    return filename


class ChunkIterable:
    def __init__(self, chunk_size, length):
        self.chunk_size = chunk_size
        self.length = length
        self.lock = threading.Lock()
        self.count = 0
        self.chunk_count = 0

    def __iter__(self):
        return self

    def __next__(self): # Python 2: def next(self)
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

    with open(f"{out_dataset_directory}/info.csv", 'w', newline='\n') as csvfile:
        fieldnames = ['num_megachunks', 'sample_rate', 'sample_size', 'overlap_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "num_megachunks": num_megachunks,
            "sample_rate": resample_rate,
            "sample_size": sample_size,
            "overlap_size": overlap_size
        })

    chunk_iter = ChunkIterable(megachunk_size, data_loader_len)
    batched_dataset = zip(
        data_loader, chunk_iter, itertools.repeat((data_loader_len, t0, sample_size, overlap_size, resample_rate)))

    with Pool(processes=process_count) as pool:
        for i in pool.imap_unordered(process_file, batched_dataset, chunksize=8):
            pass

if __name__ == '__main__':
    process_data()
