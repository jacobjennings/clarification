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

out_dataset_directory = '/home/jacob/noisy-commonvoice-24k-300ms-10ms/en'

num_processed = Value("l", 0)

sample_ms = 300
overlap_ms = 10

resample_rate = 24000
sample_size = int((sample_ms / 1000) * resample_rate)
overlap_size = int((overlap_ms / 1000) * resample_rate)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def overlapping_samples(audio, sample_size, overlap_size):
    chunks = audio.unfold(dimension=0, size=sample_size, step=sample_size - overlap_size)
    return chunks


def process_file(data):
    # print("1")
    add_noise = AddGaussianNoise(std=0.1)

    # print("2")

    # if num_processed.value >= 5000:
    #     print("DONE")
    #     return

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

    # print(f"Data[2]: {data[2]}")

    filename = data[2]["path"][0]
    original_waveform = data[0].squeeze()

    # print("3")

    # print(f"data: {data}")

    sample_rate = data[1][0].item()
    # print("4")

    resampler = TAT.Resample(sample_rate, resample_rate, dtype=torch.float32)
    # print("5")

    # print(f"original_waveform: {original_waveform.size()}")

    resampled_clear = resampler(original_waveform)

    # print("6")

    # print(f"resampled_clear: {resampled_clear.size()}")

    noisy_waveform = add_noise(resampled_clear)

    # print(f"noisy_waveform: {noisy_waveform.size()}")

    resampled_clear_chunks = overlapping_samples(resampled_clear, sample_size=sample_size, overlap_size=overlap_size)

    noisy_chunks = overlapping_samples(noisy_waveform, sample_size=sample_size, overlap_size=overlap_size)

    with open(f"{out_dataset_directory}/info.csv", 'a', newline='\n') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        filename_no_ext = filename.removesuffix(".mp3")

        noisy_path = f"{filename_no_ext}_noisy"
        clear_path = f"{filename_no_ext}_clear"

        writer.writerow({
            "noisy_path": noisy_path,
            "clear_path": clear_path,
            "chunk_count": len(noisy_chunks),
            "sentence_id": data[2]["sentence_id"]
        })

        for idx, (noisy_chunk, clear_chunk) in enumerate(zip(noisy_chunks, resampled_clear_chunks)):
            noisy_path_absolute = f"{out_dataset_directory}/clips/{filename_no_ext}_noisy_{idx}.mp3"
            clear_path_absolute = f"{out_dataset_directory}/clips/{filename_no_ext}_clear_{idx}.mp3"

            torchaudio.save(
                uri=noisy_path_absolute,
                src=noisy_chunk.unsqueeze(1),
                sample_rate=resample_rate,
                format="mp3",
                channels_first=False
            )

            torchaudio.save(
                uri=clear_path_absolute,
                src=clear_chunk.unsqueeze(1),
                sample_rate=resample_rate,
                format="mp3",
                channels_first=False
            )

    # print("3")

    return filename


if __name__ == '__main__':
    print(torch.__version__)
    print(torchaudio.__version__)

    training_speech_dataset = torchaudio.datasets.COMMONVOICE(root=base_dataset_directory)

    training_speech_dataset_noisy = copy.deepcopy(training_speech_dataset)

    data_loader = DataLoader(
        training_speech_dataset_noisy,
        batch_size=1)

    print(torchaudio.utils.ffmpeg_utils.get_audio_encoders())

    print(torchaudio.list_audio_backends)

    print(f"Detected {os.cpu_count()} cpus.")

    data_loader_len = len(data_loader)

    with open(f"{out_dataset_directory}/info.csv", 'w', newline='\n') as csvfile:
        fieldnames = ['noisy_path', 'clear_path', 'chunk_count', 'sentence_id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # writer.writeheader()

    t0 = time.time()

    process_count = 4

    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=process_count,
        mp_context=multiprocessing.get_context("fork")
    )

    # for thing in data_loader:
    #     noisify_file(thing)
    #
    # while True:
    #     chunk = list(itertools.islice(data_loader, process_count))
    #     for result in executor.map(noisify_file, chunk):
    #         print(result)

    with Pool(processes=process_count) as pool:
        for i in pool.imap_unordered(process_file, data_loader, chunksize=process_count):
            pass
