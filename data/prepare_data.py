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
# resampled_clear_dataset_directory = '/home/jacob/noisy-commonvoice-48k/en/clear'
# noisy_dataset_directory = '/home/jacob/noisy-commonvoice-48k/en/noisy'

num_processed = Value("l", 0)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def noisify_file(data):
    # print("1")
    add_noise = AddGaussianNoise(std=0.1)

    # print("2")

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

    noisy_waveform = noisy_waveform.unsqueeze(1)

    # print(f"noisy_waveform: {noisy_waveform.size()}")

    torchaudio.save(
        uri=f"{resampled_clear_dataset_directory}/clips/{filename}",
        src=resampled_clear.unsqueeze(1),
        sample_rate=resample_rate,
        format="mp3",
        channels_first=False
    )

    torchaudio.save(
        uri=f"{noisy_dataset_directory}/clips/{filename}",
        src=noisy_waveform,
        sample_rate=resample_rate,
        format="mp3",
        channels_first=False
    )

    # print("3")

    return filename


if __name__ == '__main__':
    print(torch.__version__)
    print(torchaudio.__version__)

    resample_rate = 24000

    training_speech_dataset = torchaudio.datasets.COMMONVOICE(root=base_dataset_directory)

    training_speech_dataset_noisy = copy.deepcopy(training_speech_dataset)

    data_loader = DataLoader(
        training_speech_dataset_noisy,
        batch_size=1)

    print(torchaudio.utils.ffmpeg_utils.get_audio_encoders())

    print(torchaudio.list_audio_backends)

    print(f"Detected {os.cpu_count()} cpus.")

    data_loader_len = len(data_loader)

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
        for i in pool.imap_unordered(noisify_file, data_loader, chunksize=process_count):
            pass


