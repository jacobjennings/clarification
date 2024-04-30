import argparse
import copy
import os
import logging
import secrets
import numpy
import copy
import threading

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
from torio.io import CodecConfig

# Image datasets and image manipulation
import torchvision
import torchvision.transforms as transforms

# Image display
import matplotlib.pyplot as plt
import numpy as np

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

print(torch.__version__)
print(torchaudio.__version__)

base_dataset_directory = '/home/jacob/cv-corpus-17.0-2024-03-15/en'
# noisy_dataset_directory = '/home/jacob/noisy-commonvoice/en'

### Load base dataset

training_speech_dataset = torchaudio.datasets.COMMONVOICE(root=base_dataset_directory)

# torch.set_default_dtype()
torch.manual_seed(314)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


add_noise = AddGaussianNoise(std=0.5)

training_speech_dataset_noisy = copy.deepcopy(training_speech_dataset)

data_loader = DataLoader(
    training_speech_dataset_noisy,
    batch_size=1)

print(torchaudio.utils.ffmpeg_utils.get_audio_encoders())

print(torchaudio.list_audio_backends)

print(f"Detected {os.cpu_count()} cpus.")

data_loader_len = len(data_loader)

num_processed = Value("l", 0)

t0 = time.time()

process_count = os.cpu_count()

def noisify_file(data):
    if num_processed.value % 1000 == 0 and num_processed.value != 0:
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
    noisy_waveform = add_noise(data[0].squeeze())

    noisy_waveform = noisy_waveform.unsqueeze(1)

    src = noisy_waveform

    torchaudio.save(
        uri=f"{noisy_dataset_directory}/clips/{filename}",
        src=src,
        sample_rate=data[1][0].item(),
        format="mp3",
        channels_first=False
    )

    return 0


with Pool(processes=process_count) as pool:
    for i in pool.imap_unordered(noisify_file, data_loader, chunksize=os.cpu_count()):
        pass
