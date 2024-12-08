"""Utility for loading mozilla common voice noisy / clear datasets."""

import functools

import argparse
import copy
import os
import logging
import secrets
import numpy
import copy
import gc
import math
from datetime import timedelta
# import mplcursors

from ipywidgets import IntProgress
# from IPython.display import display
# from IPython.display import Audio
from IPython import display

import time

# PyTorch model and training necessities
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from complexPyTorch.complexFunctions import complex_relu

import auraloss

# Audio
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

from torio.io import CodecConfig

# Image datasets and image manipulation
import torchvision
import torchvision.transforms as transforms
import torchvision.models as TVM

# Image display
import matplotlib.pyplot as plt
import numpy as np

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter


class CommonVoiceLoader():
    """Utility for loading mozilla common voice noisy / clear datasets."""

    def __init__(self, base_dataset_directory, noisy_dataset_directory, summary_writer, device):
        self.base_dataset_directory = base_dataset_directory
        self.noisy_dataset_directory = noisy_dataset_directory
        self.summary_writer = summary_writer
        self.device = device


    def create_loaders(self):
        loader_batch_size = 1
        
        common_voice_dataset = torchaudio.datasets.COMMONVOICE(root=self.base_dataset_directory)
        common_voice_noisy_dataset = torchaudio.datasets.COMMONVOICE(root=self.noisy_dataset_directory)

        split_generator_0 = torch.Generator().manual_seed(314)
        noisy_train, noisy_test = random_split(common_voice_dataset, [0.9, 0.1], generator=split_generator_0)

        split_generator_1 = torch.Generator().manual_seed(314)
        clear_train, clear_test = random_split(common_voice_noisy_dataset, [0.9, 0.1], generator=split_generator_1)

        clear_generator = torch.Generator()
        self.summary_writer.add_text("CommonVoiceLoader", f"Generator seed: {clear_generator.initial_seed()}")

        clear_loader = DataLoader(
            noisy_train,
            batch_size=loader_batch_size,
            pin_memory=True,
            pin_memory_device=self.device,
            generator=clear_generator,
        )

        noisy_generator = torch.Generator()
        noisy_generator.manual_seed(clear_generator.initial_seed())

        noisy_loader = DataLoader(
            clear_train,
            batch_size=loader_batch_size,
            pin_memory=True,
            pin_memory_device=self.device,
            generator=noisy_generator,
        )
        
        self.noisy_loader = noisy_loader
        self.clear_loader = clear_loader
        self.noisy_test = noisy_test
        self.clear_test = clear_test
