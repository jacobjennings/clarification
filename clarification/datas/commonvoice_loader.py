"""Utility for loading mozilla common voice noisy / clear datasets."""

import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from . import noisy_dataset


class CommonVoiceLoader():
    """Utility for loading mozilla common voice noisy / clear datasets."""

    def __init__(self,
                 base_dir,
                 summary_writer,
                 should_pin_memory,
                 device):
        self.base_dir = base_dir
        self.summary_writer = summary_writer
        self.should_pin_memory = should_pin_memory
        self.device = device

        self.train_loader = None
        self.test_loader = None

    def create_loaders(self):
        """Creates loaders"""
        loader_batch_size = 1

        dataset = noisy_dataset.NoisyCommonsDataset(base_dir=self.base_dir)

        split_generator = torch.Generator()
        train, test = random_split(dataset, [0.9, 0.1], generator=split_generator)

        loader_generator = torch.Generator()
        self.summary_writer.add_text("CommonVoiceLoader", f"split_generator seed: {split_generator.initial_seed()}, "
                                                          f"loader_generator seed: {loader_generator.initial_seed()}")

        train_loader = DataLoader(
            train,
            batch_size=loader_batch_size,
            pin_memory=self.should_pin_memory,
            pin_memory_device=self.device if self.should_pin_memory else "",
            generator=loader_generator,
        )

        test_loader = DataLoader(
            test,
            batch_size=loader_batch_size,
            pin_memory=self.should_pin_memory,
            pin_memory_device=self.device if self.should_pin_memory else "",
        )

        self.train_loader = train_loader
        self.test_loader = test_loader
