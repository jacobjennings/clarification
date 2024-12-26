"""Utility for loading mozilla common voice noisy / clear datasets."""

import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from . import noisy_dataset


class CommonVoiceLoader:
    """Utility for loading mozilla common voice noisy / clear datasets."""

    def __init__(self,
                 base_dir,
                 summary_writer,
                 dataset_batch_size,
                 batches_per_iteration,
                 should_pin_memory,
                 num_workers,
                 device):
        self.base_dir = base_dir
        self.summary_writer = summary_writer
        self.dataset_batch_size = dataset_batch_size
        self.loader_batch_size = batches_per_iteration // dataset_batch_size
        self.should_pin_memory = should_pin_memory
        self.num_workers = num_workers
        self.device = device
        self.train_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.test_loader = None

    def create_loaders(self):
        """Creates loaders"""
        self.train_dataset = noisy_dataset.NoisyCommonsDataset(
            batch_size=self.dataset_batch_size, 
            device=self.device, 
            base_dir=self.base_dir,
            csv_filename="train.csv")
        
        self.test_dataset = noisy_dataset.NoisyCommonsDataset(
            batch_size=self.dataset_batch_size, 
            device=self.device, 
            base_dir=self.base_dir,
            csv_filename="test.csv")

        loader_generator = torch.Generator()

        if self.summary_writer:
            self.summary_writer.add_text("CommonVoiceLoader",
                                         f"loader_generator seed: {loader_generator.initial_seed()}")

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.loader_batch_size,
            generator=loader_generator,
            num_workers=self.num_workers
        )

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.loader_batch_size,
            num_workers=self.num_workers
        )

        self.train_loader = train_loader
        self.test_loader = test_loader
