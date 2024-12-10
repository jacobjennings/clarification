"""Utility for loading mozilla common voice noisy / clear datasets."""

import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

class CommonVoiceLoader():
    """Utility for loading mozilla common voice noisy / clear datasets."""

    def __init__(self, 
                 base_dataset_directory, 
                 noisy_dataset_directory, 
                 summary_writer, 
                 should_pin_memory,
                 device):
        self.base_dataset_directory = base_dataset_directory
        self.noisy_dataset_directory = noisy_dataset_directory
        self.summary_writer = summary_writer
        self.should_pin_memory = should_pin_memory
        self.device = device

        self.noisy_loader = None
        self.clear_loader = None
        self.noisy_test = None
        self.clear_test = None


    def create_loaders(self):
        """Creates loaders"""
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
            pin_memory=self.should_pin_memory,
            pin_memory_device=self.device if self.should_pin_memory else "",
            generator=clear_generator,
        )

        noisy_generator = torch.Generator()
        noisy_generator.manual_seed(clear_generator.initial_seed())

        noisy_loader = DataLoader(
            clear_train,
            batch_size=loader_batch_size,
            pin_memory=self.should_pin_memory,
            pin_memory_device=self.device if self.should_pin_memory else "",
            generator=noisy_generator,
        )
        
        self.noisy_loader = noisy_loader
        self.clear_loader = clear_loader
        self.noisy_test = noisy_test
        self.clear_test = clear_test
