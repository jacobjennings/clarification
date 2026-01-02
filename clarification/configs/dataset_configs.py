from clarification.datas import *

from typing import Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)
import torch

from clarification.util import *

@dataclass(kw_only=True)
class DatasetConfig:
    """Configuration for the dataset.

    Args:
        sample_rate: Sample rate of the audio.
        sample_batch_ms: Length of each audio batch in milliseconds.
        dataset_batch_size: Batch size for the dataset. This is the number of batches each worker loads.
        batches_per_iteration: Number of batches to process in each iteration.
    
    Note: overlap is read from the dataset's info.csv file via the loader.
    """
    sample_rate: int
    sample_batch_ms: int
    dataset_batch_size: int
    batches_per_iteration: int

    def __post_init__(self):
        self.samples_per_batch = int((self.sample_batch_ms / 1000) * self.sample_rate)
        self.samples_per_iteration = self.samples_per_batch * self.batches_per_iteration


@dataclass(kw_only=True)
class PresetDatasetConfig1(DatasetConfig):
    sample_rate: int = field(default=24000)
    sample_batch_ms: int = field(default=300)
    dataset_batch_size: int = field(default=16)

class PresetCommonVoiceLoader(CommonVoiceLoader):
    def __init__(self, dataset_batch_size, batches_per_iteration, device: Optional[torch.device] = None, 
                 use_cpp_loader=False, dataset_path=None, use_lz4=False, num_preload_batches=16):
        if dataset_path:
            base_dir = dataset_path
        else:
            base_dir = dataset_dir()
            
        if batches_per_iteration % 16 != 0:
            print("batches_per_iteration must be divisible by 16 due to consumption_batch_size in prepare_dataset")

        if not device:
            device = str(torch.get_default_device())

        print(f"PresetCommonVoiceLoader device: {device}, use_cpp_loader: {use_cpp_loader}, "
              f"use_lz4: {use_lz4}, base_dir: {base_dir}, num_preload_batches: {num_preload_batches}")

        super().__init__(base_dir=base_dir,
                         dataset_batch_size=dataset_batch_size,
                         batches_per_iteration=batches_per_iteration,
                         should_pin_memory=True,
                         num_workers=8,
                         device=device,
                         use_cpp_loader=use_cpp_loader,
                         use_lz4=use_lz4,
                         num_preload_batches=num_preload_batches)
