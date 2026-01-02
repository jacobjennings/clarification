from collections.abc import Sequence, Iterable
from typing import Optional
from dataclasses import dataclass, field
import logging
import pprint
import json
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .model_training_configs import *
from .loss_configs import *
from .log_configs import *

@dataclass(kw_only=True)
class AudioTrainerState:
    """State required to recreate the trainer and continue training if desired. Will be changed during training.

    Fields are not intended to be set manually, but are updated during training by the AudioTrainer.
    """
    batches_trained: int = 0
    batches_validated: int = 0
    batches_since_last_save: int = 0
    batches_since_last_send_audio: int = 0
    rotation_count: int = 0
    epoch_count: int = 0
    iteration_count: int = 0
    samples_processed: int = 0
    last_samples_processed_log_time: Optional[float] = None
    samples_processed_since_last_log: int = 0
    train_start_time: Optional[float] = None
    epoch_start_time: Optional[float] = None
    batches_since_last_validation: int = 0
    batches_since_last_log: int = 0
    batches_since_last_profile: int = 0
    last_matmul_value: str = "highest"
    data_loader_iter: Optional[Iterable[DataLoader]] = None
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    # File index for resuming with CppDataLoader (which doesn't support __len__)
    data_loader_file_idx: int = 0

    def __post_init__(self):
        pass


@dataclass(kw_only=True)
class AudioTrainerConfig:
    """Configuration for the audio trainer.

    Args:
        model_training_config: ModelTrainingConfig.
        log_behavior_config: LogBehaviorConfig.
        device: Device to train on.
        training_date_str: String representing the date of training. Useful to keep a group of runs together with
          matching dates.
        state: AudioTrainerState.
    """
    model_training_config: ModelTrainingConfig
    log_behavior_config: LogBehaviorConfig
    training_date_str: str
    state: AudioTrainerState = field(default_factory=AudioTrainerState)
    device: torch.device = None

    def __post_init__(self):
        if not self.device:
            self.device = torch.get_default_device()
        pass


    def __hash__(self):
        return hash(self.model_training_config.name) + hash(self.training_date_str)

@dataclass(kw_only=True)
class TrainMultipleConfig:
    trainer_configs: Sequence[AudioTrainerConfig]
    should_perform_memory_test: bool = False
    resume_from_run_dir: Optional[str] = None
    auto_resume: bool = True
    # If True, all models train on the SAME data each round (fair architecture comparison)
    # This requires re-reading data from disk for each model after the first.
    # If False, models train on different sequential data (faster, but unfair comparison)
    fair_comparison_mode: bool = True

    def __post_init__(self):
        pass
