
from collections.abc import Sequence, Iterable
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from clarification.util import *


@dataclass(kw_only=True)
class LogBehaviorConfig:
    """Configuration for logging behavior.

    Args:
        writer: SummaryWriter for TensorBoard logging.
        log_info_every_batches: Log information every this number of batches.
        model_weights_dir: Directory to save model weights.
        model_weights_save_every_batches: Save model weights every this number of batches.
        send_audio_clip_every_batches: Send audio clips every this number of batches.
        profiling_data_output_dir: Directory to save profiling data. Must be set if profile_every_batches is not None.
        profile_every_batches: Profile every this number of batches. If None, do not profile
    """
    writer: SummaryWriter
    log_info_every_batches: int = 15000

    model_weights_dir: Optional[str] = None
    model_weights_save_every_batches: int = 100000
    send_audio_clip_every_batches: int = 45000
    profiling_data_output_dir: Optional[str] = None
    profile_every_batches: int | None = 50000

    def __post_init__(self):
        pass


@dataclass(kw_only=True)
class PresetLogBehaviorConfig1(LogBehaviorConfig):
    runs_subdir_name: str
    writer: Optional[SummaryWriter] = None

    def __post_init__(self):
        super().__post_init__()
        runs_subdir_full = runs_dir() + "/" + self.runs_subdir_name
        self.profiling_data_output_dir = profiling_data_dir(a_runs_dir=runs_subdir_full)
        self.model_weights_dir = models_dir(a_runs_dir=runs_subdir_full)
        if not self.writer:
            self.writer = SummaryWriter(log_dir=runs_subdir_full)
