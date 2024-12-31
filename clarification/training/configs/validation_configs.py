
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

@dataclass(kw_only=True)
class ValidationConfig:
    """Configuration for validation.

    Args:
        test_loader: DataLoader for the test set.
        test_batches: Number of batches to validate during each validation run.
        run_validation_every_batches: Number of training batches between validations.
        log_every_batches: Number of batches between log events during validation.
    """
    test_loader: DataLoader
    test_batches: int
    run_validation_every_batches: int
    log_every_batches: int = 5000

    def __post__init__(self):
        pass

@dataclass
class PresetValidationConfig1(ValidationConfig):
    # test_loader: DataLoader
    test_batches: int = 20000
    run_validation_every_batches: int = 100000
    log_every_batches: int = 100000

