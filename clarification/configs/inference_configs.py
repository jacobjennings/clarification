from dataclasses import dataclass
from dataclasses import field
from typing import Callable, Tuple, Any

import torch
import torch.nn as nn

from ..util import *
from .dataset_configs import *

@dataclass
class InferenceBenchmarkConfig:
    model_name: str
    model_function: Callable[..., nn.Module]
    model_args: Tuple[Any, ...]
    dataset_config: DatasetConfig
    # model: nn.Module = None # Don't populate. It will be created with model_function so we can memory benchmark.
    dataset_loader: DataLoader = None
    model_weights_path: Optional[str] = None
    device: torch.device = field(default=torch.get_default_device())
    batch_size: int = 1
    num_test_batches: int = 1000
    verbose: bool = False
