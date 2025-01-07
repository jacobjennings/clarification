from dataclasses import dataclass
from dataclasses import field

import torch
import torch.nn as nn

from ..util import *
from .dataset_configs import *

@dataclass
class InferenceBenchmarkConfig:
    model_name: str
    model: nn.Module
    dataset_config: DatasetConfig
    dataset_loader: DataLoader = None
    model_weights_path: Optional[str] = None
    device: torch.device = field(default=torch.get_default_device())
    batch_size: int = 1
    num_test_batches: int = 1000
    verbose: bool = False
