from clarification.training import AudioTrainerConfig

from collections.abc import Sequence
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)
import torch

from ...models import *
from ...util import *
from .loss_configs import *
from .dataset_configs import *
from .validation_configs import *
from .mixed_precision_configs import *

@dataclass(kw_only=True)
class ModelTrainingConfig:
    """Configuration for model training.

    Args:
        name: Name of the model config. Use different names for different model configurations so files and logs do
            not conflict.
        model: The model to train.
        loss_function_configs: Sequence of AudioLossFunctionConfig. WARNING: Don't use duplicates. One instance per
            ModelTrainingConfig.
        optimizer: The optimizer to use.
        scheduler: The learning rate scheduler to use.
        batches_per_rotation: Number of batches to train during one call to train_one_rotation().
        dataset_config: DatasetConfig.
        dataset_loader: DataLoader for the input dataset.
        step_every_iterations: Step the scheduler every this number of iterations.
        validation_config: Optional validation configuration.
        training_classifier: If true, this is a classifier model. Data is expected to be a tuple of
            ([batches, channels, samples], [batches, labels]).
        norm_clip: If set, clip gradient normals to this value.
        mixed_precision_config: Configuration for mixed precision training.
    """
    name: str
    model: nn.Module
    loss_function_configs: Sequence[AudioLossFunctionConfig]
    optimizer: optim.Optimizer
    scheduler: Optional[optim.lr_scheduler.LRScheduler]
    dataset_config: DatasetConfig
    dataset_loader: DataLoader
    batches_per_rotation: int
    step_every_iterations: int = 1
    validation_config: Optional[ValidationConfig] = None
    training_classifier: bool = False
    norm_clip: float | None = 1.5
    mixed_precision_config: Optional[MixedPrecisionConfig] = None

    def __post__init__(self):
        self.dataset_batches_total_length = len(self.dataset_loader)

        pass


@dataclass(kw_only=True)
class PresetTrainingConfig1(ModelTrainingConfig):
    batches_per_iteration: int
    training_date_str: str

    writer: SummaryWriter = None
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    loss_function_configs: Optional[Sequence[AudioLossFunctionConfig]] = None
    optimizer: Optional[optim.Optimizer] = None
    dataset_config: Optional[DatasetConfig] = None
    dataset_loader: Optional[DataLoader] = None

    def __post__init__(self):
        super().__post__init__()
        if not self.writer:
            self.runs_dir = runs_dir(f"{self.training_date_str}-{self.name}")
            self.writer = SummaryWriter(log_dir=self.runs_dir)
        if not self.device:
            self.device = torch.get_default_device()
        if not self.dtype:
            self.dtype = torch.get_default_dtype()

        if not self.loss_function_configs:
            self.loss_function_configs = loss_group_1(self.dataset_config, self.device)

        if not self.optimizer:
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=0.01)

        if not self.dataset_config:
            self.dataset_config = PresetDatasetConfig1(batches_per_iteration=self.batches_per_iteration,
                                                       dataset_batch_size=16)

        if not self.dataset_loader:
            self.common_voice_loader = PresetCommonVoiceLoader(
                summary_writer=self.writer,
                dataset_batch_size=self.dataset_config.dataset_batch_size,
                batches_per_iteration=self.batches_per_iteration,
                device=self.device)
            self.common_voice_loader.create_loaders()
            self.dataset_loader = self.common_voice_loader.train_loader


@dataclass
class SimpleModelTrainingConfig(PresetTrainingConfig1):
    layer_sizes: Sequence[int]

    def __post__init__(self):
        self.model = ClarificationSimple(
                name=self.name,
                layer_sizes=self.layer_sizes, device=self.device, dtype=self.dtype)
        super().__post__init__()

@dataclass
class DenseTrainingConfig(PresetTrainingConfig1):
    layer_sizes: Sequence[int]

    def __post__init__(self):
        self.model = ClarificationSimple(
                name=self.name,
                layer_sizes=self.layer_sizes, device=self.device, dtype=self.dtype)
        super().__post__init__()

@dataclass
class ResnetTrainingConfig(PresetTrainingConfig1):
    channel_size: int
    layer_count: int

    def __post__init__(self):
        self.model = ClarificationResNet(
            name=self.name,
            channel_size=self.channel_size,
            layer_count=self.layer_count,
            device=self.device, dtype=self.dtype)
        super().__post__init__()