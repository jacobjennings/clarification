from collections.abc import Sequence
from dataclasses import dataclass
import logging
import pprint
import json

import clarification.schedulers

logger = logging.getLogger(__name__)
import torch

from clarification.models import *
from clarification.util import *
from .loss_configs import *
from .dataset_configs import *
from .validation_configs import *
from .mixed_precision_configs import *
from clarification.schedulers import *

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
        batches_per_rotation: Number of batches to train before switching models.
        should_use_dataparallel: If true, use DataParallel to train the model. This overrides model_wrapper.
        model_wrapper: Optional model wrapper. If set, this will be used for training (aka DataParallel).
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
    should_use_dataparallel: bool = True
    model_wrapper: Optional[nn.Module] = None
    step_every_iterations: int = 1
    validation_config: Optional[ValidationConfig] = None
    training_classifier: bool = False
    norm_clip: float | None = 1
    mixed_precision_config: Optional[MixedPrecisionConfig] = None

    def __post_init__(self):
        self.dataset_batches_total_length = len(self.dataset_loader)
        if not self.mixed_precision_config:
            self.mixed_precision_config = MixedPrecisionConfig()

        if self.should_use_dataparallel:
            self.model_wrapper = nn.DataParallel(self.model)

        if not self.model_wrapper:
            self.model_wrapper = self.model
        pass

    def prettyprint(self):
          return json.dumps(self.__dict__, indent=4)

    def training_model(self) -> nn.Module:
        return self.model_wrapper or self.model

@dataclass(kw_only=True)
class PresetTrainingConfig1(ModelTrainingConfig):
    batches_per_iteration: int
    training_date_str: str

    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    loss_function_configs: Optional[Sequence[AudioLossFunctionConfig]] = None
    optimizer: Optional[optim.Optimizer] = None
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
    dataset_config: Optional[DatasetConfig] = None
    dataset_loader: Optional[DataLoader] = None

    def __post_init__(self):
        super().__post_init__()
        if not self.device:
            self.device = torch.get_default_device()
        if not self.dtype:
            self.dtype = torch.get_default_dtype()

        if not self.loss_function_configs:
            self.loss_function_configs = loss_group_1(self.dataset_config, self.device)

        if not self.optimizer:
            self.optimizer = torch.optim.SGD(params=self.training_model().parameters(), lr=0.01)

        if not self.scheduler:
            self.scheduler = clarification.schedulers.InterpolatingLR(
                optimizer=self.optimizer,
                milestones=[(0, 0.01), (1000000, 0.0001)])

        if not self.dataset_config:
            self.dataset_config = PresetDatasetConfig1(batches_per_iteration=self.batches_per_iteration,
                                                       dataset_batch_size=16)


@dataclass(kw_only=True)
class SimpleTrainingConfig(PresetTrainingConfig1):
    layer_sizes: Sequence[int]
    model: Optional[nn.Module] = None

    def __post_init__(self):
        self.model = ClarificationSimple(
                name=self.name,
                layer_sizes=self.layer_sizes, device=self.device, dtype=self.dtype)
        super().__post_init__()

@dataclass(kw_only=True)
class DenseTrainingConfig(PresetTrainingConfig1):
    layer_sizes: Sequence[int]
    model: Optional[nn.Module] = None

    def __post_init__(self):
        self.model = ClarificationDense(
                name=self.name,
                layer_sizes=self.layer_sizes, device=self.device, dtype=self.dtype)
        super().__post_init__()

@dataclass(kw_only=True)
class ResnetTrainingConfig(PresetTrainingConfig1):
    channel_size: int
    layer_count: int
    model: Optional[nn.Module] = None

    def __post_init__(self):
        self.model = ClarificationResNet(
            name=self.name,
            channel_size=self.channel_size,
            layer_count=self.layer_count,
            device=self.device, dtype=self.dtype)
        super().__post_init__()
