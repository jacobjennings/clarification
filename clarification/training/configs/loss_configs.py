"""Loss configuration presets."""

from collections.abc import Sequence
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)
import auraloss
import torch.nn as nn

from .dataset_configs import *


@dataclass(kw_only=True)
class AudioLossFunctionConfig:
    """Configuration for an audio loss function.

    Args:
        name: Name of the loss function.
        weight: Weight of the loss function.
        fn: The loss function.
        is_unary: If true, the loss function is unary (classifier) and does not take golden values.
        batch_size: Batch size for the loss function (optional). If None, the batch size is batches_per_iteration.
    """
    name: str
    fn: nn.Module
    weight: float = 1.0
    is_unary: bool = False
    batch_size: Optional[int] = None

    def __post__init__(self):
        pass

def loss_group_1(dataset_config: DatasetConfig,
                 device: Optional[torch.device] = None) -> Sequence[AudioLossFunctionConfig]:
    if not device:
        device = torch.get_default_device()
    return [
        # Main group
        AudioLossFunctionConfig(
            name="L1Loss", weight=2.0, fn=nn.L1Loss().to(device), is_unary=False, batch_size=None),
        AudioLossFunctionConfig(
            name="SISDRLoss", weight=1.5, fn=auraloss.time.SISDRLoss().to(device), is_unary=False, batch_size=None),
        AudioLossFunctionConfig(
            name="MelSTFTLoss", weight=0.5,
            fn=auraloss.freq.MelSTFTLoss(sample_rate=dataset_config.sample_rate, n_mels=128, device=device).to(device),
            is_unary=False, batch_size=None),
    ]
