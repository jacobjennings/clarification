"""Loss configuration presets."""

from collections.abc import Sequence, Callable
from dataclasses import dataclass, field
import logging
import math

logger = logging.getLogger(__name__)
import auraloss
import torch.nn as nn

from .dataset_configs import *


# ============================================================================
# Weight Schedule Functions
# ============================================================================
# These functions take (step, **kwargs) and return a weight multiplier [0, 1]

def constant_weight(step: int, **kwargs) -> float:
    """Constant weight of 1.0 (default behavior)."""
    return 1.0

def linear_warmup(step: int, warmup_steps: int = 10000, **kwargs) -> float:
    """Linear warmup from 0 to 1 over warmup_steps."""
    return min(1.0, step / warmup_steps)

def linear_decay(step: int, decay_start: int = 0, decay_end: int = 10000, **kwargs) -> float:
    """Linear decay from 1 to 0 between decay_start and decay_end."""
    if step <= decay_start:
        return 1.0
    if step >= decay_end:
        return 0.0
    return 1.0 - (step - decay_start) / (decay_end - decay_start)

def linear_ramp(step: int, start_step: int = 0, end_step: int = 10000, 
                start_weight: float = 0.0, end_weight: float = 1.0, **kwargs) -> float:
    """Linear interpolation from start_weight to end_weight between start_step and end_step."""
    if step <= start_step:
        return start_weight
    if step >= end_step:
        return end_weight
    progress = (step - start_step) / (end_step - start_step)
    return start_weight + progress * (end_weight - start_weight)

def cosine_warmup(step: int, warmup_steps: int = 10000, **kwargs) -> float:
    """Cosine warmup from 0 to 1 (smoother than linear)."""
    if step >= warmup_steps:
        return 1.0
    return 0.5 * (1 - math.cos(math.pi * step / warmup_steps))

def cosine_decay(step: int, decay_start: int = 0, decay_end: int = 10000, **kwargs) -> float:
    """Cosine decay from 1 to 0 (smoother than linear)."""
    if step <= decay_start:
        return 1.0
    if step >= decay_end:
        return 0.0
    progress = (step - decay_start) / (decay_end - decay_start)
    return 0.5 * (1 + math.cos(math.pi * progress))

def step_function(step: int, switch_step: int = 5000, before: float = 1.0, after: float = 0.0, **kwargs) -> float:
    """Step function: returns 'before' until switch_step, then 'after'."""
    return before if step < switch_step else after


def make_schedule(fn: Callable, **fixed_kwargs) -> Callable[[int], float]:
    """Create a schedule function with fixed kwargs.
    
    Example:
        schedule = make_schedule(linear_decay, decay_start=1000, decay_end=5000)
        weight = schedule(step=2500)  # Returns 0.5
    """
    def scheduled_fn(step: int) -> float:
        return fn(step, **fixed_kwargs)
    return scheduled_fn


@dataclass(kw_only=True)
class AudioLossFunctionConfig:
    """Configuration for an audio loss function.

    Args:
        name: Name of the loss function.
        weight: Base weight of the loss function.
        weight_fn: Optional function(step) -> multiplier for dynamic weighting.
                   Final weight = weight * weight_fn(step)
        fn: The loss function.
        is_unary: If true, the loss function is unary (classifier) and does not take golden values.
        batch_size: Batch size for the loss function (optional). If None, the batch size is batches_per_iteration.
    """
    name: str
    fn: nn.Module
    weight: float = 1.0
    weight_fn: Optional[Callable[[int], float]] = None
    is_unary: bool = False
    batch_size: Optional[int] = None

    def get_weight(self, step: int) -> float:
        """Get the effective weight at a given step."""
        if self.weight_fn is None:
            return self.weight
        return self.weight * self.weight_fn(step)

    def __post_init__(self):
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


def loss_group_2(dataset_config: DatasetConfig,
                 device: Optional[torch.device] = None) -> Sequence[AudioLossFunctionConfig]:
    if not device:
        device = torch.get_default_device()
    return [
        AudioLossFunctionConfig(
            name="SISDRLoss", weight=0.5, fn=auraloss.time.SISDRLoss().to(device), is_unary=False, batch_size=None),
        AudioLossFunctionConfig(
            name="MelSTFTLoss", weight=0.5,
            fn=auraloss.freq.MelSTFTLoss(sample_rate=dataset_config.sample_rate, n_mels=128, device=device).to(device),
            is_unary=False, batch_size=None),
    ]


def loss_group_scheduled(dataset_config: DatasetConfig,
                         device: Optional[torch.device] = None,
                         transition_steps: int = 500000) -> Sequence[AudioLossFunctionConfig]:
    """
    Scheduled loss group: L1 dominates early training, then fades out
    while perceptual losses (SI-SDR, Mel-STFT) take over.
    
    Args:
        dataset_config: Dataset configuration
        device: Target device
        transition_steps: Number of steps for the transition (default 50k)
    
    Weight schedule:
        - L1Loss: starts at 2.0, decays to 0.2 over transition_steps
        - SISDRLoss: starts at 0.2, ramps to 1.5 over transition_steps  
        - MelSTFTLoss: starts at 0.1, ramps to 0.5 over transition_steps
    """
    if not device:
        device = torch.get_default_device()
    
    return [
        # L1 Loss: Strong early, fades to small contribution
        AudioLossFunctionConfig(
            name="L1Loss",
            weight=2.0,
            weight_fn=make_schedule(linear_ramp, 
                                    start_step=0, 
                                    end_step=transition_steps,
                                    start_weight=1.0,  # 2.0 * 1.0 = 2.0 initially
                                    end_weight=0.1),   # 2.0 * 0.1 = 0.2 finally
            fn=nn.L1Loss().to(device),
            is_unary=False,
            batch_size=None
        ),
        # SI-SDR: Weak early, becomes dominant
        AudioLossFunctionConfig(
            name="SISDRLoss",
            weight=1.5,
            weight_fn=make_schedule(linear_ramp,
                                    start_step=0,
                                    end_step=transition_steps,
                                    start_weight=0.13,  # 1.5 * 0.13 ≈ 0.2 initially
                                    end_weight=1.0),    # 1.5 * 1.0 = 1.5 finally
            fn=auraloss.time.SISDRLoss().to(device),
            is_unary=False,
            batch_size=None
        ),
        # Mel-STFT: Weak early, ramps up
        AudioLossFunctionConfig(
            name="MelSTFTLoss",
            weight=0.5,
            weight_fn=make_schedule(linear_ramp,
                                    start_step=0,
                                    end_step=transition_steps,
                                    start_weight=0.2,   # 0.5 * 0.2 = 0.1 initially
                                    end_weight=1.0),    # 0.5 * 1.0 = 0.5 finally
            fn=auraloss.freq.MelSTFTLoss(
                sample_rate=dataset_config.sample_rate, 
                n_mels=128, 
                device=device
            ).to(device),
            is_unary=False,
            batch_size=None
        ),
    ]


def loss_group_three_phase(dataset_config: DatasetConfig,
                           device: Optional[torch.device] = None,
                           phase1_end: int = 500000,
                           phase2_end: int = 1000000) -> Sequence[AudioLossFunctionConfig]:
    """
    Three-phase loss schedule:
    
    Phase 1 (0 to phase1_end): L1 dominates
        - L1Loss: 1.0
        - SISDRLoss: 0.0
        - MelSTFTLoss: 0.0
    
    Phase 2 (phase1_end to phase2_end): Mix of SI-SDR and Mel-STFT
        - L1Loss: decays 1.0 → 0.0
        - SISDRLoss: ramps 0.0 → 0.5 → 0.0 (peaks in middle, then fades)
        - MelSTFTLoss: ramps 0.0 → 1.0
    
    Phase 3 (after phase2_end): Only MelSTFTLoss
        - L1Loss: 0.0
        - SISDRLoss: 0.0
        - MelSTFTLoss: 1.0
    """
    if not device:
        device = torch.get_default_device()
    
    def l1_schedule(step: int) -> float:
        """L1: 1.0 during phase 1, decays to 0 during phase 2, stays 0 in phase 3"""
        if step <= phase1_end:
            return 1.0
        elif step <= phase2_end:
            # Linear decay from 1.0 to 0.0
            progress = (step - phase1_end) / (phase2_end - phase1_end)
            return 1.0 - progress
        else:
            return 0.0
    
    def sisdr_schedule(step: int) -> float:
        """SI-SDR: 0 in phase 1, ramps up then down in phase 2, 0 in phase 3"""
        if step <= phase1_end:
            return 0.0
        elif step <= phase2_end:
            # Ramp up to peak at middle of phase 2, then decay
            progress = (step - phase1_end) / (phase2_end - phase1_end)
            # Triangle: peaks at 0.5 progress
            if progress <= 0.5:
                return progress * 2  # 0 → 1
            else:
                return (1 - progress) * 2  # 1 → 0
        else:
            return 0.0
    
    def melstft_schedule(step: int) -> float:
        """MelSTFT: 0 in phase 1, ramps up in phase 2, 1.0 in phase 3"""
        if step <= phase1_end:
            return 0.0
        elif step <= phase2_end:
            # Linear ramp from 0 to 1
            progress = (step - phase1_end) / (phase2_end - phase1_end)
            return progress
        else:
            return 1.0
    
    return [
        AudioLossFunctionConfig(
            name="L1Loss",
            weight=1.0,
            weight_fn=l1_schedule,
            fn=nn.L1Loss().to(device),
            is_unary=False,
        ),
        AudioLossFunctionConfig(
            name="SISDRLoss",
            weight=1.0,
            weight_fn=sisdr_schedule,
            fn=auraloss.time.SISDRLoss().to(device),
            is_unary=False,
        ),
        AudioLossFunctionConfig(
            name="MelSTFTLoss",
            weight=1.0,
            weight_fn=melstft_schedule,
            fn=auraloss.freq.MelSTFTLoss(
                sample_rate=dataset_config.sample_rate,
                n_mels=128,
                device=device
            ).to(device),
            is_unary=False,
        ),
    ]


def loss_group_l1_to_perceptual(dataset_config: DatasetConfig,
                                device: Optional[torch.device] = None,
                                l1_decay_start: int = 10000,
                                l1_decay_end: int = 50000,
                                perceptual_warmup: int = 30000) -> Sequence[AudioLossFunctionConfig]:
    """
    Two-phase training:
    1. Phase 1 (0 to l1_decay_start): Pure L1 for basic structure
    2. Phase 2 (l1_decay_start to l1_decay_end): L1 fades, perceptual ramps up
    3. Phase 3 (after l1_decay_end): Perceptual-only for fine details
    """
    if not device:
        device = torch.get_default_device()
    
    return [
        AudioLossFunctionConfig(
            name="L1Loss",
            weight=1.0,
            weight_fn=make_schedule(linear_decay,
                                    decay_start=l1_decay_start,
                                    decay_end=l1_decay_end),
            fn=nn.L1Loss().to(device),
            is_unary=False,
        ),
        AudioLossFunctionConfig(
            name="SISDRLoss",
            weight=1.0,
            weight_fn=make_schedule(cosine_warmup, warmup_steps=perceptual_warmup),
            fn=auraloss.time.SISDRLoss().to(device),
            is_unary=False,
        ),
        AudioLossFunctionConfig(
            name="MelSTFTLoss",
            weight=0.3,
            weight_fn=make_schedule(cosine_warmup, warmup_steps=perceptual_warmup),
            fn=auraloss.freq.MelSTFTLoss(
                sample_rate=dataset_config.sample_rate,
                n_mels=128,
                device=device
            ).to(device),
            is_unary=False,
        ),
    ]

    # def dd_encoder_maker(name, scalar, layer_sizes):
    #     dd_model = clarification.loss.DistortionDetectorDenseEncoder(
    #         in_channels=1, samples_per_batch=samples_per_batch * dataset_batch_size,
    #         layer_sizes=layer_sizes, device=device, dtype=dtype)

    #     return name, scalar, dd_model, True, samples_per_batch * dataset_batch_size

# def loss_group_2(dataset_config: DatasetConfig,
#                  device: Optional[torch.device] = None) -> Sequence[AudioLossFunctionConfig]:
#     if not device:
#         device = torch.get_default_device()

