import torch
from torch.optim.lr_scheduler import LRScheduler
from bisect import bisect_right


class InterpolatingLR(LRScheduler):
    """
    Linearly interpolates learning rate based on a list of (step, target) tuples.

    The learning rate will change linearly between the target values specified at
    the given steps.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list of (int, float) tuples): List of (step, target_lr) pairs,
            sorted by step in ascending order.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, milestones, last_epoch=-1):
        self.milestones = milestones
        self.milestone_steps, self.milestone_lrs = zip(
            *milestones
        )  # Unzip for easier access

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Calculates the current learning rate for each parameter group.
        """
        if not self._get_lr_called_within_step:
            print(
                "Warning: To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        if self.last_epoch == 0:
            return self.base_lrs

        # Find the interval where the current step falls
        idx = bisect_right(self.milestone_steps, self.last_epoch)

        if idx == 0:
            # Before the first milestone
            return self.base_lrs
        elif idx == len(self.milestone_steps):
            # After the last milestone
            return [group["lr"] for group in self.optimizer.param_groups]
        else:
            # Between two milestones, interpolate linearly
            prev_step = self.milestone_steps[idx - 1]
            next_step = self.milestone_steps[idx]
            prev_lr = self.milestone_lrs[idx - 1]
            next_lr = self.milestone_lrs[idx]

            fraction = (self.last_epoch - prev_step) / (next_step - prev_step)

            lrs = []
            for group in self.optimizer.param_groups:
                interpolated_lr = prev_lr + fraction * (next_lr - prev_lr)
                # Ensure LR doesn't go below 0 and respect user set min/max if any
                lrs.append(interpolated_lr)

            return lrs


class WarmupThenDecayLR(LRScheduler):
    """
    Learning rate scheduler with warmup followed by decay.
    
    This is the recommended scheduler for training neural networks:
    1. Warmup phase: Linearly increase LR from a small value to peak_lr
       - Helps stabilize early training when gradients are large/noisy
       - Prevents divergence from bad initial weight configurations
    2. Decay phase: Linearly decrease LR from peak_lr to final_lr
       - Allows fine-grained optimization as training progresses
    
    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of steps for warmup phase (default 5000).
        peak_lr: Learning rate at end of warmup (default 1e-4).
        decay_end_step: Step at which decay ends (default 5M).
        final_lr: Learning rate at end of decay (default 1e-5).
        warmup_start_lr: Learning rate at start of warmup (default 1e-7).
            Should be very small to avoid initial instability.
        last_epoch: The index of last epoch (default -1).
    
    Example milestones this creates:
        warmup_steps=5000, peak_lr=1e-4, decay_end=5M, final_lr=1e-5
        Step 0:     1e-7 (warmup_start_lr)
        Step 5000:  1e-4 (peak_lr) 
        Step 5M:    1e-5 (final_lr)
    """
    
    def __init__(
        self, 
        optimizer, 
        warmup_steps: int = 5000,
        peak_lr: float = 1e-4,
        decay_end_step: int = 5_000_000,
        final_lr: float = 1e-5,
        warmup_start_lr: float = 1e-7,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.decay_end_step = decay_end_step
        self.final_lr = final_lr
        self.warmup_start_lr = warmup_start_lr
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate current learning rate based on step."""
        if not self._get_lr_called_within_step:
            print(
                "Warning: To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )
        
        step = self.last_epoch
        
        if step <= 0:
            # At initialization, return warmup start
            lr = self.warmup_start_lr
        elif step <= self.warmup_steps:
            # Warmup phase: linear interpolation from start to peak
            fraction = step / self.warmup_steps
            lr = self.warmup_start_lr + fraction * (self.peak_lr - self.warmup_start_lr)
        elif step <= self.decay_end_step:
            # Decay phase: linear interpolation from peak to final
            fraction = (step - self.warmup_steps) / (self.decay_end_step - self.warmup_steps)
            lr = self.peak_lr + fraction * (self.final_lr - self.peak_lr)
        else:
            # After decay ends, stay at final_lr
            lr = self.final_lr
        
        return [lr for _ in self.optimizer.param_groups]
