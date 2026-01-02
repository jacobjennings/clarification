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
