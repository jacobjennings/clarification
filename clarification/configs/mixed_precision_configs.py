from dataclasses import dataclass, field
from typing import Optional
import logging

import torch

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MixedPrecisionConfig:
    """Configuration for mixed precision training.

    Args:
        use_scaler_dtype: If set, use this dtype for AMP autocast (e.g., torch.bfloat16).
            If None, AMP will not be enabled. bfloat16 is recommended as it has the same
            exponent range as float32, avoiding overflow/underflow issues.
        matmul_batch_count_to_precision: Dictionary mapping batch counts to precision levels
            (e.g. {0: 'high', 10000: "highest"}).
    """
    use_scaler_dtype: Optional[torch.dtype] = None
    matmul_batch_count_to_precision: Optional[dict[int, str]] = None

    def __post_init__(self):
        pass
    
    @property
    def needs_grad_scaler(self) -> bool:
        """Returns True if gradient scaling is needed for this dtype.
        
        GradScaler is only needed for float16, which has limited dynamic range
        and can suffer from underflow during backpropagation.
        
        bfloat16 has the same exponent range as float32, so it doesn't need
        gradient scaling - the loss values won't underflow.
        """
        if self.use_scaler_dtype is None:
            return False
        # Only float16 needs gradient scaling
        return self.use_scaler_dtype == torch.float16
