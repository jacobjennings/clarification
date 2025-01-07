from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)
@dataclass(kw_only=True)
class MixedPrecisionConfig:
    """Configuration for mixed precision training.

    Args:
        use_scaler_dtype: If set, use this dtype for the scaler. If None, AMP will not be enabled.
        amp_loss_scalar: Loss scalar for AMP. Compensates for loss functions that behave differently under AMP.
        stop_amp_after_batches: Stop using AMP after this number of batches. Note: Loss functions behave very different
          after this point.
        matmul_batch_count_to_precision: Dictionary mapping batch counts to precision levels
          (e.g. {0: 'high', 10000: "highest"}).
    """
    use_scaler_dtype = None
    amp_loss_scalar: float = 0.1
    stop_amp_after_batches = 300000
    matmul_batch_count_to_precision: dict[int, str] = None

    def __post_init__(self):
        pass
