import gc
import platform
import getpass
import logging

logger = logging.getLogger(__name__)
import torch
import torch.nn as nn


def is_mac() -> bool:
    return platform.system() == "Darwin"


def is_cloud() -> bool:
    return getpass.getuser() == "root"


def set_logical_default_device():
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif is_mac():
        torch.set_default_device("mps")
    else:
        print(f"WARNING: No CUDA devices available. Using CPU.")
        torch.set_default_device("cpu")


def clear_cache_and_gc():
    gc.collect()
    if is_mac():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def workspace_dir():
    return "/workspace"


def runs_dir():
    return f"{workspace_dir()}/runs"


def models_dir(a_runs_dir: str):
    return f"{a_runs_dir}/weights"


def profiling_data_dir(a_runs_dir: str):
    return f"{a_runs_dir}/profiling_data"


def dataset_dir():
    return f"{workspace_dir()}/noisy-commonvoice-24k-300ms-5ms-opus"


def run_dir(label: str):
    return f"{runs_dir()}/{label}"


# ============================================================================
# Weight Initialization Functions
# ============================================================================

def init_weights_kaiming(module: nn.Module) -> None:
    """
    Kaiming (He) initialization for networks with ReLU activations.
    
    This initialization maintains variance of activations through ReLU layers,
    preventing vanishing/exploding gradients in deep networks.
    
    Use this for: ClarificationSimple, ClarificationDense
    
    Args:
        module: A PyTorch module to initialize. Apply with model.apply(init_weights_kaiming)
    """
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def init_weights_kaiming_zero_residual(module: nn.Module, zero_bn_names: set = None) -> None:
    """
    Kaiming initialization with zero-init for residual/dense connection outputs.
    
    For layers whose names are in zero_bn_names, the BatchNorm scale is set to 0.
    This makes residual blocks act as identity at initialization, improving
    gradient flow and training stability.
    
    Use this for: ClarificationResNet, ClarificationDense (for the last BN in each block)
    
    Args:
        module: A PyTorch module to initialize
        zero_bn_names: Set of module names whose BatchNorm should be zero-initialized.
                       If None, defaults to standard Kaiming init.
    """
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        # Check if this BN should be zero-initialized
        if zero_bn_names and hasattr(module, '_zero_init') and module._zero_init:
            nn.init.zeros_(module.weight)
        else:
            nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def apply_weight_init(model: nn.Module, init_fn=None) -> None:
    """
    Apply weight initialization to a model.
    
    Args:
        model: The model to initialize
        init_fn: Initialization function (default: init_weights_kaiming)
    """
    if init_fn is None:
        init_fn = init_weights_kaiming
    model.apply(init_fn)
