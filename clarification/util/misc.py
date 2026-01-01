import gc
import platform
import getpass
import logging

logger = logging.getLogger(__name__)
import torch

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
