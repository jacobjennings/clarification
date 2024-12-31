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

def clear_cache_and_gc():
    gc.collect()
    if is_mac():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

def workspace_dir():
    return "/workspace"

def models_dir(a_runs_dir: str):
    return f"{a_runs_dir}/weights"

def profiling_data_dir(a_runs_dir: str):
    return f"{a_runs_dir}/profiling_data"

def dataset_dir():
    return f"{workspace_dir()}/mounted_image/noisy-commonvoice-24k-300ms-5ms-opus"

def runs_dir(label: str):
    return f"{workspace_dir()}/{label}"
