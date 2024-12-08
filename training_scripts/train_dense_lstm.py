"""Training binary for dense LSTM model."""
import platform

import torch
import torch.nn as nn

import auraloss

from ..clarification.models import clarification_dense_lstm as clarification_model
from ..clarification.data import clarification_commonvoice_loader as loader

def train():
    mac = platform.system() == "Darwin"
    device = "mps" if mac else "cuda"

    models_dir = "."
    if mac:
        base_dataset_directory = '/Users/jacobjennings/noisy-commonvoice-24k/en/clear'
        noisy_dataset_directory = '/Users/jacobjennings/noisy-commonvoice-24k/en/noisy'
        models_dir = '/Users/jacobjennings/denoise-models/2'
    else:
        base_dataset_directory = '/home/jacob/noisy-commonvoice-24k/en/clear'
        noisy_dataset_directory = '/home/jacob/noisy-commonvoice-24k/en/noisy'
        models_dir = '/home/jacob/denoise-models/2'

    if device == "cuda":
        torch.cuda.empty_cache()
    else:
        torch.mps.empty_cache()

    sample_rate = 24000
    dtype=torch.float32

    sample_rate = 24000

    sample_batch_ms = 400
    hidden_size_ms = 600

    samples_per_batch = int((sample_batch_ms / 1000) * sample_rate)
    samples_per_hidden = int((hidden_size_ms / 1000) * sample_rate)

    model_dict_path = models_dir + "/model-20241208-062219"

    # def __init__(self, in_channels, samples_per_batch, device, dtype):

    model = clarification_model.UNet1D(in_channels=1, samples_per_batch=samples_per_batch, device=device, dtype=dtype)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"total_params: {total_params}")

    # model.load_state_dict(torch.load(model_dict_path, weights_only=True))
    # model.eval()

    loss_functions = [
        ("L1Loss", 1.0, nn.L1Loss()),
        ("SISDRLoss", 1.0, auraloss.time.SISDRLoss()),
        ("MelSTFTLoss", 1.0 / 3.0, auraloss.freq.MelSTFTLoss(sample_rate=sample_rate, n_mels=128, device=device)),
    ]
    
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.006, end_factor=0.004, total_iters=100000)

    print(sequence_model)
    
    

if __name__ == '__main__':
    pass
