"""Training binary for dense LSTM model."""
import os
import platform

import torch
import torch.nn as nn

import auraloss

from clarification import datas
from clarification import models
from clarification import training

from torch.utils.tensorboard import SummaryWriter


def train():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    summary_writer = SummaryWriter()

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
    dtype = torch.float32

    sample_rate = 24000

    sample_batch_ms = 400

    samples_per_batch = int((sample_batch_ms / 1000) * sample_rate)

    # model_dict_path = models_dir + "/model-20241208-174318"

    model = models.clarification_dense_lstm.ClarificationDenseLSTM(
        in_channels=1, samples_per_batch=samples_per_batch, device=device, dtype=dtype)

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
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.02, end_factor=0.06, total_iters=6000)

    loader = datas.commonvoice_loader.CommonVoiceLoader(
        base_dataset_directory=base_dataset_directory,
        noisy_dataset_directory=noisy_dataset_directory,
        summary_writer=summary_writer,
        should_pin_memory=device == "cuda",
        device=device
    )

    loader.create_loaders()


    trainer = training.AudioTrainer(
        input_dataset_loader=loader.noisy_loader,
        golden_dataset_loader=loader.clear_loader,
        models=[("clarification_dense_lstm", model, optimizer, scheduler)],
        loss_function_tuples=loss_functions,
        sample_rate=sample_rate,
        samples_per_batch=samples_per_batch,
        batches_per_iteration=24,
        device=device,
        model_weights_dir=models_dir,
        model_weights_save_every_iterations=2000,
        summary_writer=summary_writer,
        send_audio_clip_every_iterations=100
        
    )

    trainer.train()


if __name__ == '__main__':
    train()
