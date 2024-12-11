"""Training binary."""
import os
import platform

import torch
import torch.nn as nn

import auraloss

import clarification

from torch.utils.tensorboard import SummaryWriter


def train():
    summary_writer = SummaryWriter()

    mac = platform.system() == "Darwin"
    device = "mps" if mac else "cuda"

    models_dir = "."
    if mac:
        base_dataset_directory = '/Users/jacobjennings/noisy-commonvoice-24k-300ms-10ms/en/clear'
        models_dir = '/Users/jacobjennings/denoise-models/2'
    else:
        base_dataset_directory = '/home/jacob/noisy-commonvoice-24k-300ms-10ms/en'
        models_dir = '/home/jacob/denoise-models/2'

    if device == "cuda":
        torch.cuda.empty_cache()
    else:
        torch.mps.empty_cache()

    sample_rate = 24000
    dtype = torch.float32

    sample_batch_ms = 300
    samples_per_batch = int((sample_batch_ms / 1000) * sample_rate)

    overlap_ms = 10
    overlap_samples = int((overlap_ms / 1000) * sample_rate)

    # model_dict_path = models_dir + "/model-20241208-174318"

    def simple_maker(name, layer_sizes):
        simple_model = nn.DataParallel(clarification.models.ClarificationSimple(
            in_channels=1, samples_per_batch=samples_per_batch,
            layer_sizes=layer_sizes,
            device=device, dtype=dtype))
        simple_optimizer = torch.optim.SGD(params=simple_model.parameters(), lr=0.01)
        simple_scheduler = torch.optim.lr_scheduler.LinearLR(
            simple_optimizer, start_factor=0.02, end_factor=0.006, total_iters=6000)

        return name, simple_model, simple_optimizer, simple_scheduler

    models = [
        # simple_maker("simple1", [64, 128, 256, 512, 1024]),
        simple_maker("simple2", [200, 300, 400, 900]),
        # simple_maker("simple3", [100, 300, 500, 900]),
        # simple_maker("simple4", [100, 500, 900]),
        # simple_maker("simple5", [10, 20, 40, 80, 160, 400, 550, 700]),
        simple_maker("simple6", [100, 150, 200, 900]),
        simple_maker("simple7", [300, 500, 900]),
    ]

    models = [
        nn.DataParallel(model) for model in models
    ]

    for model_tuple in models:
        total_params = sum(p.numel() for p in model_tuple[1].parameters())
        summary_writer.add_scalar(f"total_params_{model_tuple[0]}", total_params)
        print(f"total_params_{model_tuple[0]}: {total_params}")

    # model.load_state_dict(torch.load(model_dict_path, weights_only=True))
    # model.eval()

    loss_functions = [
        ("L1Loss", 3.0, nn.L1Loss()),
        ("SISDRLoss", 1.5, auraloss.time.SISDRLoss()),
        ("MelSTFTLoss", 0.5, auraloss.freq.MelSTFTLoss(sample_rate=sample_rate, n_mels=128, device=device)),
    ]

    loader = clarification.datas.commonvoice_loader.CommonVoiceLoader(
        base_dir=base_dataset_directory,
        summary_writer=summary_writer,
        should_pin_memory=device == "cuda",
        device=device
    )

    loader.create_loaders()

    trainer = clarification.training.AudioTrainer(
        input_dataset_loader=loader.train_loader,
        models=models,
        loss_function_tuples=loss_functions,
        sample_rate=sample_rate,
        samples_per_batch=samples_per_batch,
        batches_per_iteration=70,
        device=device,
        overlap_samples=overlap_samples,
        model_weights_dir=models_dir,
        model_weights_save_every_iterations=2000,
        summary_writer=summary_writer,
        send_audio_clip_every_iterations=100
    )

    trainer.train()


if __name__ == '__main__':
    train()
