"""Training binary."""
import platform
import getpass
import pathlib

import torch
import torch.nn as nn

import auraloss

import clarification

from torch.utils.tensorboard import SummaryWriter


def train():
    summary_writer = SummaryWriter()

    mac = platform.system() == "Darwin"
    device = "mps" if mac else "cuda"

    is_cloud = getpass.getuser() == "root"

    models_dir = "."
    if mac:
        base_dataset_directory = '/Users/jacobjennings/distorted-commonvoice-24k-300ms-10ms-opus2/en'
        models_dir = '/Users/jacobjennings/denoise-models/2'
    elif is_cloud:
        base_dataset_directory = '/workspace/mounted_image/distorted-commonvoice-24k-300ms-10ms-opus2/en'
        models_dir = '/workspace/weights'
    else:
        base_dataset_directory = '/workspace/distorted-commonvoice-24k-300ms-10ms-opus2/en'
        models_dir = '/workspace/weights'

    pathlib.Path(models_dir).mkdir(parents=True, exist_ok=True)

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

    batches_per_iteration = 320
    dataset_batch_size = 8

    print(f"sample_rate: {sample_rate} samples_per_batch: {samples_per_batch} overlap_samples: {overlap_samples} batches_per_iteration: {batches_per_iteration} dataset_batch_size: {dataset_batch_size}")

    def dd_maker(name, convblock_sizes):
        dd_model = clarification.loss.DistortionDetector(
            convblock_sizes=convblock_sizes, samples_per_batch=samples_per_batch,
            batches_per_iteration=batches_per_iteration, device=device, dtype=dtype)
        simple_optimizer = torch.optim.SGD(params=dd_model.parameters(), lr=0.01)
        simple_scheduler = torch.optim.lr_scheduler.LinearLR(
            simple_optimizer, start_factor=0.02, end_factor=0.006, total_iters=10000)

        return name, dd_model, simple_optimizer, simple_scheduler

    def dd_encoder_maker(name, layer_sizes):
        dd_model = clarification.loss.DistortionDetectorDenseEncoder(
            in_channels=1, samples_per_batch=samples_per_batch * dataset_batch_size,
            layer_sizes=layer_sizes, device=device, dtype=dtype)
        simple_optimizer = torch.optim.SGD(params=dd_model.parameters(), lr=0.01)
        simple_scheduler = torch.optim.lr_scheduler.LinearLR(
            simple_optimizer, start_factor=0.02, end_factor=0.006, total_iters=10000)

        return name, dd_model, simple_optimizer, simple_scheduler

    def dd_spec_maker(name, convblock_sizes):
        dd_model = clarification.loss.DistortionDetectorSpec(
            convblock_sizes=convblock_sizes, samples_per_batch=samples_per_batch,
            batches_per_iteration=batches_per_iteration, device=device, dtype=dtype)
        simple_optimizer = torch.optim.SGD(params=dd_model.parameters(), lr=0.01)
        simple_scheduler = torch.optim.lr_scheduler.LinearLR(
            simple_optimizer, start_factor=0.02, end_factor=0.006, total_iters=10000)

        return name, dd_model, simple_optimizer, simple_scheduler

    def dd_combo_maker(name, convblock_sizes):
        dd_model = clarification.loss.DistortionDetectorCombo(
            convblock_sizes=convblock_sizes, samples_per_batch=samples_per_batch,
            batches_per_iteration=batches_per_iteration, device=device, dtype=dtype)
        simple_optimizer = torch.optim.SGD(params=dd_model.parameters(), lr=0.01)
        simple_scheduler = torch.optim.lr_scheduler.LinearLR(
            simple_optimizer, start_factor=0.02, end_factor=0.006, total_iters=10000)

        return name, dd_model, simple_optimizer, simple_scheduler

    models = [
        # dd_maker("dd1", [64, 128, 256]),
        # dd_maker("dd2", [64, 64, 64, 128, 128, 128, 256, 256, 256]),
        # dd_maker("dd3", [256, 256, 256, 256, 256, 256, 256, 256, 256, 256]),
        # dd_unet_maker("dd_unet1", [300, 400, 600]),
        # dd_unet_maker("dd_unet2", [128, 96, 64, 96, 128]),
        # dd_unet_maker("dd_unet3", [128, 64, 128]),
        # dd_unet_maker("dd_unet4", [196, 96, 64]),
        # dd_unet_maker("dd_unet5", [64, 96, 128, 96, 64]),
        # dd_unet_maker("dd_unet6", [128, 128, 128]),
        # dd_encoder_maker("dd_dense_1", [128, 64, 32]),
        # dd_encoder_maker("dd_dense_2", [200, 100]),
        dd_encoder_maker("dd_dense_2", [64, 64, 64, 64, 64]),
        # dd_encoder_maker("dd_dense_3", [64, 64, 64, 64, 64]),
        # dd_spec_maker("dd_spec2", [300, 200, 100, 50]),
        # dd_combo_maker("dd_combo1", [200, 100, 50, 25]),
    ]

    for model_name, model, optimizer, scheduler in models:
        print(f"model: {model_name}\n{model}")

    model_dict_path = models_dir + "/dd_dense_2-20241219-225445"
    model = models[0][1]
    model.load_state_dict(torch.load(model_dict_path, weights_only=True))
    model.eval()

    models = [(a, model.to(device), b, c) for a, model, b, c in models]

    for model_tuple in models:
        total_params = sum(p.numel() for p in model_tuple[1].parameters())
        summary_writer.add_scalar(f"total_params_{model_tuple[0]}", total_params)
        print(f"total_params_{model_tuple[0]}: {total_params}")

    loss_functions = [
        ("L1Loss", 1.0, nn.L1Loss()),
    ]

    loss_functions = [(a, b, c.to(device)) for a, b, c in loss_functions]

    loader = clarification.datas.commonvoice_loader.CommonVoiceLoader(
        base_dir=base_dataset_directory,
        summary_writer=summary_writer,
        dataset_batch_size=dataset_batch_size,
        loader_batch_size=batches_per_iteration // dataset_batch_size,
        should_pin_memory=device == "cuda",
        num_workers=3,
        device=device,
        dataset= clarification.datas.DistortedCommonsDataset(batch_size=8, device=device, base_dir=base_dataset_directory)
    )

    loader.create_loaders()

    summary_writer.add_text("loader_info",
                            f"Training data size in batches: {len(loader.train_loader)}, samples: {len(loader.train_loader) * samples_per_batch}")

    trainer = clarification.training.AudioTrainer(
        input_dataset_loader=loader.train_loader,
        models=models,
        loss_function_tuples=loss_functions,
        sample_rate=sample_rate,
        samples_per_batch=samples_per_batch,
        batches_per_iteration=batches_per_iteration,
        dataset_batch_size=dataset_batch_size,
        device=device,
        overlap_samples=overlap_samples,
        model_weights_dir=models_dir,
        model_weights_save_every_iterations=1000,
        summary_writer=summary_writer,
        send_audio_clip_every_iterations=100,
        dataset_batches_total_length=len(loader.train_loader),
        training_classifier=True
    )

    trainer.train()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    train()
