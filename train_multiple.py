"""Training binary."""
import platform
import getpass
import pathlib
from datetime import datetime
from pathlib import Path
from multiprocessing import Process
import subprocess
import gc
import time
import shutil

from collections.abc import Sequence
import torch
import torch.nn as nn
import auraloss
import clarification
from torch.utils.tensorboard import SummaryWriter

sample_rate = 24000
dtype = torch.float32

sample_batch_ms = 300
samples_per_batch = int((sample_batch_ms / 1000) * sample_rate)

overlap_ms = 5
overlap_samples = int((overlap_ms / 1000) * sample_rate)
dataset_batch_size = 16
# batches_per_iteration = 32
# batches_per_iteration = 64
# batches_per_iteration = 96
# batches_per_iteration = 128
# batches_per_iteration = 160
# batches_per_iteration = 192
# batches_per_iteration = 224
batches_per_iteration = 256
# batches_per_iteration = 288
# batches_per_iteration = 320
# batches_per_iteration = 352
# batches_per_iteration = 384
# batches_per_iteration = 416
# batches_per_iteration = 448
# batches_per_iteration = 480
# batches_per_iteration = 512

def start_tensorboard(logdir):
    subprocess.run(["venv/bin/tensorboard", "--logdir", logdir, "--bind_all"])

def train():
    global batches_per_iteration, sample_rate, dtype, samples_per_batch, sample_batch_ms, overlap_ms, overlap_samples, dataset_batch_size
    mac = platform.system() == "Darwin"
    device = "mps" if mac else "cuda"

    is_cloud = getpass.getuser() == "root"

    if mac:
        base_dataset_directory = '/Users/jacobjennings/noisy-commonvoice-24k-300ms-10ms-opus/en'
        models_dir = '/Users/jacobjennings/denoise-models/2'
        profile_dir = '/Users/jacobjennings/profiling_data'
    else:
        base_dataset_directory = '/workspace/noisy-commonvoice-24k-300ms-5ms-opus'
        models_dir = '/workspace/weights'
        profile_dir = '/workspace/profiling_data'


    pathlib.Path(models_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Tensorboard start in directory: /workspace/runs")
    # noinspection PyRedundantParentheses
    tb_process = Process(target=start_tensorboard, args=(("/workspace/runs",)))
    tb_process.start()

    if device == "cuda":
        torch.cuda.empty_cache()
    else:
        torch.mps.empty_cache()
    
    should_retry = True
    if batches_per_iteration % 16 != 0:
        print("batches_per_iteration must be divisible by 16")

    while should_retry:
        print(f"Attempting training with batches_per_iteration: {batches_per_iteration}")
        date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_dir = f'/workspace/runs/runs-{date_str}'
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        try:
            train_2(device, base_dataset_directory, models_dir, log_dir, profile_dir, date_str)
    
        except torch.OutOfMemoryError as e:
            print(e)
            should_retry = True
            batches_per_iteration = batches_per_iteration - 32
            if batches_per_iteration < 32:
                should_retry = False
            else:
                print(f"Retrying with batches_per_iteration: {batches_per_iteration}")
                shutil.rmtree(log_dir)
                time.sleep(1)
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(1)
            continue
    

    
def train_2(device: str, base_dataset_directory: str, models_dir: str, log_dir: str, profile_dir: str, date_str: str):
    global batches_per_iteration, sample_rate, dtype, samples_per_batch, sample_batch_ms, overlap_ms, overlap_samples, dataset_batch_size

    # Rotate tensorboard outputs
    summary_writer = SummaryWriter(log_dir=log_dir)

    # def dd_encoder_maker(name, scalar, layer_sizes):
    #     dd_model = clarification.loss.DistortionDetectorDenseEncoder(
    #         in_channels=1, samples_per_batch=samples_per_batch * dataset_batch_size,
    #         layer_sizes=layer_sizes, device=device, dtype=dtype)

    #     return name, scalar, dd_model, True, samples_per_batch * dataset_batch_size

    def simple_maker(name, layer_sizes, invert=False, num_output_convblocks=2):
        global batches_per_iteration, sample_rate, dtype, samples_per_batch, sample_batch_ms, overlap_ms, overlap_samples, dataset_batch_size

        simple_model = nn.DataParallel(clarification.models.ClarificationSimple(
            name=name,
            in_channels=1,
            layer_sizes=layer_sizes, device=device, dtype=dtype, invert=invert,
            num_output_convblocks=num_output_convblocks)).to(device)
        simple_optimizer = torch.optim.SGD(params=simple_model.parameters(), lr=0.01)
        simple_scheduler = torch.optim.lr_scheduler.LinearLR(
            simple_optimizer, start_factor=0.001, end_factor=0.00001, total_iters=100000)
        
        config = clarification.training.AudioModelTrainingConfig(
            name=name,
            model=simple_model,
            optimizer=simple_optimizer,
            scheduler=simple_scheduler,
            batches_per_model_rotation=15000,
            writer=SummaryWriter(log_dir=f"{log_dir}-{name}")
        )

        return config

    def dense_maker(name, layer_sizes, invert=False, num_output_convblocks=2, milestones = [(0, 0.02), (45000, 0.01), (1000000, 0.005)], batches_per_model_rotation=100000):
        global batches_per_iteration, sample_rate, dtype, samples_per_batch, sample_batch_ms, overlap_ms, overlap_samples, dataset_batch_size

        simple_model = nn.DataParallel(clarification.models.ClarificationDense(
            name=name,
            in_channels=1,
            layer_sizes=layer_sizes, device=device, dtype=dtype, invert=invert,
            num_output_convblocks=num_output_convblocks)).to(device)
        simple_optimizer = torch.optim.SGD(params=simple_model.parameters(), lr=0.01)
        scheduler = clarification.schedulers.InterpolatingLR(
            optimizer=simple_optimizer,
            milestones=milestones,
            verbose=True
        )
        
        config = clarification.training.AudioModelTrainingConfig(
            name=name,
            model=simple_model,
            optimizer=simple_optimizer,
            scheduler=scheduler,
            batches_per_model_rotation=batches_per_model_rotation,
            writer=SummaryWriter(log_dir=f"{log_dir}-{name}")
        )

        return config

    def dense_lstm_maker(name, layer_sizes, invert=False, num_output_convblocks=2, milestones = [(0, 0.02), (45000, 0.01), (1000000, 0.005)], batches_per_model_rotation=100000):
        global batches_per_iteration, sample_rate, dtype, samples_per_batch, sample_batch_ms, overlap_ms, overlap_samples, dataset_batch_size

        simple_model = nn.DataParallel(clarification.models.ClarificationDenseLSTM(
            name=name,
            in_channels=1,
            layer_sizes=layer_sizes, 
            samples_per_batch=samples_per_batch,
            device=device, dtype=dtype, invert=invert,
            num_output_convblocks=num_output_convblocks)).to(device)
        simple_optimizer = torch.optim.SGD(params=simple_model.parameters(), lr=0.01)
        scheduler = clarification.schedulers.InterpolatingLR(
            optimizer=simple_optimizer,
            milestones=milestones,
            verbose=True
        )

        config = clarification.training.AudioModelTrainingConfig(
            name=name,
            model=simple_model,
            optimizer=simple_optimizer,
            scheduler=scheduler,
            batches_per_model_rotation=batches_per_model_rotation,
            writer=SummaryWriter(log_dir=f"{log_dir}-{name}")
        )
        return config

    models = [
        # simple_maker("simple1", [64, 128, 256, 512, 1024]),
        # simple_maker("simple2", [200, 300, 400, 900]), # Way underperformed 6 and 7. Clipping a lot.
        # simple_maker("simple3", [100, 300, 500, 900]),
        # simple_maker("simple4", [100, 500, 900]),
        # simple_maker("simple5", [10, 20, 40, 80, 160, 400, 550, 700]),
        # simple_maker("simple6", [100, 150, 200, 900]), # loss: -9, sounded not as good as simple7.
        # simple_maker("simple7", [300, 500, 900]),
        # simple_maker("simple8", [400, 500, 600]),
        # simple_maker("simple9", [300, 500, 700]),
        # simple_maker("simple9_3convout", [300, 500, 700], num_output_convblocks=3),
        # simple_maker("simple9_inverted", [300, 500, 700], invert=True),
        # simple_maker("simple9_inverted_3convout", [300, 500, 700], invert=True, num_output_convblocks=3),
        # simple_maker("simple10", [300, 400, 600]),
        # simple_maker("simple11", [200, 400, 700, 1000]),
        # simple_maker("simple12", [200, 400]),
        # simple_maker("simple13", [128, 64, 32, 64, 128]),
        # simple_maker("simple14", [32, 64, 128, 64, 32]),
        # simple_maker("simple15", [40, 80, 160, 80, 40]),
        # simple_maker("simple16", [64, 128, 256, 128, 64]),
        # simple_maker("simple17", [16, 32, 64, 32, 16]),
        # simple_maker("simple20", [224, 288, 352, 288, 224]),

        # dense_maker("dense2", [48, 96, 192, 96, 48]),
        # dense_maker("dense1", [32, 64, 128, 64, 32]),
        # dense_maker("dense3", [64, 128, 256, 128, 64])

        # dense_maker("dense3-0.00001", [64, 128, 256, 128, 64], milestones=[(0, 0.00001)], batches_per_model_rotation=10000),
        # dense_maker("dense3-0.0001", [64, 128, 256, 128, 64], milestones=[(0, 0.0001)], batches_per_model_rotation=10000),
        # dense_maker("dense3-0.001", [64, 128, 256, 128, 64], milestones=[(0, 0.001)], batches_per_model_rotation=10000),
        # dense_maker("dense3-0.01", [64, 128, 256, 128, 64], milestones=[(0, 0.01)], batches_per_model_rotation=10000),

        # dense_maker("dense4", [96, 64, 64, 64, 96])
        # dense_maker("dense5", [64, 64, 64, 64, 64])
        # dense_maker("dense6", [128, 196, 256, 196, 128], start_factor=0.03, end_factor=0.006, iters=1000000),
        # dense_lstm_maker("denselstm7", [256, 320, 384, 320, 256]),
        # dense_maker("dense9-1", [192, 256, 320, 256, 192], milestones=[(0, 0.016)], batches_per_model_rotation=10000),
        # dense_maker("dense9-2", [192, 256, 320, 256, 192], milestones=[(0, 0.019)], batches_per_model_rotation=10000),
        # dense_maker("dense9-3", [192, 256, 320, 256, 192], milestones=[(0, 0.021)], batches_per_model_rotation=10000),

        # dense_maker("dense9-4", [192, 256, 320, 256, 192], milestones=[(0, 0.00001)], batches_per_model_rotation=15000),

        # dense_maker("dense9-5", [192, 256, 320, 256, 192], milestones=[(0, 0.025)], batches_per_model_rotation=10000),
        # dense_maker("dense9-6", [192, 256, 320, 256, 192], milestones=[(0, 0.027)], batches_per_model_rotation=10000),
        # dense_maker("dense9", [192, 256, 320, 256, 192]),
        # dense_lstm_maker("denselstm6", [128, 196, 256, 196, 128], milestones=[(0, 0.00001)], batches_per_model_rotation=15000),
        # dense_maker("dense7", [256, 320, 384, 320, 256], start_factor=0.03, end_factor=0.006, iters=1000000)
        # dense_maker("dense8", [256, 512, 1024, 512, 256])
        # simple_maker("simple18", [96, 96, 96, 96, 96]),
        # simple_maker("simple19", [256, 512, 768, 1024, 768, 512, 256]),
        # dense_lstm_maker("denselstm8", [128, 128, 128, 128, 128])

        # dense_maker("dense10", [96, 96, 96, 96, 96], milestones=[(0, 0.01), (500000, 0.00001)], batches_per_model_rotation=30000),

        # dense_maker("dense80k-1", [72, 64, 72], milestones=[(0, 0.01), (500000, 0.00001)], batches_per_model_rotation=30000),
        # dense_maker("dense80k-2", [64, 80, 64], milestones=[(0, 0.01), (500000, 0.00001)], batches_per_model_rotation=30000),
        # dense_maker("dense80k-3", [64, 96, 64], milestones=[(0, 0.01), (500000, 0.00001)], batches_per_model_rotation=30000), # winner
        # dense_maker("dense80k-4", [40, 40, 40, 40, 40], milestones=[(0, 0.01), (500000, 0.00001)], batches_per_model_rotation=30000),
        # dense_maker("dense80k-5", [16, 48, 80, 48, 16], milestones=[(0, 0.01), (500000, 0.00001)], batches_per_model_rotation=30000),

        # Dense Params: 167008 Dense Layers: [32, 64, 96, 64, 32]
        # Dense Params: 168120 Dense Layers: [8, 72, 136, 72, 8]
        dense_maker("dense160k-1", [8, 72, 136, 72, 8], milestones=[(0, 0.01), (500000, 0.00001)], batches_per_model_rotation=30000),
        dense_maker("dense160k-2", [16, 48, 80, 48, 16], milestones=[(0, 0.01), (500000, 0.00001)], batches_per_model_rotation=30000),
    ]

    for model_config in models:
        print(f"{model_config}")

    # model_dict_path = models_dir + "/dense4-20241222-184328"
    # model = models[0][1]
    # model.load_state_dict(torch.load(model_dict_path, weights_only=True))
    # model.eval()

    for model_config in models:
        total_params = sum(p.numel() for p in model_config.model.parameters())
        summary_writer.add_text(f"total_params_{model_config.name}", f"{total_params}")
        print(f"total_params_{model_config.name}: {total_params}")

    # distortion_loss = dd_encoder_maker("dd_dense_2", 10.0, [64, 64, 64, 64, 64])
    # distortion_dict_path = models_dir + "/dd_dense_2-20241219-225445"
    # distortion_loss[2].load_state_dict(torch.load(distortion_dict_path, weights_only=True))
    # distortion_loss[2].eval()

    loss_functions = [
        # Main group
        clarification.training.AudioLossFunctionConfig(
            name="L1Loss", weight=2.0, fn=nn.L1Loss().to(device), is_unary=False, batch_size=None),
        clarification.training.AudioLossFunctionConfig(
            name="SISDRLoss", weight=1.5, fn=auraloss.time.SISDRLoss().to(device), is_unary=False, batch_size=None),
        clarification.training.AudioLossFunctionConfig(
            name="MelSTFTLoss", weight=0.5,
            fn=auraloss.freq.MelSTFTLoss(sample_rate=sample_rate, n_mels=128, device=device).to(device),
            is_unary=False, batch_size=None),

        # clarification.training.AudioLossFunctionConfig(
        #     name="L1Loss", weight=0.5, fn=nn.L1Loss().to(device), is_unary=False, batch_size=None),
        # clarification.training.AudioLossFunctionConfig(
        #     name="SISDRLoss", weight=.375, fn=auraloss.time.SISDRLoss().to(device), is_unary=False, batch_size=None),
        # clarification.training.AudioLossFunctionConfig(
        #     name="MelSTFTLoss", weight=0.125,
        #     fn=auraloss.freq.MelSTFTLoss(sample_rate=sample_rate, n_mels=128, device=device).to(device),
        #     is_unary=False, batch_size=None),

        # clarification.training.AudioLossFunctionConfig(
        #     name="SISDRLoss", weight=.375, fn=auraloss.time.SISDRLoss().to(device), is_unary=False, batch_size=None),

        # ("L1Loss", 5.0, nn.L1Loss(), False, None),
        # ("MSELoss", 10.0, nn.MSELoss(), False, None),
        # ("TripletMarginLoss", 1.0, nn.TripletMarginLoss(), False, None),
        # ("SISDRLoss", 1.0, auraloss.time.SISDRLoss().to(device), False, None),
        # # Mel and random seem to fight each other when loss levels out.
        # ("MelSTFTLoss", 1.0, auraloss.freq.MelSTFTLoss(sample_rate=sample_rate, n_mels=128, device=device), False, None),

        # distortion_loss,
        # ("RandomResolutionSTFTLoss", 1.3, auraloss.freq.RandomResolutionSTFTLoss(sample_rate=sample_rate, n_mels=128, device=device)),
        # ("ESRLoss", 10.0, auraloss.time.ESRLoss().to(device))

        # ("MelSTFTLoss", 1.0, auraloss.freq.MelSTFTLoss(sample_rate=sample_rate, n_mels=128, device=device)),

        # ("SISDRLoss", 1.5, auraloss.time.SISDRLoss().to(device), False, None),
    ]

    loader = clarification.datas.commonvoice_loader.CommonVoiceLoader(
        base_dir=base_dataset_directory,
        summary_writer=summary_writer,
        dataset_batch_size=dataset_batch_size,
        batches_per_iteration=batches_per_iteration,
        should_pin_memory=device == "cuda",
        num_workers=2,
        device=device
    )

    loader.create_loaders()

    summary_writer.add_text("loader_info",
                            f"Training data size in batches: {len(loader.train_loader)}, samples: {len(loader.train_loader) * batches_per_iteration * samples_per_batch}, length: {len(loader.train_loader) * batches_per_iteration * 0.3  / 60 / 60} hours")

    validation_config = clarification.training.ValidationConfig(
        test_loader=loader.test_loader,
        test_batches=15000,
        run_validation_every_batches=45000
    )
    
    trainer = clarification.training.AudioTrainer(
        input_dataset_loader=loader.train_loader,
        models=models,
        loss_function_configs=loss_functions,
        sample_rate=sample_rate,
        samples_per_batch=samples_per_batch,
        batches_per_iteration=batches_per_iteration,
        device=device,
        overlap_samples=overlap_samples,
        training_date_str=date_str,
        model_weights_dir=models_dir,
        profile_output_dir=profile_dir,
        summary_writer=summary_writer,
        dataset_batches_length=len(loader.train_loader),
        dataset_batch_size=dataset_batch_size,
        validation_config=validation_config,
    )
    trainer.train()



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    train()
