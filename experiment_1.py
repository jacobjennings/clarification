"""Training binary."""

import subprocess
import datetime
import pprint

from multiprocessing import Process
import torch
from torch.utils.tensorboard import SummaryWriter

import clarification as c
from clarification.util import *

def start_tensorboard(logdir):
    subprocess.run(["venv/bin/tensorboard", "--logdir", logdir, "--bind_all"])

def dense_config(training_date_str: str):
    # batches_per_iteration = 32
    # batches_per_iteration = 64
    # batches_per_iteration = 96
    # batches_per_iteration = 128
    batches_per_iteration = 160
    # batches_per_iteration = 192
    # batches_per_iteration = 224
    # batches_per_iteration = 256
    # batches_per_iteration = 288
    # batches_per_iteration = 320
    # batches_per_iteration = 352
    # batches_per_iteration = 384
    # batches_per_iteration = 416
    # batches_per_iteration = 448
    # batches_per_iteration = 480
    # batches_per_iteration = 512
    # batches_per_iteration = 640
    # batches_per_iteration = 768

    model_config_name = "dense-fight-resnet2"
    log_config = c.configs.PresetLogBehaviorConfig1(
        log_info_every_batches=5000,
        runs_subdir_name=f"{training_date_str}-{model_config_name}",
    )

    dataset_config = c.configs.PresetDatasetConfig1(batches_per_iteration=batches_per_iteration)

    dataset_loader = c.configs.PresetCommonVoiceLoader(
        summary_writer=log_config.writer,
        dataset_batch_size=dataset_config.dataset_batch_size,
        batches_per_iteration=dataset_config.batches_per_iteration,
    )
    dataset_loader.create_loaders()

    validation_config = c.configs.PresetValidationConfig1(
        test_batches=5000,
        run_validation_every_batches=80000,
        log_every_batches=5000,
        test_loader=dataset_loader.test_loader,
    )

    model_config = c.configs.DenseTrainingConfig(
        name=model_config_name,
        layer_sizes= [88, 104, 88],
        dataset_config=dataset_config,
        dataset_loader=dataset_loader.train_loader,
        batches_per_iteration=batches_per_iteration,
        batches_per_rotation=5000,
        training_date_str=training_date_str,
        validation_config=validation_config,
    )

    trainer_config = c.configs.AudioTrainerConfig(
        model_training_config=model_config,
        log_behavior_config=log_config,
        training_date_str=training_date_str,
    )
    return trainer_config


def resnet_config(training_date_str: str):
    # batches_per_iteration = 32
    # batches_per_iteration = 64
    # batches_per_iteration = 96
    batches_per_iteration = 128
    # batches_per_iteration = 160
    # batches_per_iteration = 192
    # batches_per_iteration = 224
    # batches_per_iteration = 256
    # batches_per_iteration = 288
    # batches_per_iteration = 320
    # batches_per_iteration = 352
    # batches_per_iteration = 384
    # batches_per_iteration = 416
    # batches_per_iteration = 448
    # batches_per_iteration = 480
    # batches_per_iteration = 512
    # batches_per_iteration = 640
    # batches_per_iteration = 768

    model_config_name = "resnet2"
    log_config = c.configs.PresetLogBehaviorConfig1(
        log_info_every_batches=5000,
        runs_subdir_name=f"{training_date_str}-{model_config_name}",
    )

    dataset_config = c.configs.PresetDatasetConfig1(batches_per_iteration=batches_per_iteration)

    dataset_loader = c.configs.PresetCommonVoiceLoader(
        summary_writer=log_config.writer,
        dataset_batch_size=dataset_config.dataset_batch_size,
        batches_per_iteration=dataset_config.batches_per_iteration,
    )
    dataset_loader.create_loaders()

    validation_config = c.configs.PresetValidationConfig1(
        test_batches=5000,
        run_validation_every_batches=80000,
        log_every_batches=5000,
        test_loader=dataset_loader.test_loader,
    )

    model_config = c.configs.ResnetTrainingConfig(
        name=model_config_name,
        channel_size=128,
        layer_count=6,
        dataset_config=dataset_config,
        dataset_loader=dataset_loader.train_loader,
        batches_per_iteration=batches_per_iteration,
        batches_per_rotation=5000,
        training_date_str=training_date_str,
        validation_config=validation_config,
    )

    trainer_config = c.configs.AudioTrainerConfig(
        model_training_config=model_config,
        log_behavior_config=log_config,
        training_date_str=training_date_str,
    )
    return trainer_config


def train():
    set_logical_default_device()

    training_date_str = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

    dense_1_config = dense_config(training_date_str)
    resnet_1_config = resnet_config(training_date_str)

    train_multiple_config = c.training.train_multiple.TrainMultipleConfig(
        trainer_configs=[
            dense_1_config,
            resnet_1_config,
        ],
        should_perform_memory_test=True,
    )
    train_multiple = c.training.train_multiple.TrainMultiple(
        config=train_multiple_config
    )

    # noinspection PyRedundantParentheses
    # tb_process = Process(target=start_tensorboard, args=(("/workspace/runs",)))
    # tb_process.start()

    train_multiple.run()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    train()
