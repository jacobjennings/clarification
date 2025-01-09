"""Training binary."""

import subprocess
import datetime
import pprint

from multiprocessing import Process
import torch
from torch.utils.tensorboard import SummaryWriter

import clarification as c
from clarification.util import *

# batches_per_iteration = 32
# batches_per_iteration = 64
# batches_per_iteration = 96
# batches_per_iteration = 128
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
batches_per_iteration = 512
# batches_per_iteration = 640
# batches_per_iteration = 768
# batches_per_iteration = 896
# batches_per_iteration = 1024
# batches_per_iteration = 1152
# batches_per_iteration = 1280
# batches_per_iteration = 1408
# batches_per_iteration = 1536
# batches_per_iteration = 1664
# batches_per_iteration = 1792
# batches_per_iteration = 1920
# batches_per_iteration = 2048

def start_tensorboard(logdir):
    subprocess.run(["venv/bin/tensorboard", "--logdir", logdir, "--bind_all"])

class Experiment1:
    def __init__(self):
        set_logical_default_device()

        self.dataset_config, self.dataset_loader, self.validation_config = self.make_shared_configs()
        self.training_date_str = ""


    def dense_config(self, name, layer_sizes):
        log_config = c.configs.PresetLogBehaviorConfig1(
            log_info_every_batches=25000,
            runs_subdir_name=f"{self.training_date_str}-{name}",
        )

        model_config = c.configs.DenseTrainingConfig(
            name=name,
            layer_sizes=layer_sizes,
            dataset_config=self.dataset_config,
            dataset_loader=self.dataset_loader.train_loader,
            batches_per_iteration=batches_per_iteration,
            batches_per_rotation=50000,
            training_date_str=self.training_date_str,
            validation_config=self.validation_config,
        )

        trainer_config = c.configs.AudioTrainerConfig(
            model_training_config=model_config,
            log_behavior_config=log_config,
            training_date_str=self.training_date_str,
        )
        return trainer_config

    def simple_config(self, name, layer_sizes):
        log_config = c.configs.PresetLogBehaviorConfig1(
            log_info_every_batches=5000,
            runs_subdir_name=f"{self.training_date_str}-{name}",
        )

        model_config = c.configs.SimpleTrainingConfig(
            name=name,
            layer_sizes=layer_sizes,
            dataset_config=self.dataset_config,
            dataset_loader=self.dataset_loader.train_loader,
            batches_per_iteration=batches_per_iteration,
            batches_per_rotation=5000,
            training_date_str=self.training_date_str,
            validation_config=self.validation_config,
        )

        trainer_config = c.configs.AudioTrainerConfig(
            model_training_config=model_config,
            log_behavior_config=log_config,
            training_date_str=self.training_date_str,
        )
        return trainer_config


    def resnet_config(self, name, channel_size, layer_count):
        log_config = c.configs.PresetLogBehaviorConfig1(
            log_info_every_batches=5000,
            runs_subdir_name=f"{self.training_date_str}-{name}",
        )

        model_config = c.configs.ResnetTrainingConfig(
            name=name,
            channel_size=channel_size,
            layer_count=layer_count,
            dataset_config=self.dataset_config,
            dataset_loader=self.dataset_loader.train_loader,
            batches_per_iteration=batches_per_iteration,
            batches_per_rotation=5000,
            training_date_str=self.training_date_str,
            validation_config=self.validation_config,
        )

        trainer_config = c.configs.AudioTrainerConfig(
            model_training_config=model_config,
            log_behavior_config=log_config,
            training_date_str=self.training_date_str,
        )
        return trainer_config

    def make_shared_configs(self):
        dataset_config = c.configs.PresetDatasetConfig1(batches_per_iteration=batches_per_iteration)

        dataset_loader = c.configs.PresetCommonVoiceLoader(
            dataset_batch_size=dataset_config.dataset_batch_size,
            batches_per_iteration=dataset_config.batches_per_iteration,
        )
        dataset_loader.create_loaders()

        validation_config = c.configs.PresetValidationConfig1(
            test_batches=5000,
            run_validation_every_batches=100000,
            log_every_batches=5000,
            test_loader=dataset_loader.test_loader,
        )
        return dataset_config, dataset_loader, validation_config

    def train(self):
        self.training_date_str = datetime.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

        configs = [
            # resnet_config(training_date_str, "resnet450k-1", resnet_dataset_config, resnet_dataset_loader, resnet_validation_config, 152, 11, 128),
            # resnet_config(training_date_str, "resnet450k-2", resnet_dataset_config, resnet_dataset_loader, resnet_validation_config, 184, 7, 128),
            # resnet_config(training_date_str, "resnet450k-3", resnet_dataset_config, resnet_dataset_loader, resnet_validation_config, 216, 5, 128),
            # dense_config(training_date_str, "dense450k-1", dense_dataset_config, dense_dataset_loader, dense_validation_config, [128, 192, 128], 128),
            # dense_config(training_date_str, "dense450k-2", dense_dataset_config, dense_dataset_loader, dense_validation_config, [64, 96, 128, 96, 64], 128),
            # dense_config(training_date_str, "dense450k-3", dense_dataset_config, dense_dataset_loader, dense_validation_config, [72, 64, 56, 48, 56, 64, 72], 128),
            # simple_config(training_date_str, "simple450k-1", dense_dataset_config, dense_dataset_loader, dense_validation_config, [120, 120, 120, 120, 120], 128),
            # simple_config(training_date_str, "simple450k-2", dense_dataset_config, dense_dataset_loader, dense_validation_config, [24, 72, 120, 168, 120, 72, 24], 128),

            # dense_config(training_date_str, "dense90k-1", dense_dataset_config, dense_dataset_loader,
            #              dense_validation_config, [28, 32, 36, 40, 44], batches_per_iteration),
            # dense_config(training_date_str, "dense90k-2", dense_dataset_config, dense_dataset_loader,
            #              dense_validation_config, [44, 40, 36, 32, 28], batches_per_iteration),

            self.dense_config("dense70k-61", [80, 24, 64]),
            self.dense_config("dense70k-77", [88, 64, 32]),
            self.dense_config("dense70k-159", [56, 32, 48, 16]),
            self.dense_config("dense70k-171", [64, 40, 16, 24]),
            self.dense_config("dense70k-401", [40, 24, 16, 56, 32]),
            self.dense_config("dense70k-523", [64, 48, 16, 24, 32]),
        ]

        train_multiple_config = c.training.train_multiple.TrainMultipleConfig(
            trainer_configs=configs,
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
    experiment_1 = Experiment1()
    experiment_1.train()
