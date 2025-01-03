"""Training binary."""

import subprocess
import datetime

from multiprocessing import Process
import torch

import clarification as c

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

def start_tensorboard(logdir):
    subprocess.run(["venv/bin/tensorboard", "--logdir", logdir, "--bind_all"])

def train():
    # noinspection PyRedundantParentheses
    tb_process = Process(target=start_tensorboard, args=(("/workspace/runs",)))
    tb_process.start()

    dataset_config = c.training.configs.PresetDatasetConfig1(batches_per_iteration=batches_per_iteration)

    training_date_str = datetime.now().strftime('%Y-%m-%d__%H-%M-%S')

    model_config = c.training.configs.DenseTrainingConfig(
        name="dense_1",
        layer_sizes=[64, 64, 64, 64, 64],
        dataset_config=dataset_config,
        batches_per_iteration=batches_per_iteration,
        training_date_str=training_date_str,
    )

    log_config = c.training.configs.PresetLogBehaviorConfig1(
        runs_subdir_name=f"{model_config.name}-{training_date_str}",
    )

    trainer_config = c.training.configs.AudioTrainerConfig(
        model_training_config=model_config,
        log_behavior_config=log_config,
        training_date_str=training_date_str,
    )

    train_multiple = c.training.train_multiple.TrainMultiple(
        config=c.training.train_multiple.TrainMultipleConfig(
            trainer_configs=[trainer_config],
        )
    )
    train_multiple.run()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    train()
