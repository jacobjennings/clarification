from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

import clarification
from clarification.eval.inference_benchmark import InferenceBenchmark
from clarification.configs.inference_configs import InferenceBenchmarkConfig
from clarification.models import ClarificationDense
from clarification.util import *


def run():
    set_logical_default_device()

    # runs_dir_str = "/workspace/benchmark_runs"
    # Path(runs_dir_str).mkdir(parents=True, exist_ok=True)
    # writer = SummaryWriter(log_dir=runs_dir_str)

    dataset_config = clarification.configs.PresetDatasetConfig1(
        dataset_batch_size=1,
        batches_per_iteration=1,
    )

    dense_model_1_name = "dense_1"
    dense_model_1 = ClarificationDense(
                name=dense_model_1_name,
                in_channels=1,
                layer_sizes=[88, 104, 88])

    # Benchmark 1
    if torch.cuda.is_available():
        benchmark_1_config = InferenceBenchmarkConfig(
            model_name=dense_model_1_name,
            model = dense_model_1,
            dataset_config=dataset_config,
            model_weights_path=None,
            device=torch.get_default_device(),
            batch_size=1,
            num_test_batches=1000,
        )
        benchmark_1 = InferenceBenchmark(config=benchmark_1_config)
        benchmark_1.run()

    # Benchmark 2

    benchmark_2_config = InferenceBenchmarkConfig(
        model_name=dense_model_1_name,
        model = dense_model_1,
        dataset_config=dataset_config,
        model_weights_path=None,
        device=torch.device("cpu"),
        batch_size=1,
        num_test_batches=1000,
    )
    benchmark_2 = InferenceBenchmark(config=benchmark_2_config)
    benchmark_2.run()

    pass

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    run()
