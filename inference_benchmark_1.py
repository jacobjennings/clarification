from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

import clarification
from clarification.eval.inference_benchmark import InferenceBenchmark
from clarification.configs.inference_configs import InferenceBenchmarkConfig
from clarification.models import *
from clarification.util import *

def benchmark_gpu_cpu(name, model, dataset_config, batch_size, num_test_batches):
    # GPU
    if torch.cuda.is_available():
        benchmark_1_config = InferenceBenchmarkConfig(
            model_name=name,
            model = model,
            dataset_config=dataset_config,
            device=torch.get_default_device(),
            batch_size=batch_size,
            num_test_batches=num_test_batches,
        )
        benchmark_1 = InferenceBenchmark(config=benchmark_1_config)
        benchmark_1.run()

    # CPU

    benchmark_2_config = InferenceBenchmarkConfig(
        model_name=name,
        model=model,
        dataset_config=dataset_config,
        device=torch.device("cpu"),
        batch_size=batch_size,
        num_test_batches=num_test_batches,
    )
    benchmark_2 = InferenceBenchmark(config=benchmark_2_config)
    benchmark_2.run()

def benchmark_dense(name, layer_sizes):
    dataset_config = clarification.configs.PresetDatasetConfig1(
        dataset_batch_size=1,
        batches_per_iteration=1,
    )
    dense_model_1_name = name
    dense_model_1 = ClarificationDense(
                name=dense_model_1_name,
                in_channels=1,
                layer_sizes=layer_sizes)
    benchmark_gpu_cpu(dense_model_1_name, dense_model_1, dataset_config, 1, 1000)

def benchmark_resnet(name, channel_size, layer_count):
    dataset_config = clarification.configs.PresetDatasetConfig1(
        dataset_batch_size=1,
        batches_per_iteration=1,
    )
    resnet_model_1_name = name
    resnet_model_1 = ClarificationResNet(
                name=resnet_model_1_name,
                channel_size=channel_size,
                layer_count=layer_count)
    benchmark_gpu_cpu(resnet_model_1_name, resnet_model_1, dataset_config, 1, 1000)

def run():
    set_logical_default_device()

    # runs_dir_str = "/workspace/benchmark_runs"
    # Path(runs_dir_str).mkdir(parents=True, exist_ok=True)
    # writer = SummaryWriter(log_dir=runs_dir_str)

    benchmark_dense("dense-fight-resnet1", [32, 48, 80, 48, 32])
    benchmark_dense("dense80k-3", [64, 96, 64])
    benchmark_dense("dense-fight-resnet2", [88, 104, 88])
    benchmark_resnet("resnet2", 128, 6)

    pass

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    run()
