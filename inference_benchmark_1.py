from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

import clarification
from clarification.eval.inference_benchmark import InferenceBenchmark
from clarification.configs.inference_configs import InferenceBenchmarkConfig
from clarification.models import *
from clarification.util import *

default_batch_size = 1
default_num_test_batches = 500

def benchmark_gpu_cpu(name, model, dataset_config, batch_size=default_batch_size, num_test_batches=default_num_test_batches, verbose=False):
    # GPU
    if torch.cuda.is_available():
        benchmark_1_config = InferenceBenchmarkConfig(
            model_name=name,
            model = model,
            dataset_config=dataset_config,
            device=torch.get_default_device(),
            batch_size=batch_size,
            num_test_batches=num_test_batches,
            verbose=verbose,
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
        verbose=verbose,
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
    benchmark_gpu_cpu(dense_model_1_name, dense_model_1, dataset_config)

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
    benchmark_gpu_cpu(resnet_model_1_name, resnet_model_1, dataset_config)

def benchmark_simple(name, layer_sizes):
    dataset_config = clarification.configs.PresetDatasetConfig1(
        dataset_batch_size=1,
        batches_per_iteration=1,
    )
    simple_model_1_name = name
    simple_model_1 = ClarificationSimple(
                name=simple_model_1_name,
                layer_sizes=layer_sizes)
    benchmark_gpu_cpu(simple_model_1_name, simple_model_1, dataset_config)

def benchmark_lstm(name, layer_sizes):
    dataset_config = clarification.configs.PresetDatasetConfig1(
        dataset_batch_size=1,
        batches_per_iteration=1,
    )
    lstm_model_1_name = name
    lstm_model_1 = ClarificationDenseLSTM(
                name=lstm_model_1_name,
                in_channels=1,
                samples_per_batch=dataset_config.samples_per_batch,
                layer_sizes=layer_sizes
    )
    benchmark_gpu_cpu(lstm_model_1_name, lstm_model_1, dataset_config)

def run():
    set_logical_default_device()

    # runs_dir_str = "/workspace/benchmark_runs"
    # Path(runs_dir_str).mkdir(parents=True, exist_ok=True)
    # writer = SummaryWriter(log_dir=runs_dir_str)

    # benchmark_dense("dense-fight-resnet1", [32, 48, 80, 48, 32])
    # benchmark_dense("dense80k-3", [64, 96, 64])
    # benchmark_dense("dense-fight-resnet2", [88, 104, 88])
    # benchmark_resnet("resnet2", 128, 6)

    # 300k class
    benchmark_dense("dense300k-1", [128, 144, 128])
    benchmark_dense("dense300k-2", [80, 80, 80, 80, 80])
    benchmark_dense("dense300k-3", [56, 48, 40, 32, 24, 32, 40, 48, 56])
    benchmark_lstm("denseLSTM300k-1", [24, 24, 24, 24, 24, 24, 24])
    benchmark_resnet("resnet300k-1", 128, 10)
    benchmark_resnet("resnet300k-2", 184, 5)
    benchmark_simple("simple300k-1", [128, 144, 128])
    benchmark_simple("simple300k-2", [96, 96, 96, 96, 96])
    benchmark_simple("simple300k-3", [56, 72, 88, 104, 88, 72, 56])

    pass

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    run()
