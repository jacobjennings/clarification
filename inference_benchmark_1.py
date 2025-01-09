from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

import clarification
from clarification.eval.inference_benchmark import InferenceBenchmark
from clarification.configs.inference_configs import InferenceBenchmarkConfig
from clarification.models import *
from clarification.util import *

default_batch_size = 3
default_num_test_batches = 500
default_verbose = False
default_samples_per_batch = 7200

def benchmark_gpu_cpu(name, model_fn, model_args, batch_size=default_batch_size, num_test_batches=default_num_test_batches, verbose=default_verbose):
    dataset_config = clarification.configs.PresetDatasetConfig1(
        dataset_batch_size=1,
        batches_per_iteration=1,
    )
    # GPU
    if torch.cuda.is_available():
        benchmark_1_config = InferenceBenchmarkConfig(
            model_name=name,
            model_function=model_fn,
            model_args=model_args,
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
        model_function=model_fn,
        model_args=model_args,
        dataset_config=dataset_config,
        device=torch.device("cpu"),
        batch_size=batch_size,
        num_test_batches=num_test_batches,
        verbose=verbose,
    )
    benchmark_2 = InferenceBenchmark(config=benchmark_2_config)
    benchmark_2.run()

def dense_maker(name, layer_sizes):
    dense_model_1_name = name
    dense_model_1 = ClarificationDense(
                name=dense_model_1_name,
                layer_sizes=layer_sizes)
    return dense_model_1

def resnet_maker(name, channel_size, layer_count):
    resnet_model_1_name = name
    resnet_model_1 = ClarificationResNet(
                name=resnet_model_1_name,
                channel_size=channel_size,
                layer_count=layer_count)
    return resnet_model_1

def simple_maker(name, layer_sizes):
    simple_model_1_name = name
    simple_model_1 = ClarificationSimple(
                name=simple_model_1_name,
                layer_sizes=layer_sizes)
    return simple_model_1

# def lstm_maker(name, layer_sizes, samples_per_batch=default_samples_per_batch):
#     lstm_model_1_name = name
#     lstm_model_1 = ClarificationDenseLSTM(
#                 name=lstm_model_1_name,
#                 in_channels=1,
#                 samples_per_batch=samples_per_batch,
#                 layer_sizes=layer_sizes
#     )
#     return lstm_model_1

def benchmark_dense(name, layer_sizes):
    benchmark_gpu_cpu(name, model_fn=dense_maker, model_args=(name, layer_sizes))

def benchmark_resnet(name, channel_size, layer_count):
    benchmark_gpu_cpu(name, model_fn=resnet_maker, model_args=(name, channel_size, layer_count))

def benchmark_simple(name, layer_sizes):
    benchmark_gpu_cpu(name, model_fn=simple_maker, model_args=(name, layer_sizes))

# def benchmark_lstm(name, layer_sizes):
#     benchmark_gpu_cpu(name, model_fn=lstm_maker, model_args=(name, layer_sizes))

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
    # benchmark_lstm("denseLSTM300k-1", [24, 24, 24, 24, 24, 24, 24])
    # benchmark_resnet("resnet300k-1", 128, 10)
    # benchmark_resnet("resnet300k-2", 184, 5)
    # benchmark_simple("simple300k-1", [128, 144, 128])
    # benchmark_simple("simple300k-2", [96, 96, 96, 96, 96])
    # benchmark_simple("simple300k-3", [56, 72, 88, 104, 88, 72, 56])

    # 450k class
    benchmark_resnet("resnet450k-1", 152, 11)
    benchmark_resnet("resnet450k-2", 184, 7)
    benchmark_resnet("resnet450k-3", 216, 5)
    benchmark_dense("dense450k-1", [128, 192, 128])
    benchmark_dense("dense450k-2", [64, 96, 128, 96, 64])
    benchmark_dense("dense450k-3", [72, 64, 56, 48, 56, 64, 72])
    benchmark_simple("simple450k-1", [120, 120, 120, 120, 120])
    benchmark_simple("simple450k-2", [24, 72, 120, 168, 120, 72, 24])
    pass

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    run()
