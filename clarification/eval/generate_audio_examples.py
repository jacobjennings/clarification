import time
import itertools

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

from ..configs import InferenceBenchmarkConfig
from ..util import *

class InferenceBenchmark:
    def __init__(self, config: InferenceBenchmarkConfig):
        self.c = config

    def run(self):
        if self.c.verbose:
            print(f"\nPreparing inference benchmark for {self.c.model_name}.")

        if self.c.dataset_loader is None:
            loader_iter = RandomTensorIter(shape=(self.c.batch_size, 2, self.c.dataset_config.samples_per_batch))
        else:
            loader_iter = TensorChain(SqueezeIter(iter(self.c.dataset_loader)))

        if self.c.verbose:
            print(f"Starting memory benchmark for {self.c.model_name}")

        memory_max_mb = -1
        batch = next(loader_iter)[:, 0:1, :]
        batch = batch.to(self.c.device)

        model = None
        if str(self.c.device) == "cpu":
            with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
                with record_function("model_inference_cpu"):
                    model = self.c.model_function(*self.c.model_args).to(self.c.device)
                    model.eval()
                    _ = model(batch)
            if self.c.verbose:
                print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
            for event in prof.events():
                memory_max_mb = max(memory_max_mb, event.cpu_memory_usage / 1024 / 1024)

        elif str(self.c.device).startswith("cuda"):
            with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
                with record_function("model_inference_cuda"):
                    model = self.c.model_function(*self.c.model_args).to(self.c.device)
                    model.eval()
                    _ = model(batch)
            if self.c.verbose:
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            for event in prof.events():
                memory_max_mb = max(memory_max_mb, event.device_memory_usage / 1024 / 1024)

        total_params = sum(p.numel() for p in model.parameters())
        if self.c.model_weights_path is not None:
            model.load_state_dict(torch.load(self.c.model_weights_path))

        if self.c.verbose:
            print(f"Starting inference benchmark for {self.c.model_name}")

        start_perf_time = time.perf_counter()

        for batch_idx in range(self.c.num_test_batches):
            batch = next(loader_iter)[:, 0:1, :]
            batch = batch.to(self.c.device)

            _ = model(batch)

        end_perf_time = time.perf_counter()

        perf_time = end_perf_time - start_perf_time

        batches_per_s = self.c.num_test_batches * self.c.batch_size / perf_time
        samples_per_s = batches_per_s * self.c.dataset_config.samples_per_batch
        seconds_per_s = samples_per_s / self.c.dataset_config.sample_rate
        extra_tab = "\t" if len(str(self.c.device)) < 5 else ""
        extra_tab_2 = "\t" if len(str(batches_per_s)) < 5 else ""
        print(f"{self.c.model_name}\t params: {total_params}, \tdevice: {self.c.device}, {extra_tab}\tbatch_size: {self.c.batch_size}, \tnum_batches: {self.c.num_test_batches}, \tbatches_per_s: {batches_per_s:.2f} batches/s, {extra_tab_2}\ttime ratio: {seconds_per_s:.2f} s/s, \tmax memory: {memory_max_mb:.2f} MB")

        del model
