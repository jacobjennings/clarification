import time
import itertools

import torch
import torch.nn as nn

from ..configs import InferenceBenchmarkConfig
from ..util import *

class InferenceBenchmark:
    def __init__(self, config: InferenceBenchmarkConfig):
        self.c = config

    def run(self):
        total_params = sum(p.numel() for p in self.c.model.parameters())
        if self.c.verbose:
            print(f"\nPreparing inference benchmark for {self.c.model_name}. total_params: {total_params}, batch size {self.c.batch_size}, num_batches {self.c.num_test_batches}, device {self.c.device}\nLoading weights, dataset, etc.")
        if self.c.model_weights_path is not None:
            self.c.model.load_state_dict(torch.load(self.c.model_weights_path))

        self.c.model.eval()
        self.c.model = self.c.model.to(self.c.device)

        if self.c.dataset_loader is None:
            loader_iter = RandomTensorIter(shape=(self.c.batch_size, 2, self.c.dataset_config.samples_per_batch))
        else:
            loader_iter = TensorChain(SqueezeIter(iter(self.c.dataset_loader)))

        if self.c.verbose:
            print(f"Starting inference benchmark for {self.c.model_name}")
        start_perf_time = time.perf_counter()

        for batch_idx in range(self.c.num_test_batches):
            batch = next(loader_iter)[:, 0:1, :]
            batch = batch.to(self.c.device)
            _ = self.c.model(batch)

        end_perf_time = time.perf_counter()

        perf_time = end_perf_time - start_perf_time

        batches_per_s = self.c.num_test_batches * self.c.batch_size / perf_time
        samples_per_s = batches_per_s * self.c.dataset_config.samples_per_batch
        seconds_per_s = samples_per_s / self.c.dataset_config.sample_rate
        extra_tab = "\t" if len(str(self.c.device)) < 5 else ""
        extra_tab_2 = "\t" if len(str(batches_per_s)) < 6 else ""
        print(f"{self.c.model_name}\t params: {total_params}, \tdevice: {self.c.device}, {extra_tab}\tbatch_size: {self.c.batch_size}, \tnum_batches: {self.c.num_test_batches}, \tbatches_per_s: {batches_per_s:.2f} batches/s, {extra_tab_2}\ttime ratio: {seconds_per_s:.2f} s/s")

        self.c.model = self.c.model.to("cpu")
