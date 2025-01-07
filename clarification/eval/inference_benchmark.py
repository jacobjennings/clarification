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
        print(f"Preparing inference benchmark for {self.c.model_name} with batch size {self.c.batch_size} and num_batches {self.c.num_test_batches} on device {self.c.device}\nLoading weights, dataset, etc.")
        if self.c.model_weights_path is not None:
            self.c.model.load_state_dict(torch.load(self.c.model_weights_path))

        self.c.model.eval()
        self.c.model = self.c.model.to(self.c.device)

        if self.c.dataset_loader is None:
            loader_iter = RandomTensorIter(shape=(self.c.batch_size, 2, self.c.dataset_config.samples_per_batch))
        else:
            loader_iter = TensorChain(SqueezeIter(iter(self.c.dataset_loader)))

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
        print(f"{self.c.model_name} performance: {batches_per_s:.2f} batches/s, time ratio: {seconds_per_s:.2f} s/s")

        self.c.model = self.c.model.to("cpu")
