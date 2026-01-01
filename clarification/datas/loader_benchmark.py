import unittest
import time

from . import commonvoice_loader
from ..configs import dataset_configs as c

class MyTestCase(unittest.TestCase):
    def test_perf(self):
        dataset_loader = c.PresetCommonVoiceLoader(
            dataset_batch_size=16,
            batches_per_iteration=64,
        )
        dataset_loader.create_loaders()

        dataset_iter = iter(dataset_loader.test_loader)

        start_time = time.perf_counter()

        num_batches_test = 100

        for i in range(num_batches_test):
            batch = next(dataset_iter)
            print(f"batch: {batch.shape}")

        end_time = time.perf_counter()

        # Log frames per second, and samples per microsecond
        print(f"dataset_loader.batch_size: {dataset_loader.loader_batch_size}")
        print(f"Frames per second: {num_batches_test * dataset_loader.loader_batch_size * dataset_loader.dataset_batch_size / (end_time - start_time)}")
        print(f"Samples per microsecond: {dataset_loader.loader_batch_size * dataset_loader.dataset_batch_size * 7200 / (end_time - start_time) / 1000000}")





if __name__ == '__main__':
    unittest.main()
