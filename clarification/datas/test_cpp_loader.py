import unittest
import torch
import os
import shutil
import subprocess
from clarification.datas.cpp_loader import CppDataLoader

class TestCppLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Dedicated directory for unit tests
        cls.test_dir = "/workspace/noisy-commonvoice-unittests-forunittests"
        
        # Cleanup if exists
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
            
        print(f"\nGenerating test dataset (10000 files) in: {cls.test_dir}")
        # Run prepare_data.py to generate 10000 files
        cmd = [
            "venv/bin/python", "clarification/datas/prepare_data.py",
            "--format", "lz4",
            "--dtype", "fp16",
            "--limit", "10000",
            "--out_dir", cls.test_dir
        ]
        # Set PYTHONPATH so relative imports work
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error generating test data: {result.stderr}")
            raise RuntimeError("Failed to generate test data for unit tests.")
        print("Data generation complete.")

    def test_loader_iteration(self):
        print(f"\nTesting C++ loader with base_dir: {self.test_dir}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        try:
            # We use a larger batch size since we have more data
            batch_size = 16
            loader = CppDataLoader(
                device=device,
                base_dir=self.test_dir + "/train",
                csv_filename="train.csv",
                batch_size=batch_size,
                num_preload_batches=8,
                num_threads=8,
                use_lz4=True
            )
            
            print(f"Successfully initialized loader. Total files: {loader.total_files}")
            
            # Check total_files property
            total_files = loader.total_files
            self.assertEqual(total_files, loader.loader.total_files())
            print(f"Total files reported correctly: {total_files}")
            
            # Iterate through more batches to check performance/stability
            num_batches_to_test = 200
            print(f"Iterating through {num_batches_to_test} batches...")
            for i in range(num_batches_to_test):
                try:
                    batch = next(loader)
                    if i == 0:
                        print(f"First batch shape: {batch.shape}, dtype: {batch.dtype}, device: {batch.device}")
                    
                    self.assertEqual(len(batch.shape), 3) # [batch_size, 2, samples]
                    self.assertEqual(batch.shape[0], batch_size)
                    self.assertEqual(batch.shape[1], 2)
                except StopIteration:
                    print(f"Reached end of dataset at batch {i}")
                    break
                
            print(f"Successfully iterated through {i+1} batches.")
            
        except Exception as e:
            self.fail(f"C++ loader failed with error: {e}")

    @classmethod
    def tearDownClass(cls):
        # Cleanup test data after successful run
        # if os.path.exists(cls.test_dir):
        #     shutil.rmtree(cls.test_dir)
        pass

if __name__ == "__main__":
    unittest.main()
