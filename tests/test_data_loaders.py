#!/usr/bin/env python3
"""
Comprehensive tests for C++ data loaders (Opus and LZ4).

Tests include:
- Data shape and dtype validation
- VRAM usage and memory lifecycle
- Performance benchmarks (samples/sec)
- Storage independence (no view issues)

Run with: python -m tests.test_data_loaders
"""

import unittest
import gc
import os
import sys
import time
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_config import (
    TEST_OPUS_DIR, TEST_LZ4_DIR,
    EXPECTED_SAMPLE_RATE, EXPECTED_SAMPLE_SIZE, EXPECTED_CHANNELS,
    ensure_test_dirs
)


def get_gpu_memory_mb(device_idx: int = 0) -> float:
    """Get current GPU memory allocated in MB."""
    if not torch.cuda.is_available():
        return 0
    torch.cuda.synchronize(device_idx)
    return torch.cuda.memory_allocated(device_idx) / (1024 * 1024)


def dataset_exists(directory: str) -> bool:
    """Check if a dataset directory has data in it."""
    train_csv = os.path.join(directory, "train", "train.csv")
    return os.path.exists(train_csv)


class BaseLoaderTest(unittest.TestCase):
    """Base class for loader tests with common setup. Not run directly."""
    
    use_lz4 = None  # Override in subclasses
    test_dir = None  # Override in subclasses
    format_name = None  # Override in subclasses
    
    @classmethod
    def setUpClass(cls):
        """Check if CUDA and test data are available."""
        # Skip base class - only run subclasses
        if cls is BaseLoaderTest:
            raise unittest.SkipTest("Base class, not a test")
            
        cls.cuda_available = torch.cuda.is_available()
        if not cls.cuda_available:
            return
            
        ensure_test_dirs()
        
        if cls.test_dir is None or not dataset_exists(cls.test_dir):
            print(f"\n  SKIP: {cls.format_name} test data not found at {cls.test_dir}")
            print(f"  Run: python -m tests.generate_test_data")
            cls.test_dir = None
    
    def _create_loader(self, batch_size=16, num_preload_batches=8, num_threads=4):
        """Create a loader with standard settings."""
        from clarification.datas.cpp_loader import CppDataLoader
        
        return CppDataLoader(
            device=torch.device('cuda:0'),
            base_dir=os.path.join(self.test_dir, "train"),
            csv_filename="train.csv",
            batch_size=batch_size,
            num_preload_batches=num_preload_batches,
            num_threads=num_threads,
            use_lz4=self.use_lz4
        )
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_batch_shape_and_dtype(self):
        """Test that batches have correct shape and dtype."""
        if not self.test_dir:
            self.skipTest(f"{self.format_name} test data not available")
            
        batch_size = 16
        loader = self._create_loader(batch_size=batch_size, num_preload_batches=4)
        
        # Wait for preload
        time.sleep(3)
        batch = next(loader)
        
        print(f"\n  [{self.format_name}] Batch shape: {batch.shape}")
        print(f"  [{self.format_name}] Batch dtype: {batch.dtype}")
        print(f"  [{self.format_name}] Batch device: {batch.device}")
        
        # Check shape
        self.assertEqual(len(batch.shape), 3, "Batch should be 3D")
        self.assertEqual(batch.shape[0], batch_size, 
            f"First dim should be batch_size={batch_size}")
        self.assertEqual(batch.shape[1], EXPECTED_CHANNELS, 
            f"Second dim should be {EXPECTED_CHANNELS} channels")
        self.assertEqual(batch.shape[2], EXPECTED_SAMPLE_SIZE, 
            f"Third dim should be sample_size={EXPECTED_SAMPLE_SIZE}")
        
        # Check dtype - both loaders output fp16
        self.assertEqual(batch.dtype, torch.float16, 
            "Batch dtype should be float16")
        
        # Check device
        self.assertTrue(batch.device.type == 'cuda', 
            "Batch should be on CUDA device")
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_batch_storage_independence(self):
        """
        CRITICAL: Verify batches have independent storage.
        
        Without .clone() after narrow(), batches share storage with parent,
        causing massive VRAM retention until ALL batches are consumed.
        """
        if not self.test_dir:
            self.skipTest(f"{self.format_name} test data not available")
            
        loader = self._create_loader()
        time.sleep(5)
        
        # Get multiple batches
        batch1 = next(loader)
        batch2 = next(loader)
        batch3 = next(loader)
        
        # Check storage pointers
        storage1 = batch1.untyped_storage().data_ptr()
        storage2 = batch2.untyped_storage().data_ptr()
        storage3 = batch3.untyped_storage().data_ptr()
        
        print(f"\n  [{self.format_name}] Batch storage pointers:")
        print(f"    Batch 1: {storage1}")
        print(f"    Batch 2: {storage2}")
        print(f"    Batch 3: {storage3}")
        
        self.assertNotEqual(storage1, storage2, 
            "Batch 1 and 2 share storage - missing .clone()")
        self.assertNotEqual(storage2, storage3,
            "Batch 2 and 3 share storage - missing .clone()")
        self.assertNotEqual(storage1, storage3,
            "Batch 1 and 3 share storage - missing .clone()")
        
        print(f"  [{self.format_name}] PASS: Each batch has independent storage")
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_vram_sanity(self):
        """Test that VRAM usage is reasonable (not leaking)."""
        if not self.test_dir:
            self.skipTest(f"{self.format_name} test data not available")
            
        gc.collect()
        torch.cuda.empty_cache()
        initial_mem = get_gpu_memory_mb()
        
        loader = self._create_loader(num_preload_batches=8)
        time.sleep(5)
        
        after_preload = get_gpu_memory_mb()
        
        # Expected per batch: [16, 2, 7200] * 2 bytes (float16) = ~0.44 MB
        # Both loaders now output fp16
        bytes_per_element = 2
        expected_per_batch_mb = 16 * 2 * EXPECTED_SAMPLE_SIZE * bytes_per_element / (1024 * 1024)
        actual_delta = after_preload - initial_mem
        estimated_batches = actual_delta / expected_per_batch_mb if expected_per_batch_mb > 0 else 0
        
        print(f"\n  [{self.format_name}] VRAM Usage:")
        print(f"    Initial: {initial_mem:.1f} MB")
        print(f"    After preload: {after_preload:.1f} MB")
        print(f"    Delta: {actual_delta:.1f} MB")
        print(f"    Expected per batch: {expected_per_batch_mb:.2f} MB")
        print(f"    Estimated batches in memory: {estimated_batches:.0f}")
        
        # Should not exceed reasonable limit
        max_expected_mb = 500  # Generous limit
        self.assertLess(actual_delta, max_expected_mb,
            f"VRAM usage {actual_delta:.1f} MB exceeds {max_expected_mb} MB - possible memory leak")
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_memory_release(self):
        """Test that consumed batches release their VRAM."""
        if not self.test_dir:
            self.skipTest(f"{self.format_name} test data not available")
            
        gc.collect()
        torch.cuda.empty_cache()
        initial_mem = get_gpu_memory_mb()
        
        loader = self._create_loader()
        time.sleep(5)
        
        # Hold some batches
        batches = []
        for _ in range(10):
            batches.append(next(loader))
        
        torch.cuda.synchronize()
        with_batches = get_gpu_memory_mb()
        
        # Release batches
        del batches
        gc.collect()
        torch.cuda.empty_cache()
        after_release = get_gpu_memory_mb()
        
        released = with_batches - after_release
        # Both loaders now output fp16
        bytes_per_element = 2
        expected_per_batch = 16 * 2 * EXPECTED_SAMPLE_SIZE * bytes_per_element / (1024 * 1024)
        expected_release = 10 * expected_per_batch
        
        print(f"\n  [{self.format_name}] Memory Release:")
        print(f"    With 10 batches: {with_batches:.1f} MB")
        print(f"    After release: {after_release:.1f} MB")
        print(f"    Released: {released:.1f} MB")
        print(f"    Expected to release: {expected_release:.1f} MB")
        
        # Should release at least 50% of expected (allow for caching)
        self.assertGreater(released, expected_release * 0.5,
            "Consumed batches did not release expected VRAM")
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_performance_samples_per_second(self):
        """Benchmark loader throughput in samples per second."""
        if not self.test_dir:
            self.skipTest(f"{self.format_name} test data not available")
            
        batch_size = 32
        loader = self._create_loader(batch_size=batch_size, num_preload_batches=16)
        
        # Warm up
        time.sleep(3)
        for _ in range(5):
            _ = next(loader)
        
        # Benchmark
        num_batches = 50
        total_samples = 0
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_batches):
            batch = next(loader)
            total_samples += batch.shape[0]
        
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        samples_per_sec = total_samples / elapsed
        batches_per_sec = num_batches / elapsed
        
        print(f"\n  [{self.format_name}] Performance Benchmark:")
        print(f"    Batches processed: {num_batches}")
        print(f"    Total samples: {total_samples}")
        print(f"    Time: {elapsed:.2f}s")
        print(f"    Samples/sec: {samples_per_sec:.0f}")
        print(f"    Batches/sec: {batches_per_sec:.1f}")
        
        # Basic sanity check - should be able to process at least 100 samples/sec
        self.assertGreater(samples_per_sec, 100,
            f"Performance too low: {samples_per_sec:.0f} samples/sec")


class TestOpusLoader(BaseLoaderTest):
    """Tests for Opus audio loader."""
    use_lz4 = False
    test_dir = TEST_OPUS_DIR
    format_name = "Opus"


class TestLz4Loader(BaseLoaderTest):
    """Tests for LZ4 compressed audio loader."""
    use_lz4 = True
    test_dir = TEST_LZ4_DIR
    format_name = "LZ4"


class TestLoaderComparison(unittest.TestCase):
    """Compare behavior between Opus and LZ4 loaders."""
    
    @classmethod
    def setUpClass(cls):
        cls.cuda_available = torch.cuda.is_available()
        cls.opus_available = dataset_exists(TEST_OPUS_DIR)
        cls.lz4_available = dataset_exists(TEST_LZ4_DIR)
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_both_loaders_produce_similar_shapes(self):
        """Both loaders should produce same batch shapes."""
        if not self.opus_available or not self.lz4_available:
            self.skipTest("Both Opus and LZ4 datasets required for comparison")
        
        from clarification.datas.cpp_loader import CppDataLoader
        
        batch_size = 16
        device = torch.device('cuda:0')
        
        opus_loader = CppDataLoader(
            device=device,
            base_dir=os.path.join(TEST_OPUS_DIR, "train"),
            csv_filename="train.csv",
            batch_size=batch_size,
            num_preload_batches=4,
            num_threads=4,
            use_lz4=False
        )
        
        lz4_loader = CppDataLoader(
            device=device,
            base_dir=os.path.join(TEST_LZ4_DIR, "train"),
            csv_filename="train.csv",
            batch_size=batch_size,
            num_preload_batches=4,
            num_threads=4,
            use_lz4=True
        )
        
        time.sleep(5)
        
        opus_batch = next(opus_loader)
        lz4_batch = next(lz4_loader)
        
        print(f"\n  Opus batch shape: {opus_batch.shape}, dtype: {opus_batch.dtype}")
        print(f"  LZ4 batch shape: {lz4_batch.shape}, dtype: {lz4_batch.dtype}")
        
        self.assertEqual(opus_batch.shape[0], lz4_batch.shape[0], 
            "Batch sizes should match")
        self.assertEqual(opus_batch.shape[1], lz4_batch.shape[1], 
            "Channel counts should match")
        self.assertEqual(opus_batch.shape[2], lz4_batch.shape[2], 
            "Sample sizes should match")


if __name__ == '__main__':
    unittest.main(verbosity=2)

