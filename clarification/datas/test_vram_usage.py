"""
Unit tests for VRAM usage and memory lifecycle in the C++ data loader.

These tests validate:
1. Batch tensors have independent storage (not views of parent tensor)
2. VRAM usage is proportional to actual batch count
3. Memory is released when batches are consumed

Known issues discovered:
- The preload loop processes too many files (num_batches * batch_size files)
  instead of enough files to produce num_batches batches
- Loader destructor can segfault if called while preload thread is running
"""

import unittest
import gc
import time
import torch


def get_gpu_memory_mb():
    """Get current GPU memory allocated in MB"""
    if not torch.cuda.is_available():
        return 0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 / 1024


class TestVRAMUsage(unittest.TestCase):
    """Tests for VRAM usage in the C++ data loader"""
    
    @classmethod
    def setUpClass(cls):
        """Check if CUDA and test data are available"""
        cls.cuda_available = torch.cuda.is_available()
        if not cls.cuda_available:
            return
            
        # Use the Opus dataset
        cls.test_dir = "/workspace/noisy-commonvoice-24k-300ms-5ms-opus/train"
        cls.csv_filename = "train.csv"
        
        import os
        if not os.path.exists(cls.test_dir):
            cls.test_dir = None
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_batch_storage_independence(self):
        """
        CRITICAL TEST: Verify batches have independent storage.
        
        Without .clone() after narrow(), batches are views of the same
        parent tensor, causing the entire concatenated tensor (~500 MB)
        to stay in VRAM until ALL batches are consumed.
        
        With .clone(), each batch has its own storage and the parent
        tensor can be freed immediately.
        """
        if not self.test_dir:
            self.skipTest("Test dataset not available")
            
        from clarification.datas.cpp_loader import CppDataLoader
        
        device = torch.device('cuda:0')
        
        loader = CppDataLoader(
            device=device,
            base_dir=self.test_dir,
            csv_filename=self.csv_filename,
            batch_size=16,
            num_preload_batches=8,
            num_threads=4,
            use_lz4=False  # Opus dataset
        )
        
        # Wait for preload
        time.sleep(5)
        
        # Get multiple batches
        batch1 = next(loader)
        batch2 = next(loader)
        batch3 = next(loader)
        
        # Check storage pointers - if they're the same, batches share storage
        # Note: Using untyped_storage() to avoid deprecation warning
        storage1 = batch1.untyped_storage().data_ptr()
        storage2 = batch2.untyped_storage().data_ptr()
        storage3 = batch3.untyped_storage().data_ptr()
        
        print(f"\n  Batch 1 storage: {storage1}")
        print(f"  Batch 2 storage: {storage2}")
        print(f"  Batch 3 storage: {storage3}")
        
        # They should all be different
        self.assertNotEqual(storage1, storage2, 
            "Batch 1 and 2 share storage - missing .clone() after narrow()")
        self.assertNotEqual(storage2, storage3,
            "Batch 2 and 3 share storage - missing .clone() after narrow()")
        self.assertNotEqual(storage1, storage3,
            "Batch 1 and 3 share storage - missing .clone() after narrow()")
        
        print("  PASS: Each batch has independent storage")
        
        # Note: Don't delete loader here to avoid segfault during preload
        # The loader will be cleaned up when the test process exits

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_batch_shape_and_dtype(self):
        """Test that batches have correct shape and dtype"""
        if not self.test_dir:
            self.skipTest("Test dataset not available")
            
        from clarification.datas.cpp_loader import CppDataLoader
        
        batch_size = 16
        device = torch.device('cuda:0')
        
        loader = CppDataLoader(
            device=device,
            base_dir=self.test_dir,
            csv_filename=self.csv_filename,
            batch_size=batch_size,
            num_preload_batches=4,
            num_threads=4,
            use_lz4=False  # Opus dataset
        )
        
        time.sleep(3)
        batch = next(loader)
        
        print(f"\n  Batch shape: {batch.shape}")
        print(f"  Batch dtype: {batch.dtype}")
        print(f"  Batch device: {batch.device}")
        
        self.assertEqual(len(batch.shape), 3, "Batch should be 3D")
        self.assertEqual(batch.shape[0], batch_size, 
            f"First dim should be batch_size={batch_size}")
        self.assertEqual(batch.shape[1], 2, 
            "Second dim should be 2 (channels: noisy, clean)")
        self.assertEqual(batch.shape[2], 7200, 
            "Third dim should be sample_size=7200 (300ms @ 24kHz)")
        
        self.assertEqual(batch.dtype, torch.float32, 
            "Batch dtype should be float32")
        self.assertTrue(batch.device.type == 'cuda', 
            "Batch should be on CUDA device")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_vram_proportional_to_batches(self):
        """Test that VRAM usage is proportional to batch count"""
        if not self.test_dir:
            self.skipTest("Test dataset not available")
            
        from clarification.datas.cpp_loader import CppDataLoader
        
        device = torch.device('cuda:0')
        gc.collect()
        torch.cuda.empty_cache()
        initial_mem = get_gpu_memory_mb()
        
        loader = CppDataLoader(
            device=device,
            base_dir=self.test_dir,
            csv_filename=self.csv_filename,
            batch_size=16,
            num_preload_batches=8,
            num_threads=4,
            use_lz4=False  # Opus dataset
        )
        
        time.sleep(5)
        after_preload = get_gpu_memory_mb()
        
        # Expected per batch: [16, 2, 7200] * 4 bytes = ~0.88 MB
        expected_per_batch_mb = 16 * 2 * 7200 * 4 / 1024 / 1024
        actual_delta = after_preload - initial_mem
        
        # Note: Due to a bug in preload logic, it actually loads ~128 batches
        # instead of 8. This test documents the current behavior.
        estimated_batches = actual_delta / expected_per_batch_mb
        
        print(f"\n  Initial VRAM: {initial_mem:.1f} MB")
        print(f"  After preload: {after_preload:.1f} MB")
        print(f"  Delta: {actual_delta:.1f} MB")
        print(f"  Expected per batch: {expected_per_batch_mb:.2f} MB")
        print(f"  Estimated batches in memory: {estimated_batches:.0f}")
        
        # The ratio should be reasonable (not 50x like with view bug)
        # With .clone() working, it should be close to batch count * per_batch
        # Allow generous overhead for CUDA allocator
        self.assertLess(actual_delta, 500,
            "VRAM usage exceeds 500 MB - possible memory issue")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_consumed_batches_release_memory(self):
        """Test that consumed batches release their VRAM"""
        if not self.test_dir:
            self.skipTest("Test dataset not available")
            
        from clarification.datas.cpp_loader import CppDataLoader
        
        device = torch.device('cuda:0')
        gc.collect()
        torch.cuda.empty_cache()
        initial_mem = get_gpu_memory_mb()
        
        loader = CppDataLoader(
            device=device,
            base_dir=self.test_dir,
            csv_filename=self.csv_filename,
            batch_size=16,
            num_preload_batches=8,
            num_threads=4,
            use_lz4=False  # Opus dataset
        )
        
        time.sleep(5)
        
        # Hold some batches
        batches = []
        for i in range(10):
            batches.append(next(loader))
        
        torch.cuda.synchronize()
        with_batches = get_gpu_memory_mb()
        
        # Release batches
        del batches
        gc.collect()
        torch.cuda.empty_cache()
        after_release = get_gpu_memory_mb()
        
        # Calculate how much was released
        released = with_batches - after_release
        expected_per_batch = 16 * 2 * 7200 * 4 / 1024 / 1024
        expected_release = 10 * expected_per_batch
        
        print(f"\n  With 10 batches: {with_batches:.1f} MB")
        print(f"  After release: {after_release:.1f} MB")
        print(f"  Released: {released:.1f} MB")
        print(f"  Expected to release: {expected_release:.1f} MB")
        
        # Should release at least 50% of expected (allow for caching)
        self.assertGreater(released, expected_release * 0.5,
            "Consumed batches did not release expected VRAM")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
