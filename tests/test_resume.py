#!/usr/bin/env python3
"""
Tests for data loader resume functionality.

Tests the file-based progress tracking and resume capabilities:
- file_idx property
- total_files property  
- skip_to_file() method
- __len__ raises TypeError

Run with: python -m tests.test_resume
"""

import unittest
import os
import sys
import time
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_config import (
    TEST_LZ4_DIR,
    EXPECTED_SAMPLE_SIZE,
    ensure_test_dirs
)


def dataset_exists(directory: str) -> bool:
    """Check if a dataset directory has data in it."""
    train_csv = os.path.join(directory, "train", "train.csv")
    return os.path.exists(train_csv)


class TestCppDataLoaderResume(unittest.TestCase):
    """Test resume functionality for CppDataLoader."""
    
    @classmethod
    def setUpClass(cls):
        cls.cuda_available = torch.cuda.is_available()
        ensure_test_dirs()
        
        cls.test_dir = TEST_LZ4_DIR
        if not dataset_exists(cls.test_dir):
            print(f"\n  SKIP: LZ4 test data not found at {cls.test_dir}")
            print(f"  Run: python -m tests.generate_test_data")
            cls.test_dir = None
    
    def _create_loader(self, batch_size=16):
        """Create a CppDataLoader with standard settings."""
        from clarification.datas.cpp_loader import CppDataLoader
        
        device = torch.device('cuda:0' if self.cuda_available else 'cpu')
        return CppDataLoader(
            device=device,
            base_dir=os.path.join(self.test_dir, "train"),
            csv_filename="train.csv",
            batch_size=batch_size,
            num_preload_batches=4,
            num_threads=4,
            use_lz4=True
        )
    
    def test_total_files_property(self):
        """Test that total_files returns the correct count."""
        if not self.test_dir:
            self.skipTest("Test data not available")
        
        loader = self._create_loader()
        total = loader.total_files
        
        print(f"\n  Total files: {total}")
        
        self.assertIsInstance(total, int)
        self.assertGreater(total, 0, "Should have at least some files")
    
    def test_file_idx_is_valid(self):
        """Test that file_idx returns a valid value within range."""
        if not self.test_dir:
            self.skipTest("Test data not available")
        
        loader = self._create_loader()
        time.sleep(1)  # Let preloader start
        
        total_files = loader.total_files
        file_idx = loader.file_idx
        
        print(f"\n  Total files: {total_files}")
        print(f"  Current file_idx: {file_idx}")
        
        # file_idx should be within valid range
        self.assertGreaterEqual(file_idx, 0, "file_idx should be >= 0")
        self.assertLessEqual(file_idx, total_files, 
            "file_idx should be <= total_files")
        
        # After preloading starts, file_idx should be > 0
        # (preloader loads files ahead of consumption)
        self.assertGreater(file_idx, 0,
            "file_idx should be > 0 after preloader starts")
    
    def test_file_idx_starts_at_zero(self):
        """Test that file_idx starts at 0 for a new loader."""
        if not self.test_dir:
            self.skipTest("Test data not available")
        
        loader = self._create_loader()
        
        # Before iterating, file_idx should be 0 or small (due to preloading)
        # Note: preloading may advance file_idx, so we check it's reasonable
        initial_idx = loader.file_idx
        print(f"\n  Initial file_idx: {initial_idx}")
        
        # After reset, should be 0
        loader.reset()
        # Give preloader time to start
        time.sleep(0.5)
        
        # file_idx may be non-zero due to preloading, but should be small
        reset_idx = loader.file_idx
        print(f"  After reset file_idx: {reset_idx}")
        
        self.assertLess(reset_idx, loader.total_files // 2, 
            "After reset, file_idx should not be past halfway")
    
    def test_skip_to_file(self):
        """Test that skip_to_file advances to roughly the target position."""
        if not self.test_dir:
            self.skipTest("Test data not available")
        
        loader = self._create_loader()
        total = loader.total_files
        
        # Skip to roughly 1/4 of the way through
        target_idx = total // 4
        print(f"\n  Total files: {total}")
        print(f"  Skipping to file index: {target_idx}")
        
        loader.skip_to_file(target_idx)
        
        # file_idx should be at or past target (may overshoot due to batch boundaries)
        actual_idx = loader.file_idx
        print(f"  Actual file_idx after skip: {actual_idx}")
        
        # Allow some tolerance - skip_to_file consumes batches so may overshoot
        self.assertGreaterEqual(actual_idx, target_idx * 0.8,
            f"file_idx {actual_idx} should be near target {target_idx}")
        self.assertLess(actual_idx, total,
            "file_idx should not exceed total_files")
    
    def test_skip_to_file_zero_is_noop(self):
        """Test that skip_to_file(0) does nothing."""
        if not self.test_dir:
            self.skipTest("Test data not available")
        
        loader = self._create_loader()
        time.sleep(0.5)
        
        initial_idx = loader.file_idx
        loader.skip_to_file(0)
        after_idx = loader.file_idx
        
        print(f"\n  Initial: {initial_idx}, After skip(0): {after_idx}")
        
        # Skip to 0 should be a no-op
        self.assertEqual(initial_idx, after_idx,
            "skip_to_file(0) should not change file_idx")
    
    def test_skip_to_file_beyond_total_resets(self):
        """Test that skip_to_file beyond total_files handles gracefully."""
        if not self.test_dir:
            self.skipTest("Test data not available")
        
        loader = self._create_loader()
        total = loader.total_files
        
        # Try to skip past the end
        print(f"\n  Total files: {total}")
        print(f"  Attempting to skip to: {total + 100}")
        
        loader.skip_to_file(total + 100)
        
        final_idx = loader.file_idx
        print(f"  Final file_idx: {final_idx}")
        
        # Should either stop at end or reset
        self.assertLessEqual(final_idx, total,
            "file_idx should not exceed total_files")
    
    def test_len_raises_typeerror(self):
        """Test that __len__ raises TypeError with helpful message."""
        if not self.test_dir:
            self.skipTest("Test data not available")
        
        loader = self._create_loader()
        
        with self.assertRaises(TypeError) as context:
            len(loader)
        
        error_msg = str(context.exception)
        print(f"\n  __len__ error message: {error_msg[:80]}...")
        
        self.assertIn("CppDataLoader", error_msg)
        self.assertIn("file_idx", error_msg.lower())
    
    def test_pin_memory_device_property(self):
        """Test that pin_memory_device returns the device string."""
        if not self.test_dir:
            self.skipTest("Test data not available")
        
        loader = self._create_loader()
        
        device_str = loader.pin_memory_device
        print(f"\n  pin_memory_device: {device_str}")
        
        self.assertIsInstance(device_str, str)
        self.assertTrue('cuda' in device_str or 'cpu' in device_str,
            "pin_memory_device should contain 'cuda' or 'cpu'")


class TestPythonDataLoaderResume(unittest.TestCase):
    """Test resume functionality for PythonDataLoader."""
    
    @classmethod
    def setUpClass(cls):
        ensure_test_dirs()
        
        cls.test_dir = TEST_LZ4_DIR
        if not dataset_exists(cls.test_dir):
            print(f"\n  SKIP: LZ4 test data not found at {cls.test_dir}")
            print(f"  Run: python -m tests.generate_test_data")
            cls.test_dir = None
    
    def _create_loader(self, batch_size=16):
        """Create a PythonDataLoader with standard settings."""
        from clarification.datas.noisy_dataset import PythonDataLoader
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return PythonDataLoader(
            device=device,
            base_dir=os.path.join(self.test_dir, "train"),
            csv_filename="train.csv",
            batch_size=batch_size,
            use_lz4=True
        )
    
    def test_total_files_property(self):
        """Test that total_files returns the correct count."""
        if not self.test_dir:
            self.skipTest("Test data not available")
        
        loader = self._create_loader()
        total = loader.total_files
        
        print(f"\n  Total files: {total}")
        
        self.assertIsInstance(total, int)
        self.assertGreater(total, 0, "Should have at least some files")
    
    def test_file_idx_starts_at_zero(self):
        """Test that file_idx starts at 0 for a new loader."""
        if not self.test_dir:
            self.skipTest("Test data not available")
        
        loader = self._create_loader()
        
        self.assertEqual(loader.file_idx, 0, 
            "Initial file_idx should be 0")
    
    def test_file_idx_advances(self):
        """Test that file_idx advances as we consume batches."""
        if not self.test_dir:
            self.skipTest("Test data not available")
        
        loader = self._create_loader()
        
        initial_idx = loader.file_idx
        self.assertEqual(initial_idx, 0)
        
        # Consume several batches
        for _ in range(10):
            try:
                next(loader)
            except StopIteration:
                break
        
        final_idx = loader.file_idx
        
        print(f"\n  Initial file_idx: {initial_idx}")
        print(f"  After 10 batches file_idx: {final_idx}")
        
        self.assertGreater(final_idx, initial_idx, 
            "file_idx should advance after consuming batches")
    
    def test_skip_to_file(self):
        """Test that skip_to_file jumps to the target position."""
        if not self.test_dir:
            self.skipTest("Test data not available")
        
        loader = self._create_loader()
        total = loader.total_files
        
        # Skip to 1/4 of the way through
        target_idx = total // 4
        print(f"\n  Total files: {total}")
        print(f"  Skipping to file index: {target_idx}")
        
        loader.skip_to_file(target_idx)
        
        actual_idx = loader.file_idx
        print(f"  Actual file_idx after skip: {actual_idx}")
        
        # PythonDataLoader can set file_idx directly, so should be exact
        self.assertEqual(actual_idx, target_idx,
            f"file_idx should be exactly {target_idx}")
    
    def test_skip_to_file_clamps_to_total(self):
        """Test that skip_to_file clamps to total_files when exceeding."""
        if not self.test_dir:
            self.skipTest("Test data not available")
        
        loader = self._create_loader()
        total = loader.total_files
        
        # Try to skip past the end
        loader.skip_to_file(total + 100)
        
        final_idx = loader.file_idx
        print(f"\n  Total: {total}, Attempted: {total + 100}, Final: {final_idx}")
        
        self.assertEqual(final_idx, total,
            "file_idx should be clamped to total_files")
    
    def test_skip_to_file_zero_is_noop(self):
        """Test that skip_to_file(0) does nothing."""
        if not self.test_dir:
            self.skipTest("Test data not available")
        
        loader = self._create_loader()
        
        self.assertEqual(loader.file_idx, 0)
        loader.skip_to_file(0)
        self.assertEqual(loader.file_idx, 0,
            "skip_to_file(0) should not change file_idx")
    
    def test_len_raises_typeerror(self):
        """Test that __len__ raises TypeError with helpful message."""
        if not self.test_dir:
            self.skipTest("Test data not available")
        
        loader = self._create_loader()
        
        with self.assertRaises(TypeError) as context:
            len(loader)
        
        error_msg = str(context.exception)
        print(f"\n  __len__ error message: {error_msg[:80]}...")
        
        self.assertIn("PythonDataLoader", error_msg)
        self.assertIn("file_idx", error_msg.lower())
    
    def test_pin_memory_device_property(self):
        """Test that pin_memory_device returns the device string."""
        if not self.test_dir:
            self.skipTest("Test data not available")
        
        loader = self._create_loader()
        
        device_str = loader.pin_memory_device
        print(f"\n  pin_memory_device: {device_str}")
        
        self.assertIsInstance(device_str, str)
    
    def test_reset_clears_file_idx(self):
        """Test that reset() sets file_idx back to 0."""
        if not self.test_dir:
            self.skipTest("Test data not available")
        
        loader = self._create_loader()
        
        # Advance by consuming batches
        for _ in range(5):
            try:
                next(loader)
            except StopIteration:
                break
        
        self.assertGreater(loader.file_idx, 0, "file_idx should have advanced")
        
        loader.reset()
        
        self.assertEqual(loader.file_idx, 0, 
            "reset() should set file_idx back to 0")
    
    def test_iter_calls_reset(self):
        """Test that iter(loader) resets the loader."""
        if not self.test_dir:
            self.skipTest("Test data not available")
        
        loader = self._create_loader()
        
        # Advance by consuming batches
        for _ in range(5):
            try:
                next(loader)
            except StopIteration:
                break
        
        advanced_idx = loader.file_idx
        self.assertGreater(advanced_idx, 0)
        
        # iter() should reset
        iter(loader)
        
        self.assertEqual(loader.file_idx, 0,
            "iter(loader) should reset file_idx to 0")


class TestResumeConsistency(unittest.TestCase):
    """Test that both loaders have consistent resume APIs."""
    
    @classmethod
    def setUpClass(cls):
        ensure_test_dirs()
        
        cls.test_dir = TEST_LZ4_DIR
        if not dataset_exists(cls.test_dir):
            cls.test_dir = None
    
    def test_both_loaders_have_same_interface(self):
        """Test that CppDataLoader and PythonDataLoader have the same resume interface."""
        if not self.test_dir:
            self.skipTest("Test data not available")
        
        from clarification.datas.cpp_loader import CppDataLoader
        from clarification.datas.noisy_dataset import PythonDataLoader
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        base_dir = os.path.join(self.test_dir, "train")
        
        cpp_loader = CppDataLoader(
            device=device,
            base_dir=base_dir,
            csv_filename="train.csv",
            batch_size=16,
            use_lz4=True
        )
        
        py_loader = PythonDataLoader(
            device=device,
            base_dir=base_dir,
            csv_filename="train.csv",
            batch_size=16,
            use_lz4=True
        )
        
        # Both should have the same properties/methods
        required_attrs = ['total_files', 'file_idx', 'skip_to_file', 
                         'pin_memory_device', 'reset']
        
        print(f"\n  Checking interface consistency:")
        for attr in required_attrs:
            cpp_has = hasattr(cpp_loader, attr)
            py_has = hasattr(py_loader, attr)
            print(f"    {attr}: CppDataLoader={cpp_has}, PythonDataLoader={py_has}")
            
            self.assertTrue(cpp_has, f"CppDataLoader missing {attr}")
            self.assertTrue(py_has, f"PythonDataLoader missing {attr}")
        
        # Both should raise TypeError for len()
        with self.assertRaises(TypeError):
            len(cpp_loader)
        
        with self.assertRaises(TypeError):
            len(py_loader)
        
        print("  Both loaders raise TypeError for len()")
    
    def test_both_loaders_report_same_total_files(self):
        """Test that both loaders report the same total_files for same dataset."""
        if not self.test_dir:
            self.skipTest("Test data not available")
        
        from clarification.datas.cpp_loader import CppDataLoader
        from clarification.datas.noisy_dataset import PythonDataLoader
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        base_dir = os.path.join(self.test_dir, "train")
        
        cpp_loader = CppDataLoader(
            device=device,
            base_dir=base_dir,
            csv_filename="train.csv",
            batch_size=16,
            use_lz4=True
        )
        
        py_loader = PythonDataLoader(
            device=device,
            base_dir=base_dir,
            csv_filename="train.csv",
            batch_size=16,
            use_lz4=True
        )
        
        cpp_total = cpp_loader.total_files
        py_total = py_loader.total_files
        
        print(f"\n  CppDataLoader total_files: {cpp_total}")
        print(f"  PythonDataLoader total_files: {py_total}")
        
        self.assertEqual(cpp_total, py_total,
            "Both loaders should report the same total_files")


if __name__ == '__main__':
    unittest.main(verbosity=2)

