#!/usr/bin/env python3
"""
Training cycle integration tests.

Tests a complete training loop without polluting production runs
by disabling tensorboard logging.

Run with: python -m tests.test_training_cycle
"""

import unittest
import gc
import os
import sys
import shutil
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import torch
from torch.utils.tensorboard import SummaryWriter

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_config import (
    TEST_OPUS_DIR, TEST_LZ4_DIR, TEST_RUNS_DIR,
    EXPECTED_SAMPLE_SIZE, ensure_test_dirs
)


def dataset_exists(directory: str) -> bool:
    """Check if a dataset directory has data in it."""
    train_csv = os.path.join(directory, "train", "train.csv")
    return os.path.exists(train_csv)


class NullSummaryWriter:
    """A no-op SummaryWriter that doesn't write anything."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def add_scalar(self, *args, **kwargs):
        pass
    
    def add_histogram(self, *args, **kwargs):
        pass
    
    def add_audio(self, *args, **kwargs):
        pass
    
    def add_text(self, *args, **kwargs):
        pass
    
    def add_graph(self, *args, **kwargs):
        pass
    
    def add_image(self, *args, **kwargs):
        pass
    
    def flush(self):
        pass
    
    def close(self):
        pass


class TestTrainingCycle(unittest.TestCase):
    """Integration tests for the training cycle."""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment."""
        cls.cuda_available = torch.cuda.is_available()
        if not cls.cuda_available:
            return
            
        ensure_test_dirs()
        
        # Check for test data
        cls.opus_available = dataset_exists(TEST_OPUS_DIR)
        cls.lz4_available = dataset_exists(TEST_LZ4_DIR)
        
        if not cls.opus_available and not cls.lz4_available:
            print("\n  SKIP: No test data available")
            print("  Run: python -m tests.generate_test_data")
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup test runs directory."""
        if os.path.exists(TEST_RUNS_DIR):
            # Only clean up test-specific subdirs, not the whole thing
            for item in os.listdir(TEST_RUNS_DIR):
                if item.startswith("test_"):
                    shutil.rmtree(os.path.join(TEST_RUNS_DIR, item), ignore_errors=True)
    
    def _create_test_configs(self, use_lz4: bool, dataset_path: str, run_name: str):
        """Create training configs with disabled tensorboard."""
        import clarification as c
        from clarification.util import set_logical_default_device
        
        set_logical_default_device()
        
        # Both loaders output fp16
        torch.set_default_dtype(torch.float16)
        
        # Use small batch sizes for quick tests
        batches_per_iteration = 16
        dataset_batch_size = 8
        
        dataset_config = c.configs.PresetDatasetConfig1(
            batches_per_iteration=batches_per_iteration,
            dataset_batch_size=dataset_batch_size,
        )
        
        dataset_loader = c.configs.PresetCommonVoiceLoader(
            dataset_batch_size=dataset_config.dataset_batch_size,
            batches_per_iteration=dataset_config.batches_per_iteration,
            use_cpp_loader=True,
            dataset_path=dataset_path,
            use_lz4=use_lz4
        )
        dataset_loader.create_loaders()
        
        validation_config = c.configs.PresetValidationConfig1(
            test_batches=10,  # Very small for testing
            run_validation_every_batches=1000,
            log_every_batches=100,
            test_loader=dataset_loader.test_loader,
        )
        
        # Create log config with null writer (no tensorboard)
        log_config = c.configs.LogBehaviorConfig(
            writer=NullSummaryWriter(),
            log_info_every_batches=100,
            model_weights_dir=os.path.join(TEST_RUNS_DIR, run_name, "weights"),
            model_weights_save_every_batches=10000,  # Don't save during short tests
            send_audio_clip_every_batches=10000,  # Don't send during short tests
            profile_every_batches=None,  # Disable profiling
        )
        
        # Use smallest possible model for testing
        model_config = c.configs.SimpleTrainingConfig(
            name=run_name,
            layer_sizes=[16, 16, 16],  # Tiny model
            dataset_config=dataset_config,
            dataset_loader=dataset_loader.train_loader,
            batches_per_iteration=batches_per_iteration,
            batches_per_rotation=100,  # Very short rotations
            training_date_str="test",
            validation_config=validation_config,
        )
        
        trainer_config = c.configs.AudioTrainerConfig(
            model_training_config=model_config,
            log_behavior_config=log_config,
            training_date_str="test",
        )
        
        return trainer_config, dataset_loader
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_training_cycle_opus(self):
        """Test a complete training cycle with Opus loader."""
        if not self.opus_available:
            self.skipTest("Opus test data not available")
        
        self._run_training_cycle(use_lz4=False, dataset_path=TEST_OPUS_DIR, 
                                  run_name="test_opus_training")
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available") 
    def test_training_cycle_lz4(self):
        """Test a complete training cycle with LZ4 loader."""
        if not self.lz4_available:
            self.skipTest("LZ4 test data not available")
        
        self._run_training_cycle(use_lz4=True, dataset_path=TEST_LZ4_DIR,
                                  run_name="test_lz4_training")
    
    def _run_training_cycle(self, use_lz4: bool, dataset_path: str, run_name: str):
        """Execute a training cycle and verify it completes."""
        import clarification as c
        
        format_name = "LZ4" if use_lz4 else "Opus"
        print(f"\n  [{format_name}] Starting training cycle test...")
        
        gc.collect()
        torch.cuda.empty_cache()
        initial_mem = torch.cuda.memory_allocated() / (1024 * 1024)
        
        trainer_config, dataset_loader = self._create_test_configs(
            use_lz4=use_lz4,
            dataset_path=dataset_path,
            run_name=run_name
        )
        
        # Wait for data loader to preload
        time.sleep(3)
        
        # Create trainer
        trainer = c.training.audio_trainer.AudioTrainer(trainer_config)
        
        # Get initial batch to verify data flow
        print(f"  [{format_name}] Getting initial batch...")
        batch = next(iter(dataset_loader.train_loader))
        print(f"  [{format_name}] Initial batch shape: {batch.shape}")
        
        self.assertEqual(len(batch.shape), 3, "Batch should be 3D")
        self.assertEqual(batch.shape[1], 2, "Should have 2 channels")
        self.assertEqual(batch.shape[2], EXPECTED_SAMPLE_SIZE, 
            f"Sample size should be {EXPECTED_SAMPLE_SIZE}")
        
        # Run a few training iterations
        print(f"  [{format_name}] Running training iterations...")
        start_time = time.time()
        
        num_iterations = 3
        for i in range(num_iterations):
            trainer.train_one_rotation()
            print(f"  [{format_name}] Completed rotation {i+1}/{num_iterations}")
        
        elapsed = time.time() - start_time
        
        # Check memory after training
        torch.cuda.synchronize()
        final_mem = torch.cuda.memory_allocated() / (1024 * 1024)
        
        print(f"\n  [{format_name}] Training cycle completed:")
        print(f"    Rotations: {num_iterations}")
        print(f"    Time: {elapsed:.2f}s")
        print(f"    Initial VRAM: {initial_mem:.1f} MB")
        print(f"    Final VRAM: {final_mem:.1f} MB")
        print(f"    Delta: {final_mem - initial_mem:.1f} MB")
        
        # Verify training progressed
        self.assertGreater(trainer.s.batches_trained, 0, 
            "Should have trained some batches")
        
        # Cleanup
        del trainer
        del dataset_loader
        gc.collect()
        torch.cuda.empty_cache()
    
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_no_tensorboard_files_created(self):
        """Verify that no tensorboard files are created in test runs."""
        test_run_dir = os.path.join(TEST_RUNS_DIR, "test_no_tb")
        
        if os.path.exists(test_run_dir):
            shutil.rmtree(test_run_dir)
        
        # Use whichever dataset is available
        if self.lz4_available:
            dataset_path = TEST_LZ4_DIR
            use_lz4 = True
        elif self.opus_available:
            dataset_path = TEST_OPUS_DIR
            use_lz4 = False
        else:
            self.skipTest("No test data available")
        
        import clarification as c
        
        trainer_config, dataset_loader = self._create_test_configs(
            use_lz4=use_lz4,
            dataset_path=dataset_path,
            run_name="test_no_tb"
        )
        
        time.sleep(2)
        
        trainer = c.training.audio_trainer.AudioTrainer(trainer_config)
        trainer.train_one_rotation()
        
        # Check for tensorboard event files
        tb_files = []
        if os.path.exists(test_run_dir):
            for root, dirs, files in os.walk(test_run_dir):
                for f in files:
                    if f.startswith("events.out.tfevents"):
                        tb_files.append(os.path.join(root, f))
        
        self.assertEqual(len(tb_files), 0, 
            f"TensorBoard files should not be created: {tb_files}")
        
        print("\n  PASS: No tensorboard files created in test runs")
        
        # Cleanup
        del trainer
        del dataset_loader
        gc.collect()
        torch.cuda.empty_cache()


# Note: Model forward pass tests are covered by the training cycle tests above


if __name__ == '__main__':
    # Ensure CUDA is initialized
    if torch.cuda.is_available():
        torch.cuda.init()
    
    unittest.main(verbosity=2)

