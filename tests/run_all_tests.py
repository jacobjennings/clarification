#!/usr/bin/env python3
"""
Main test runner for clarification unit tests.

This script:
1. Generates test datasets if they don't exist (~5000 samples each)
2. Runs all data loader tests (Opus + LZ4)
3. Runs training cycle tests
4. Reports performance metrics

Usage:
    python -m tests.run_all_tests [--skip-data-gen] [--verbose]
"""

import argparse
import os
import sys
import subprocess
import unittest
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_config import (
    TEST_OPUS_DIR, TEST_LZ4_DIR, TEST_SAMPLE_LIMIT, ensure_test_dirs
)


def dataset_exists(directory: str) -> bool:
    """Check if a dataset directory has data in it."""
    train_csv = os.path.join(directory, "train", "train.csv")
    return os.path.exists(train_csv)


def generate_test_data(force: bool = False) -> bool:
    """Generate test datasets if they don't exist."""
    opus_exists = dataset_exists(TEST_OPUS_DIR)
    lz4_exists = dataset_exists(TEST_LZ4_DIR)
    
    if opus_exists and lz4_exists and not force:
        print("Test datasets already exist. Use --force to regenerate.")
        print(f"  Opus: {TEST_OPUS_DIR}")
        print(f"  LZ4: {TEST_LZ4_DIR}")
        return True
    
    print("\n" + "="*70)
    print("GENERATING TEST DATA")
    print("="*70)
    
    # Generate test data
    cmd = [
        sys.executable, "-m", "tests.generate_test_data"
    ]
    if force:
        cmd.append("--force")
    
    result = subprocess.run(cmd, cwd="/workspace/clarification")
    return result.returncode == 0


def run_loader_tests(verbosity: int = 2) -> unittest.TestResult:
    """Run data loader tests."""
    print("\n" + "="*70)
    print("RUNNING DATA LOADER TESTS")
    print("="*70)
    
    from tests import test_data_loaders
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_data_loaders)
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


def run_training_tests(verbosity: int = 2) -> unittest.TestResult:
    """Run training cycle tests."""
    print("\n" + "="*70)
    print("RUNNING TRAINING CYCLE TESTS")
    print("="*70)
    
    from tests import test_training_cycle
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_training_cycle)
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


def print_summary(results: list):
    """Print test summary."""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    
    for name, result in results:
        tests_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        skipped = len(result.skipped)
        
        total_tests += tests_run
        total_failures += failures
        total_errors += errors
        total_skipped += skipped
        
        status = "PASSED" if failures == 0 and errors == 0 else "FAILED"
        print(f"  {name}: {status} ({tests_run} tests, {failures} failures, {errors} errors, {skipped} skipped)")
    
    print("-"*70)
    print(f"  TOTAL: {total_tests} tests, {total_failures} failures, {total_errors} errors, {total_skipped} skipped")
    
    if total_failures == 0 and total_errors == 0:
        print("\n  ✓ ALL TESTS PASSED")
        return 0
    else:
        print("\n  ✗ SOME TESTS FAILED")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Run all clarification unit tests")
    parser.add_argument("--skip-data-gen", action="store_true",
                        help="Skip test data generation")
    parser.add_argument("--force-data-gen", action="store_true",
                        help="Force regeneration of test data")
    parser.add_argument("--loaders-only", action="store_true",
                        help="Only run data loader tests")
    parser.add_argument("--training-only", action="store_true",
                        help="Only run training cycle tests")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()
    
    ensure_test_dirs()
    
    verbosity = 2 if args.verbose else 1
    
    # Step 1: Generate test data if needed
    if not args.skip_data_gen:
        if not generate_test_data(force=args.force_data_gen):
            print("ERROR: Failed to generate test data")
            return 1
    
    # Check if any test data exists
    opus_exists = dataset_exists(TEST_OPUS_DIR)
    lz4_exists = dataset_exists(TEST_LZ4_DIR)
    
    if not opus_exists and not lz4_exists:
        print("ERROR: No test data available. Run without --skip-data-gen")
        return 1
    
    print("\nTest data status:")
    print(f"  Opus dataset: {'EXISTS' if opus_exists else 'NOT FOUND'}")
    print(f"  LZ4 dataset: {'EXISTS' if lz4_exists else 'NOT FOUND'}")
    
    # Run tests
    results = []
    
    if not args.training_only:
        loader_result = run_loader_tests(verbosity)
        results.append(("Data Loaders", loader_result))
    
    if not args.loaders_only:
        training_result = run_training_tests(verbosity)
        results.append(("Training Cycle", training_result))
    
    # Print summary
    return print_summary(results)


if __name__ == "__main__":
    import torch
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.multiprocessing.set_start_method('spawn', force=True)
    
    sys.exit(main())

