#!/usr/bin/env python3
"""
Performance benchmark for prepare_data.py

Processes 10k files and measures throughput to validate optimizations.

Run with: python -m tests.test_prepare_data_perf
"""

import os
import sys
import time
import shutil
import tempfile
import subprocess
from pathlib import Path

# Test configuration
TEST_LIMIT = 10000
TEST_OUTPUT_DIR = "/workspace/test_datasets/perf_test_output"
LOCALE = "en"


def cleanup_output():
    """Remove test output directory."""
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)


def count_output_files(directory: str) -> int:
    """Count generated output files."""
    count = 0
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith('.opus') or f.endswith('.lz4'):
                count += 1
    return count


def run_prepare_data(format_type: str = "opus", dtype: str = "fp16") -> dict:
    """
    Run prepare_data with given configuration and measure performance.
    
    Returns dict with timing and throughput info.
    """
    cleanup_output()
    
    cmd = [
        sys.executable, "-m", "clarification.datas.prepare_data",
        "--format", format_type,
        "--dtype", dtype,
        "--limit", str(TEST_LIMIT),
        "--locale", LOCALE,
        "--out_dir", TEST_OUTPUT_DIR,
    ]
    
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONUNBUFFERED"] = "1"
    
    print(f"\nRunning: {' '.join(cmd)}")
    print("-" * 60)
    
    start_time = time.time()
    
    result = subprocess.run(
        cmd,
        env=env,
        cwd="/workspace/clarification",
        capture_output=False,  # Show output in real-time
    )
    
    elapsed = time.time() - start_time
    
    # Count output files
    train_files = count_output_files(os.path.join(TEST_OUTPUT_DIR, "train"))
    test_files = count_output_files(os.path.join(TEST_OUTPUT_DIR, "test"))
    total_files = train_files + test_files
    
    # Note: total_files is batches, not individual input files
    # Each batch contains process_batch_size (16) files
    files_processed = min(TEST_LIMIT, total_files * 16)  # Approximate
    
    files_per_sec = files_processed / elapsed if elapsed > 0 else 0
    
    return {
        "format": format_type,
        "dtype": dtype,
        "elapsed_sec": elapsed,
        "output_batches": total_files,
        "files_processed": files_processed,
        "files_per_sec": files_per_sec,
        "exit_code": result.returncode,
    }


def print_results(results: list):
    """Print comparison table of results."""
    print("\n" + "=" * 70)
    print("PERFORMANCE RESULTS")
    print("=" * 70)
    print(f"{'Config':<20} {'Time (s)':<12} {'Files':<12} {'Files/sec':<12} {'Status'}")
    print("-" * 70)
    
    for r in results:
        status = "✓" if r["exit_code"] == 0 else "✗"
        config = f"{r['format']}/{r['dtype']}"
        print(f"{config:<20} {r['elapsed_sec']:<12.1f} {r['files_processed']:<12} "
              f"{r['files_per_sec']:<12.1f} {status}")
    
    print("-" * 70)
    
    if results:
        best = max(results, key=lambda r: r['files_per_sec'])
        print(f"\nBest: {best['format']}/{best['dtype']} @ {best['files_per_sec']:.0f} files/sec")


def run_benchmark():
    """Run the full benchmark suite."""
    print("=" * 70)
    print("PREPARE_DATA PERFORMANCE BENCHMARK")
    print(f"Processing {TEST_LIMIT} files, locale={LOCALE}")
    print("=" * 70)
    
    results = []
    
    # Test Opus format (the main use case)
    print("\n[1/1] Testing Opus format...")
    result = run_prepare_data(format_type="opus", dtype="fp16")
    results.append(result)
    
    print_results(results)
    
    # Cleanup
    cleanup_output()
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark prepare_data performance")
    parser.add_argument("--keep-output", action="store_true", help="Don't delete output after test")
    parser.add_argument("--limit", type=int, default=TEST_LIMIT, help="Number of files to process")
    args = parser.parse_args()
    
    TEST_LIMIT = args.limit
    
    results = run_benchmark()
    
    if not args.keep_output:
        cleanup_output()
    
    # Exit with error if any test failed
    if any(r["exit_code"] != 0 for r in results):
        sys.exit(1)

