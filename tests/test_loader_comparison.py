#!/usr/bin/env python3
"""
Data Loader Performance Comparison: C++ vs Python, LZ4 vs Opus, Thread Scaling

This test provides a comprehensive comparison across:
- Loader type: C++, Python
- Format: LZ4, Opus
- Thread count: (CPUs - 2), (CPUs * 2)

Run with: python -m tests.test_loader_comparison
"""

import gc
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_config import (
    TEST_OPUS_DIR, TEST_LZ4_DIR,
    EXPECTED_SAMPLE_SIZE, EXPECTED_CHANNELS
)


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0


def dataset_exists(directory: str) -> bool:
    """Check if a dataset directory has data in it."""
    if not directory or not os.path.exists(directory):
        return False
    train_csv = os.path.join(directory, "train", "train.csv")
    return os.path.exists(train_csv)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    loader_type: str  # "C++" or "Python"
    format_type: str  # "LZ4" or "Opus"
    num_threads: int
    samples_per_sec: float
    batches_per_sec: float
    time_to_first_batch_ms: float
    
    @property
    def name(self) -> str:
        return f"{self.loader_type} {self.format_type}"


def benchmark_cpp_loader(base_dir: str, use_lz4: bool, 
                         num_threads: int,
                         batch_size: int = 32, 
                         num_batches: int = 100,
                         warmup_batches: int = 10) -> BenchmarkResult:
    """Benchmark the C++ data loader."""
    from clarification.datas.cpp_loader import CppDataLoader
    
    format_type = "LZ4" if use_lz4 else "Opus"
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Create loader
    loader = CppDataLoader(
        device=torch.device('cuda:0'),
        base_dir=os.path.join(base_dir, "train"),
        csv_filename="train.csv",
        batch_size=batch_size,
        num_preload_batches=16,
        num_threads=num_threads,
        use_lz4=use_lz4
    )
    
    # Wait for preload
    time.sleep(1)
    
    # Time to first batch
    start = time.perf_counter()
    first_batch = next(loader)
    time_to_first = (time.perf_counter() - start) * 1000
    
    # Warmup
    for _ in range(warmup_batches - 1):
        next(loader)
    
    # Benchmark
    gc.collect()
    torch.cuda.empty_cache()
    
    batches_processed = 0
    samples_processed = 0
    
    start = time.perf_counter()
    for i in range(num_batches):
        batch = next(loader)
        batches_processed += 1
        samples_processed += batch.shape[0]
    
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    
    samples_per_sec = samples_processed / total_time if total_time > 0 else 0
    batches_per_sec = batches_processed / total_time if total_time > 0 else 0
    
    del loader
    gc.collect()
    torch.cuda.empty_cache()
    
    return BenchmarkResult(
        loader_type="C++",
        format_type=format_type,
        num_threads=num_threads,
        samples_per_sec=samples_per_sec,
        batches_per_sec=batches_per_sec,
        time_to_first_batch_ms=time_to_first,
    )


def benchmark_python_loader(base_dir: str, use_lz4: bool,
                            batch_size: int = 32,
                            num_batches: int = 100,
                            warmup_batches: int = 10) -> BenchmarkResult:
    """Benchmark the Python data loader (single-threaded)."""
    from clarification.datas.noisy_dataset import PythonDataLoader
    
    format_type = "LZ4" if use_lz4 else "Opus"
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Create loader
    loader = PythonDataLoader(
        device=torch.device('cuda:0'),
        base_dir=os.path.join(base_dir, "train"),
        csv_filename="train.csv",
        batch_size=batch_size,
        use_lz4=use_lz4
    )
    
    # Time to first batch
    start = time.perf_counter()
    first_batch = next(loader)
    time_to_first = (time.perf_counter() - start) * 1000
    
    # Warmup
    for _ in range(warmup_batches - 1):
        try:
            next(loader)
        except StopIteration:
            loader.reset()
            next(loader)
    
    # Benchmark
    gc.collect()
    torch.cuda.empty_cache()
    
    batches_processed = 0
    samples_processed = 0
    
    start = time.perf_counter()
    for i in range(num_batches):
        try:
            batch = next(loader)
        except StopIteration:
            loader.reset()
            batch = next(loader)
        batches_processed += 1
        samples_processed += batch.shape[0]
    
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    
    samples_per_sec = samples_processed / total_time if total_time > 0 else 0
    batches_per_sec = batches_processed / total_time if total_time > 0 else 0
    
    del loader
    gc.collect()
    torch.cuda.empty_cache()
    
    return BenchmarkResult(
        loader_type="Python",
        format_type=format_type,
        num_threads=1,  # Python loader is single-threaded
        samples_per_sec=samples_per_sec,
        batches_per_sec=batches_per_sec,
        time_to_first_batch_ms=time_to_first,
    )


def print_3d_comparison_table(results: List[BenchmarkResult], cpu_count: int):
    """Print a 3D comparison table: Loader × Format × Threads."""
    
    print("\n" + "="*100)
    print(f"PERFORMANCE COMPARISON (CPU count: {cpu_count})")
    print("="*100)
    
    # Group results
    cpp_results = [r for r in results if r.loader_type == "C++"]
    py_results = [r for r in results if r.loader_type == "Python"]
    
    # Find best for relative comparison
    best = max(results, key=lambda r: r.samples_per_sec)
    
    # Print C++ results by thread count
    thread_counts = sorted(set(r.num_threads for r in cpp_results))
    
    print(f"\n{'':=<100}")
    print(f"{'C++ LOADER RESULTS':^100}")
    print(f"{'':=<100}")
    print(f"{'Threads':<10} {'Format':<8} {'Samples/sec':>14} {'Relative':>10} {'Batches/sec':>14} {'First Batch':>14}")
    print("-"*100)
    
    for threads in thread_counts:
        for fmt in ["LZ4", "Opus"]:
            r = next((x for x in cpp_results if x.num_threads == threads and x.format_type == fmt), None)
            if r:
                rel = r.samples_per_sec / best.samples_per_sec
                rel_str = "(best)" if rel >= 0.999 else f"({rel:.0%})"
                print(f"{threads:<10} {fmt:<8} {r.samples_per_sec:>14,.0f} {rel_str:>10} "
                      f"{r.batches_per_sec:>12,.1f}/s {r.time_to_first_batch_ms:>12.1f}ms")
    
    # Print Python results (single row per format since it's single-threaded)
    print(f"\n{'':=<100}")
    print(f"{'PYTHON LOADER RESULTS (single-threaded)':^100}")
    print(f"{'':=<100}")
    print(f"{'Format':<10} {'Samples/sec':>14} {'Relative':>10} {'Batches/sec':>14} {'First Batch':>14}")
    print("-"*100)
    
    for fmt in ["LZ4", "Opus"]:
        r = next((x for x in py_results if x.format_type == fmt), None)
        if r:
            rel = r.samples_per_sec / best.samples_per_sec
            rel_str = f"({rel:.0%})"
            print(f"{fmt:<10} {r.samples_per_sec:>14,.0f} {rel_str:>10} "
                  f"{r.batches_per_sec:>12,.1f}/s {r.time_to_first_batch_ms:>12.1f}ms")
    
    # Summary table
    print(f"\n{'':=<100}")
    print(f"{'SUMMARY: C++ SPEEDUP vs PYTHON':^100}")
    print(f"{'':=<100}")
    print(f"{'Config':<25} {'C++ (samples/s)':>18} {'Python (samples/s)':>18} {'Speedup':>12}")
    print("-"*100)
    
    for threads in thread_counts:
        for fmt in ["LZ4", "Opus"]:
            cpp_r = next((x for x in cpp_results if x.num_threads == threads and x.format_type == fmt), None)
            py_r = next((x for x in py_results if x.format_type == fmt), None)
            if cpp_r and py_r:
                speedup = cpp_r.samples_per_sec / py_r.samples_per_sec if py_r.samples_per_sec > 0 else 0
                print(f"C++ {fmt} @ {threads}t vs Py {fmt:<8} {cpp_r.samples_per_sec:>14,.0f} "
                      f"{py_r.samples_per_sec:>14,.0f} {speedup:>10.1f}x")
    
    # Thread scaling comparison
    print(f"\n{'':=<100}")
    print(f"{'THREAD SCALING (C++ only)':^100}")
    print(f"{'':=<100}")
    
    if len(thread_counts) >= 2:
        t_low, t_high = thread_counts[0], thread_counts[-1]
        for fmt in ["LZ4", "Opus"]:
            r_low = next((x for x in cpp_results if x.num_threads == t_low and x.format_type == fmt), None)
            r_high = next((x for x in cpp_results if x.num_threads == t_high and x.format_type == fmt), None)
            if r_low and r_high:
                scaling = r_high.samples_per_sec / r_low.samples_per_sec if r_low.samples_per_sec > 0 else 0
                thread_ratio = t_high / t_low
                efficiency = (scaling / thread_ratio) * 100
                print(f"  {fmt}: {t_low}→{t_high} threads = {scaling:.2f}x throughput "
                      f"(thread efficiency: {efficiency:.0f}%)")
    
    # Best configuration
    print(f"\n{'':=<100}")
    print(f"{'RECOMMENDATION':^100}")
    print(f"{'':=<100}")
    print(f"  Best: {best.loader_type} {best.format_type} @ {best.num_threads} threads")
    print(f"  Throughput: {best.samples_per_sec:,.0f} samples/sec")


def run_full_comparison():
    """Run the full 3D comparison."""
    
    cpu_count = os.cpu_count() or 4
    threads_low = max(1, cpu_count - 2)
    threads_high = cpu_count * 2
    
    print("="*100)
    print("DATA LOADER PERFORMANCE COMPARISON")
    print("Dimensions: Loader (C++/Python) × Format (LZ4/Opus) × Threads")
    print("="*100)
    
    if torch.cuda.is_available():
        torch.cuda.init()
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\nWARNING: CUDA not available")
        return
    
    print(f"CPU count: {cpu_count}")
    print(f"Thread configs: {threads_low} (cpus-2), {threads_high} (cpus×2)")
    
    lz4_available = dataset_exists(TEST_LZ4_DIR)
    opus_available = dataset_exists(TEST_OPUS_DIR)
    
    if not lz4_available:
        print(f"\nSKIP: LZ4 dataset not found at {TEST_LZ4_DIR}")
    if not opus_available:
        print(f"\nSKIP: Opus dataset not found at {TEST_OPUS_DIR}")
    
    if not lz4_available and not opus_available:
        print("Run: python -m tests.generate_test_data")
        return
    
    results = []
    batch_size = 32
    num_batches = 100
    
    test_num = 1
    total_tests = 0
    if lz4_available:
        total_tests += 3  # C++ low, C++ high, Python
    if opus_available:
        total_tests += 3
    
    # C++ LZ4 tests
    if lz4_available:
        for threads in [threads_low, threads_high]:
            print(f"\n[{test_num}/{total_tests}] C++ LZ4 @ {threads} threads...")
            result = benchmark_cpp_loader(TEST_LZ4_DIR, use_lz4=True,
                                          num_threads=threads,
                                          batch_size=batch_size, 
                                          num_batches=num_batches)
            results.append(result)
            print(f"         {result.samples_per_sec:,.0f} samples/sec")
            test_num += 1
        
        # Python LZ4
        print(f"\n[{test_num}/{total_tests}] Python LZ4 (single-threaded)...")
        result = benchmark_python_loader(TEST_LZ4_DIR, use_lz4=True,
                                         batch_size=batch_size,
                                         num_batches=num_batches)
        results.append(result)
        print(f"         {result.samples_per_sec:,.0f} samples/sec")
        test_num += 1
    
    # C++ Opus tests
    if opus_available:
        for threads in [threads_low, threads_high]:
            print(f"\n[{test_num}/{total_tests}] C++ Opus @ {threads} threads...")
            result = benchmark_cpp_loader(TEST_OPUS_DIR, use_lz4=False,
                                          num_threads=threads,
                                          batch_size=batch_size,
                                          num_batches=num_batches)
            results.append(result)
            print(f"         {result.samples_per_sec:,.0f} samples/sec")
            test_num += 1
        
        # Python Opus
        print(f"\n[{test_num}/{total_tests}] Python Opus (single-threaded)...")
        result = benchmark_python_loader(TEST_OPUS_DIR, use_lz4=False,
                                         batch_size=batch_size,
                                         num_batches=num_batches)
        results.append(result)
        print(f"         {result.samples_per_sec:,.0f} samples/sec")
        test_num += 1
    
    # Print comparison table
    print_3d_comparison_table(results, cpu_count)


if __name__ == '__main__':
    run_full_comparison()
