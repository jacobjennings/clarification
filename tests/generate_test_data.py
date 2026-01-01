#!/usr/bin/env python3
"""
Generate test datasets for unit tests.

This script generates small datasets (~5000 samples) in both LZ4 and Opus formats
for testing the data loaders. Uses unique paths to avoid polluting production data.

Usage:
    python -m tests.generate_test_data [--force]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_config import (
    TEST_OPUS_DIR, TEST_LZ4_DIR, TEST_SAMPLE_LIMIT, ensure_test_dirs
)


def check_source_data_exists():
    """Check if CommonVoice source data exists."""
    source_dir = "/workspace/cv-20/cv-corpus-20.0-2024-12-06"
    if not os.path.exists(source_dir):
        print(f"ERROR: Source data not found at {source_dir}")
        print("Cannot generate test data without source CommonVoice corpus.")
        return False
    return True


def dataset_exists(directory: str) -> bool:
    """Check if a dataset directory has data in it."""
    train_csv = os.path.join(directory, "train", "train.csv")
    test_csv = os.path.join(directory, "test", "test.csv")
    return os.path.exists(train_csv) and os.path.exists(test_csv)


def generate_dataset(format_type: str, output_dir: str, limit: int) -> bool:
    """
    Generate a dataset using prepare_data.py.
    
    Args:
        format_type: 'opus' or 'lz4'
        output_dir: Output directory
        limit: Number of samples to generate
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Generating {format_type.upper()} dataset with {limit} samples")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    # Determine dtype based on format
    dtype = "fp32" if format_type == "opus" else "fp16"
    
    cmd = [
        sys.executable,
        "-m", "clarification.datas.prepare_data",
        "--format", format_type,
        "--dtype", dtype,
        "--limit", str(limit),
        "--out_dir", output_dir
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd="/workspace/clarification",
            env={**os.environ, "PYTHONPATH": "/workspace/clarification"},
            check=True
        )
        print(f"Successfully generated {format_type.upper()} dataset")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to generate {format_type} dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate test datasets")
    parser.add_argument("--force", action="store_true", 
                        help="Regenerate datasets even if they exist")
    parser.add_argument("--opus-only", action="store_true",
                        help="Only generate Opus dataset")
    parser.add_argument("--lz4-only", action="store_true",
                        help="Only generate LZ4 dataset")
    parser.add_argument("--limit", type=int, default=TEST_SAMPLE_LIMIT,
                        help=f"Number of samples to generate (default: {TEST_SAMPLE_LIMIT})")
    args = parser.parse_args()
    
    ensure_test_dirs()
    
    if not check_source_data_exists():
        return 1
    
    success = True
    
    # Generate Opus dataset
    if not args.lz4_only:
        if args.force or not dataset_exists(TEST_OPUS_DIR):
            if not generate_dataset("opus", TEST_OPUS_DIR, args.limit):
                success = False
        else:
            print(f"Opus dataset already exists at {TEST_OPUS_DIR}")
    
    # Generate LZ4 dataset
    if not args.opus_only:
        if args.force or not dataset_exists(TEST_LZ4_DIR):
            if not generate_dataset("lz4", TEST_LZ4_DIR, args.limit):
                success = False
        else:
            print(f"LZ4 dataset already exists at {TEST_LZ4_DIR}")
    
    if success:
        print("\n" + "="*60)
        print("Test data generation complete!")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("Some datasets failed to generate")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

