"""Test configuration constants."""

import os

# Test dataset paths (isolated from production datasets)
TEST_DATA_BASE = "/workspace/test_datasets"
TEST_OPUS_DIR = f"{TEST_DATA_BASE}/noisy-commonvoice-24k-300ms-5ms-opus_test"
TEST_LZ4_DIR = f"{TEST_DATA_BASE}/noisy-commonvoice-24k-300ms-5ms-lz4_test"

# Number of samples to generate for tests
TEST_SAMPLE_LIMIT = 5000

# Test run directory (isolated from production runs)
TEST_RUNS_DIR = "/workspace/test_runs"

# Expected audio properties
EXPECTED_SAMPLE_RATE = 24000
EXPECTED_SAMPLE_SIZE = 7200  # 300ms @ 24kHz
EXPECTED_CHANNELS = 2


def ensure_test_dirs():
    """Create test directories if they don't exist."""
    os.makedirs(TEST_DATA_BASE, exist_ok=True)
    os.makedirs(TEST_RUNS_DIR, exist_ok=True)

