"""
Python dataset implementation for loading prepared audio data.

Supports both LZ4 (raw compressed fp16) and Opus audio formats.
This provides a Python-only alternative to the C++ data loader for comparison.
"""

import csv
import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)


def load_lz4_file(path: str, sample_size: int) -> torch.Tensor:
    """
    Load LZ4 compressed raw fp16 audio file.
    
    Returns tensor of shape [num_chunks, 2, sample_size] in fp16.
    """
    import lz4.block
    
    with open(path, 'rb') as f:
        compressed_data = f.read()
    
    # Data was compressed with store_size=False, so we need to provide max output size
    # Use same buffer size as C++ loader (80 MiB should be more than enough)
    max_output_size = 80 * 1024 * 1024
    decompressed = lz4.block.decompress(compressed_data, uncompressed_size=max_output_size)
    
    # Convert to numpy fp16 then torch
    audio_array = np.frombuffer(decompressed, dtype=np.float16)
    audio_tensor = torch.from_numpy(audio_array.copy())
    
    # Data is interleaved: [L0, R0, L1, R1, ...]
    num_samples = len(audio_tensor)
    num_stereo_samples = num_samples // 2
    num_chunks = num_stereo_samples // sample_size
    
    if num_chunks == 0:
        return torch.empty((0, 2, sample_size), dtype=torch.float16)
    
    # Truncate to exact multiple
    truncated = num_chunks * sample_size * 2
    audio_tensor = audio_tensor[:truncated]
    
    # Reshape: [N] -> [chunks, sample_size, 2] -> [chunks, 2, sample_size]
    return audio_tensor.view(num_chunks, sample_size, 2).permute(0, 2, 1).contiguous()


def load_opus_file(path: str, sample_size: int, target_sample_rate: int = 24000) -> torch.Tensor:
    """
    Load Opus encoded audio file, resampling to target rate if needed.
    
    Args:
        path: Path to opus file
        sample_size: Number of samples per chunk
        target_sample_rate: Target sample rate (default 24000 Hz)
    
    Returns tensor of shape [num_chunks, 2, sample_size] in fp16.
    """
    import torchaudio
    
    # torchaudio loads as [channels, samples]
    audio, source_rate = torchaudio.load(path, backend="ffmpeg")
    
    # Should be stereo [2, samples]
    if audio.shape[0] != 2:
        raise ValueError(f"Expected stereo audio, got {audio.shape[0]} channels")
    
    # Resample if needed (Opus typically encodes at 48kHz)
    if source_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(source_rate, target_sample_rate)
        audio = resampler(audio)
    
    # Reshape to [num_chunks, 2, sample_size]
    num_samples = audio.shape[1]
    num_chunks = num_samples // sample_size
    
    if num_chunks == 0:
        return torch.empty((0, 2, sample_size), dtype=torch.float16)
    
    # Truncate to exact multiple
    truncated = num_chunks * sample_size
    audio = audio[:, :truncated]
    
    # Reshape: [2, total] -> [2, chunks, sample_size] -> [chunks, 2, sample_size]
    result = audio.view(2, num_chunks, sample_size).permute(1, 0, 2).contiguous()
    
    # Convert to fp16 to match C++ output
    return result.to(torch.float16)


class NoisyCommonsDataset(Dataset):
    """
    PyTorch Dataset for loading prepared noisy/clear audio pairs.
    
    Supports:
    - LZ4 format: Raw fp16 interleaved stereo, LZ4 block compressed
    - Opus format: Opus-encoded stereo audio
    
    Output shape: [num_chunks, 2, sample_size] where:
    - Channel 0 = noisy audio
    - Channel 1 = clear audio
    """
    
    def __init__(self, base_dir: str, csv_filename: str, use_lz4: bool = True):
        super().__init__()
        self.base_dir = base_dir
        self.use_lz4 = use_lz4
        
        # Read dataset info
        info_path = os.path.join(base_dir, "info.csv")
        with open(info_path) as csvfile:
            fieldnames = ['sample_rate', 'sample_size', 'overlap_size']
            reader = csv.DictReader(csvfile, fieldnames=fieldnames)
            next(reader)  # Skip header
            row = next(reader)
            self.sample_rate = int(row['sample_rate'])
            self.sample_size = int(row['sample_size'])
            self.overlap_size = int(row['overlap_size'])
        
        # Read file paths
        csv_path = os.path.join(base_dir, csv_filename)
        with open(csv_path) as csvfile:
            fieldnames = ['path']
            samples_reader = csv.DictReader(csvfile, fieldnames=fieldnames)
            next(samples_reader)  # Skip header
            self.sample_paths = [row['path'] for row in samples_reader]
            
        logger.info(f"Dataset: {len(self.sample_paths)} files, "
                   f"format={'LZ4' if use_lz4 else 'Opus'}, "
                   f"sample_rate={self.sample_rate}, sample_size={self.sample_size}")

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and return audio data for the given index.
        
        Returns:
            Tensor of shape [num_chunks, 2, sample_size] in fp16
        """
        relative_path = self.sample_paths[idx]
        absolute_path = os.path.join(self.base_dir, relative_path)
        
        if self.use_lz4:
            return load_lz4_file(absolute_path, self.sample_size)
        else:
            return load_opus_file(absolute_path, self.sample_size, self.sample_rate)


class PythonDataLoader:
    """
    Python data loader wrapper to match the CppDataLoader interface.
    
    Uses single-threaded loading to avoid multiprocessing pickle issues
    with lz4 module. This is intentional - we want an apples-to-apples
    comparison of the core loading logic, not multiprocessing overhead.
    """
    
    def __init__(self, 
                 device: torch.device,
                 base_dir: str,
                 csv_filename: str,
                 batch_size: int = 16,
                 num_preload_batches: int = 16,
                 num_threads: int = 4,
                 use_lz4: bool = True):
        """
        Initialize the Python data loader.
        
        Args:
            device: Target device for tensors
            base_dir: Directory containing the dataset
            csv_filename: Name of the CSV file listing samples
            batch_size: Number of chunks per batch
            num_preload_batches: Ignored (for API compatibility)
            num_threads: Ignored (single-threaded for pickle safety)
            use_lz4: Whether to use LZ4 format (True) or Opus (False)
        """
        self.device = device
        self.batch_size = batch_size
        self.use_lz4 = use_lz4
        
        # Create dataset
        self.dataset = NoisyCommonsDataset(
            base_dir=base_dir,
            csv_filename=csv_filename,
            use_lz4=use_lz4
        )
        
        self.sample_rate = self.dataset.sample_rate
        self.sample_size = self.dataset.sample_size
        self.overlap_size = self.dataset.overlap_size
        
        self._file_idx = 0
        self._buffer = []  # Buffer for partial batches
    
    def __iter__(self):
        return self
    
    def __next__(self) -> torch.Tensor:
        """
        Get next batch of shape [batch_size, 2, sample_size].
        
        Accumulates chunks from multiple files until batch_size is reached.
        """
        while len(self._buffer) < self.batch_size:
            if self._file_idx >= len(self.dataset):
                # Reset for next epoch
                self._file_idx = 0
                if len(self._buffer) == 0:
                    raise StopIteration
                break
            
            # Load next file
            file_chunks = self.dataset[self._file_idx]  # [num_chunks, 2, sample_size]
            self._file_idx += 1
            
            # Add chunks to buffer
            for i in range(file_chunks.shape[0]):
                self._buffer.append(file_chunks[i])
        
        if len(self._buffer) == 0:
            raise StopIteration
        
        # Take batch_size chunks from buffer
        batch_chunks = self._buffer[:self.batch_size]
        self._buffer = self._buffer[self.batch_size:]
        
        # Stack into batch
        batch = torch.stack(batch_chunks, dim=0)
        
        # Move to target device
        return batch.to(self.device)
    
    def reset(self):
        """Reset the loader to the beginning."""
        self._file_idx = 0
        self._buffer = []
    
    def total_files(self) -> int:
        """Return total number of files in dataset."""
        return len(self.dataset)
