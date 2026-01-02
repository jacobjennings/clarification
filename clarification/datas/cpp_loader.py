import atexit
import signal
import sys
import torch

# Global registry of active loaders for cleanup
_active_loaders = []


def _cleanup_loaders():
    """Clean up all active loaders on exit."""
    for loader in _active_loaders:
        try:
            if hasattr(loader, 'loader') and loader.loader is not None:
                del loader.loader
                loader.loader = None
        except:
            pass
    _active_loaders.clear()


def _signal_handler(signum, frame):
    """Handle SIGINT by cleaning up loaders first."""
    print("\nInterrupted - cleaning up data loaders...")
    _cleanup_loaders()
    sys.exit(1)


# Register cleanup handlers
atexit.register(_cleanup_loaders)
signal.signal(signal.SIGINT, _signal_handler)


class CppDataLoader:
    """
    Python wrapper for the C++ audio dataset loaders.
    Supports both LZ4 compressed raw audio and Opus encoded audio.
    
    Note: This loader does not support __len__ because the number of iterations
    depends on the variable number of audio chunks per file, which is not known
    without reading all files. Use file_idx for progress tracking and resuming.
    """
    
    def __init__(
        self, 
        device: torch.device, 
        base_dir: str, 
        csv_filename: str, 
        num_preload_batches: int = 16, 
        batch_size: int = 16, 
        num_threads: int = 8,
        use_lz4: bool = True
    ):
        """
        Initialize the C++ data loader.
        
        Args:
            device: Target device for tensors (e.g., torch.device('cuda:0'))
            base_dir: Path to dataset directory containing info.csv and audio files
            csv_filename: Name of CSV file listing audio file paths (e.g., 'train.csv')
            num_preload_batches: Number of batches to preload in background
            batch_size: Number of samples per batch
            num_threads: Number of threads for parallel file loading
            use_lz4: If True, use LZ4 loader; if False, use Opus loader
        """
        import clarification.datas.clarification_cpp as cpp_ext
        
        self.device = device
        self.use_lz4 = use_lz4
        
        if use_lz4:
            self.loader = cpp_ext.ClarificationLz4Dataset(
                device, 
                base_dir, 
                csv_filename, 
                num_preload_batches, 
                batch_size, 
                num_threads
            )
        else:
            self.loader = cpp_ext.ClarificationOpusDataset(
                device, 
                base_dir, 
                csv_filename, 
                num_preload_batches, 
                batch_size, 
                num_threads
            )
        
        # Register for cleanup
        _active_loaders.append(self)
            
    def __del__(self):
        """Clean up on deletion."""
        self.stop()
        if self in _active_loaders:
            _active_loaders.remove(self)
    
    def stop(self):
        """Stop the loader and release resources."""
        if hasattr(self, 'loader') and self.loader is not None:
            del self.loader
            self.loader = None
            
    def __iter__(self):
        if self.loader is None:
            raise RuntimeError("Loader has been stopped")
        self.loader.reset()
        return self
        
    def __next__(self):
        if self.loader is None:
            raise StopIteration
        try:
            return self.loader.next()
        except (RuntimeError, IndexError) as e:
            if "End of dataset reached" in str(e):
                raise StopIteration
            raise e

    def reset(self):
        """Reset the loader to the beginning of the dataset."""
        if self.loader is not None:
            self.loader.reset()

    def __len__(self):
        """
        Not supported - chunk count per file varies, so total iterations unknown.
        Use total_files and file_idx for progress tracking instead.
        """
        raise TypeError(
            "CppDataLoader does not support __len__ because the number of iterations "
            "depends on variable chunk counts per file. Use total_files and file_idx "
            "for progress tracking."
        )
    
    @property
    def total_files(self) -> int:
        """Return total number of files in the dataset."""
        if self.loader is None:
            return 0
        return self.loader.total_files()
    
    @property
    def file_idx(self) -> int:
        """Return current file index (for progress tracking and resume)."""
        if self.loader is None:
            return 0
        return self.loader.file_idx
    
    def skip_to_file(self, target_file_idx: int):
        """
        Skip forward to a specific file index for resuming training.
        
        Note: This resets the loader and fast-forwards by consuming batches
        until the target file index is reached. Some overshoot may occur
        due to preloading.
        
        Args:
            target_file_idx: The file index to skip to (0-based)
        """
        if self.loader is None:
            return
        
        if target_file_idx <= 0:
            return
            
        # Reset to start
        self.loader.reset()
        
        # Consume batches until we reach or pass the target file index
        # The C++ loader's file_idx tracks which file it's currently loading
        while self.loader.file_idx < target_file_idx:
            try:
                # Consume a batch to advance the file index
                _ = self.loader.next()
            except (RuntimeError, IndexError) as e:
                if "End of dataset reached" in str(e):
                    # Reached end of dataset, reset and stop
                    self.loader.reset()
                    break
                raise e
        
        print(f"Skipped to file index {self.loader.file_idx} (target was {target_file_idx})")
    
    @property
    def sample_size(self):
        """Return the sample size (samples per chunk)."""
        if self.loader is None:
            return 0
        return self.loader.sample_size
    
    @property
    def sample_rate(self):
        """Return the sample rate in Hz."""
        if self.loader is None:
            return 0
        return self.loader.sample_rate
    
    @property
    def overlap_size(self):
        """Return the overlap size in samples (read from dataset's info.csv)."""
        if self.loader is None:
            return 0
        return self.loader.overlap_size
    
    @property
    def pin_memory_device(self):
        """Return device string for compatibility with PyTorch DataLoader."""
        return str(self.device)
