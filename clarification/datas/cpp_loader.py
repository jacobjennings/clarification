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
        """Return total number of files in the dataset."""
        if self.loader is None:
            return 0
        return self.loader.total_files()
    
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
