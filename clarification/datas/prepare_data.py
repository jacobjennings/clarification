import argparse
import os
import logging
import threading
import itertools
import csv
import sys
csv.field_size_limit(sys.maxsize)
import logging
import lz4.block
import pathlib
import time
from multiprocessing import Pool, Value
import torch
from torch.utils.data import DataLoader
import torchaudio

logger = logging.getLogger(__name__)

base_dataset_directory = '/workspace/cv-20/cv-corpus-20.0-2024-12-06'

num_processed = Value("l", 0)

# Global variables to be set by arguments
OUTPUT_FORMAT = "lz4"  # or "opus"
OUTPUT_DTYPE = torch.float16 # or torch.float32

def discard_remainder(tensors, split_size):
    if tensors[-1].size(0) != split_size:
        tensors = tensors[:-1]
    return tensors

def better_split_discard_remainder(tensor, split_size):
    splits = torch.split(tensor, split_size)

    # Check the size of the last group and discard if necessary
    if splits[-1].size(0) != split_size:
        splits = splits[:-1]

    return splits

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        # Generate noise on CPU (faster than GPU for small tensors with transfer overhead)
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def overlapping_samples(audio, sample_size, overlap_size):
    chunks = audio.unfold(dimension=0, size=sample_size, step=sample_size - overlap_size)
    return chunks


def process_file(data):
    add_noise = AddGaussianNoise(std=0.1)

    (data_batch, batch_idx), megachunk_idx, (data_loader_len, t0, sample_size, overlap_size, resample_rate, out_dataset_directory) = data

    clear_subsamples_aggregate = None
    noisy_subsamples_aggregate = None
    for data in data_batch:
        if num_processed.value % 500 == 0 and num_processed.value != 0:
            t1 = time.time()
            elapsed = t1 - t0
            files_per_second = num_processed.value / elapsed
            print(
                f"Processed {num_processed.value}/{data_loader_len} files "
                f"({num_processed.value / data_loader_len * 100:.0f}%). "
                f"Elapsed: {elapsed:.0f}s. "
                f"Files/second: {files_per_second:.0f}. "
                f"Remaining estimated hours: {(data_loader_len - num_processed.value) / files_per_second / 60 / 60:.2f}")

        num_processed.value += 1

        original_waveform = data[0].squeeze()
        if original_waveform.size()[0] < sample_size:
            return []

        noisy_waveform = add_noise(original_waveform)

        clear_subsamples = overlapping_samples(original_waveform, sample_size=sample_size, overlap_size=overlap_size)
        noisy_subsamples = overlapping_samples(noisy_waveform, sample_size=sample_size, overlap_size=overlap_size)
        if clear_subsamples_aggregate is not None:
            clear_subsamples_aggregate = torch.cat([clear_subsamples_aggregate, clear_subsamples], dim=0)
            noisy_subsamples_aggregate = torch.cat([noisy_subsamples_aggregate, noisy_subsamples], dim=0)
        else:
            clear_subsamples_aggregate = clear_subsamples
            noisy_subsamples_aggregate = noisy_subsamples

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    clips_dir = f"{out_dataset_directory}/{megachunk_idx}/clips"
    pathlib.Path(clips_dir).mkdir(parents=True, exist_ok=True)

    write_dicts = []
    
    noisy_full = clear_subsamples_aggregate.reshape(-1)
    clear_full = noisy_subsamples_aggregate.reshape(-1)

    # Note: permute(1,0) makes it (2, N) -> (N, 2) which is interleaved when flattened or saved as raw
    # But torchaudio.save expects (channels, time) if channels_first=True (default)
    # The original code did:
    # two_channels = torch.stack([noisy_full, clear_full], dim=0).permute(1, 0).to("cpu")
    # which is [samples, channels]
    
    # For Raw/LZ4, we usually want interleaved: L R L R... which corresponds to [samples, channels] flattened
    # For Opus/Torchaudio, we pass [channels, samples] usually or specify channels_first=False
    
    two_channels = torch.stack([noisy_full, clear_full], dim=0).permute(1, 0).cpu()

    if OUTPUT_FORMAT == "lz4":
        relative_path = f"{megachunk_idx}/clips/{batch_idx}.wav.lz4"
        path_absolute = f"{out_dataset_directory}/{megachunk_idx}/clips/{batch_idx}.wav.lz4"
        
        # Convert to target dtype
        two_channels = two_channels.to(OUTPUT_DTYPE)
        
        # Use block compression to match C++ LZ4_decompress_safe
        data = lz4.block.compress(two_channels.numpy().tobytes(), store_size=False)
        with open(path_absolute, 'wb') as f:
            f.write(data)
            
    elif OUTPUT_FORMAT == "opus":
        relative_path = f"{megachunk_idx}/clips/{batch_idx}.opus"
        path_absolute = f"{out_dataset_directory}/{megachunk_idx}/clips/{batch_idx}.opus"
        
        # Opus usually requires float32
        two_channels = two_channels.to(torch.float32)

        torchaudio.save(
            uri=path_absolute,
            src=two_channels,
            sample_rate=resample_rate,
            channels_first=False,  # Input is [samples, channels]
        )
    else:
        raise ValueError(f"Unknown format {OUTPUT_FORMAT}")

    write_dicts.append({
        "path": relative_path,
    })

    return write_dicts


class ChunkIterable:
    def __init__(self, chunk_size, length):
        self.chunk_size = chunk_size
        self.length = length
        self.lock = threading.Lock()
        self.count = 0
        self.chunk_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        chunk_count = 0
        with self.lock:
            self.count += 1
            if self.count == self.chunk_size:
                self.chunk_count += 1
                self.count = 0
            chunk_count = self.chunk_count

        return chunk_count

class EnumeratedIter:
    def __init__(self, it, start_index=0):
        self.it = it
        self.index = start_index

    def __next__(self):
        retval = (next(self.it), self.index)
        self.index += 1
        return retval

    def __iter__(self):
        return self

def process_data(args):
    global OUTPUT_FORMAT, OUTPUT_DTYPE
    OUTPUT_FORMAT = args.format
    if args.dtype == "fp16":
        OUTPUT_DTYPE = torch.float16
    else:
        OUTPUT_DTYPE = torch.float32
        
    train_out_dataset_directory = args.out_dir + "/train"
    test_out_dataset_directory = args.out_dir + "/test"
    
    # Config
    sample_ms = 300
    overlap_ms = 2
    megachunk_size = 1000
    process_batch_size = 16
    process_count = 32

    resample_rate = 24000
    sample_size = int((sample_ms / 1000) * resample_rate)
    overlap_size = int((overlap_ms / 1000) * resample_rate)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchaudio version: {torchaudio.__version__}")
    print(f"Output Format: {OUTPUT_FORMAT}")
    print(f"Output Dtype: {OUTPUT_DTYPE}")
    
    localization_subdirectories = [f.name for f in os.scandir(base_dataset_directory) if f.is_dir()]

    pathlib.Path(train_out_dataset_directory).mkdir(parents=True, exist_ok=True)
    pathlib.Path(test_out_dataset_directory).mkdir(parents=True, exist_ok=True)

    train_datasets = []
    test_datasets = []
    locale_filter = args.locale
    print(f"Locale filter: {locale_filter if locale_filter else 'all locales'}")
    
    for loc_dir in localization_subdirectories:
        localized_input_base_dir = base_dataset_directory + "/" + loc_dir
        
        # Filter by locale if specified
        if locale_filter and locale_filter not in loc_dir:
            continue
        
        print(f"Loading from: {localized_input_base_dir}")

        train_dataset = torchaudio.datasets.COMMONVOICE(root=localized_input_base_dir, tsv="validated.tsv")
        train_datasets.append(train_dataset)
        
        test_dataset = torchaudio.datasets.COMMONVOICE(root=localized_input_base_dir, tsv="test.tsv")
        test_datasets.append(test_dataset)
        
    train_chained_dataset = torch.utils.data.ConcatDataset(train_datasets)
    test_chained_dataset = torch.utils.data.ConcatDataset(test_datasets)
        
    train_data_loader = DataLoader(
        train_chained_dataset,
        batch_size=1,
        num_workers=16)

    test_data_loader = DataLoader(
        test_chained_dataset,
        batch_size=1,
        num_workers=16)

    train_data_loader_len = len(train_data_loader)
    test_data_loader_len = len(test_data_loader)
    
    # Resume logic: Skip already processed batches
    start_batch_idx_train = 0
    start_batch_idx_test = 0
    if args.resume:
        def get_max_batch_idx(directory):
            max_idx = -1
            if not os.path.exists(directory):
                return -1
            for root, dirs, files in os.walk(directory):
                for f in files:
                    if f.endswith(".lz4") or f.endswith(".opus"):
                        try:
                            # Extract idx from 'clips/{idx}.wav.lz4' or similar
                            idx = int(f.split('.')[0])
                            if idx > max_idx:
                                max_idx = idx
                        except (ValueError, IndexError):
                            continue
            return max_idx

        # If batch N exists, we resume from N+1
        max_idx_train = get_max_batch_idx(train_out_dataset_directory)
        max_idx_test = get_max_batch_idx(test_out_dataset_directory)
        
        if max_idx_train >= 0:
            start_batch_idx_train = max_idx_train + 1
            print(f"Resuming train from batch {start_batch_idx_train}")
        if max_idx_test >= 0:
            start_batch_idx_test = max_idx_test + 1
            print(f"Resuming test from batch {start_batch_idx_test}")

    # Apply limit and resume offsets via Subset
    start_sample_train = start_batch_idx_train * process_batch_size
    start_sample_test = start_batch_idx_test * process_batch_size

    def get_subset(dataset, start_sample, limit):
        dataset_len = len(dataset)
        end_sample = dataset_len
        if limit > 0:
            end_sample = min(limit, dataset_len)
        
        if start_sample >= end_sample:
            return None
        return torch.utils.data.Subset(dataset, range(start_sample, end_sample))

    train_subset = get_subset(train_chained_dataset, start_sample_train, args.limit)
    test_subset = get_subset(test_chained_dataset, start_sample_test, args.limit)

    train_data_loader = DataLoader(train_subset, batch_size=1, num_workers=16) if train_subset else []
    test_data_loader = DataLoader(test_subset, batch_size=1, num_workers=16) if test_subset else []

    print(f"Detected {os.cpu_count()} cpus.")
    print(f"Remaining train samples to process: {len(train_subset) if train_subset else 0}")
    print(f"Remaining test samples to process: {len(test_subset) if test_subset else 0}")

    t0 = time.time()

    # Initialize iterators with correct offsets
    train_chunk_iter = ChunkIterable(megachunk_size, len(train_chained_dataset))
    train_chunk_iter.chunk_count = (start_batch_idx_train * process_batch_size) // megachunk_size
    train_chunk_iter.count = (start_batch_idx_train * process_batch_size) % megachunk_size

    test_chunk_iter = ChunkIterable(megachunk_size, len(test_chained_dataset))
    test_chunk_iter.chunk_count = (start_batch_idx_test * process_batch_size) // megachunk_size
    test_chunk_iter.count = (start_batch_idx_test * process_batch_size) % megachunk_size

    def write_info_csv(directory):
        if args.resume and os.path.exists(f"{directory}/info.csv"):
            return # Don't overwrite existing info
        with open(f"{directory}/info.csv", 'w', newline='\n') as csvfile:
            info_field_names = ['sample_rate', 'sample_size', 'overlap_size']
            info_writer = csv.DictWriter(csvfile, fieldnames=info_field_names)
            info_writer.writeheader()
            info_writer.writerow({
                "sample_rate": resample_rate,
                "sample_size": sample_size,
                "overlap_size": overlap_size,
            })
            
    write_info_csv(train_out_dataset_directory)
    write_info_csv(test_out_dataset_directory)

    # Use EnumeratedIter with custom start index for CSV paths
    train_iter = itertools.batched(train_data_loader, process_batch_size)
    train_enumerated_batched_iter = EnumeratedIter(train_iter, start_index=start_batch_idx_train)
    train_zipped_dataset = zip(
        train_enumerated_batched_iter, train_chunk_iter, itertools.repeat((len(train_chained_dataset), t0, sample_size, overlap_size, resample_rate, train_out_dataset_directory)))

    test_iter = itertools.batched(test_data_loader, process_batch_size)
    test_enumerated_batched_iter = EnumeratedIter(test_iter, start_index=start_batch_idx_test)
    test_zipped_dataset = zip(
        test_enumerated_batched_iter, test_chunk_iter, itertools.repeat((len(test_chained_dataset), t0, sample_size, overlap_size, resample_rate, test_out_dataset_directory)))

    # Reset global counter before each process_dataset call
    # Note: num_processed is for progress reporting, not resumption logic.
    num_processed.value = start_sample_train
    train_write_path = f"{train_out_dataset_directory}/train.csv"
    process_dataset(train_write_path, train_zipped_dataset, process_count, mode='a' if args.resume else 'w')

    # Reset global counter before each process_dataset call
    num_processed.value = start_sample_test
    test_write_path = f"{test_out_dataset_directory}/test.csv"
    process_dataset(test_write_path, test_zipped_dataset, process_count, mode='a' if args.resume else 'w')


def process_dataset(write_path, train_zipped_dataset, process_count, mode='w'):
    file_exists = os.path.exists(write_path)
    with open(write_path, mode, newline='\n') as csvfile:
        fieldnames = ['path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not (mode == 'a' and file_exists):
            writer.writeheader()

        with Pool(processes=process_count) as pool:
            for write_dicts in pool.imap_unordered(process_file, train_zipped_dataset, chunksize=4):
                for write_dict in write_dicts:
                    writer.writerow(write_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare audio dataset.")
    parser.add_argument("--format", type=str, default="lz4", choices=["lz4", "opus"], help="Output format: lz4 or opus")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"], help="Data type: fp16 or fp32")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0 for all)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from where it left off based on output directory content")
    parser.add_argument("--locale", type=str, default=None, help="Filter by locale (e.g., 'en' for English only). If not set, uses all locales.")

    args = parser.parse_args()
    
    # Examples:
    # python -m clarification.datas.prepare_data --format lz4 --dtype fp16 --limit 1000 --out_dir /workspace/test_data
    # python -m clarification.datas.prepare_data --format opus --dtype fp32 --out_dir /workspace/full_opus_data
    # python -m clarification.datas.prepare_data --format opus --locale en --out_dir /workspace/english_opus_data
    
    process_data(args)
