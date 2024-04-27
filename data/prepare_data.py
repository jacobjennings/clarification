import argparse
import os
import logging
import secrets
import numpy
import torch

from datasets import load_dataset, Audio


def add_noise(batch):
    audio = batch['audio']
    noise = numpy.asarray(0.01 * numpy.random.randn(len(audio['array'])))
    audio['array'] = audio['array'] + noise
    return batch


# Parse and define arguments
parser = argparse.ArgumentParser(
                    prog='prepare_data',
                    description='Prepares datasets for clarification training.')

# Shorthand forms omitted intentionally for clarity.
parser.add_argument('--base_dataset_cache_directory', required=True)
parser.add_argument('--modified_dataset_cache_directory', required=True)
args = parser.parse_args()

# Set up log file which should appear in the working directory.
logging.basicConfig(filename="prepare_data.log", encoding="utf-8", level=logging.INFO)

# Additional argument validations
if not args.base_dataset_cache_directory.endswith("/"):
    logging.error("base_dataset_cache_directory must end with a / character.")

if not args.modified_dataset_cache_directory.endswith("/"):
    logging.error("modified_dataset_cache_directory must end with a / character.")

# Load base dataset from cache, or download it (downloads everything)
training_speech_dataset = load_dataset(
    path="fsicoli/common_voice_17_0",
    name="en",
    split="train[:1000]",
    cache_dir=os.path.dirname(args.base_dataset_cache_directory),
    trust_remote_code=True
)

training_speech_audio_dataset = training_speech_dataset.cast_column('audio', Audio(sampling_rate=16000))
training_speech_audio_dataset.map(add_noise)

training_speech_dataset.save_to_disk(os.path.dirname(args.modified_dataset_cache_directory))

logging.info("training_speech_dataset: {}".format(training_speech_dataset))
logging.info("first element: {}".format(training_speech_dataset[0]["audio"]))

dataset = training_speech_dataset.cast_column("audio", Audio(sampling_rate=16000))
logging.info("first element: {}".format(dataset[0]["audio"]))

