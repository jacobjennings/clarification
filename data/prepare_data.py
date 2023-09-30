import argparse
import os
import logging

from datasets import load_dataset, Audio

# Parse and define arguments
parser = argparse.ArgumentParser(
                    prog='prepare_data',
                    description='Prepares datasets for clarification training.')

# Shorthand forms omitted intentionally for clarity.
parser.add_argument('base_dataset_cache_directory', required=True)
parser.add_argument('modified_dataset_cache_directory', required=True)
args = parser.parse_args()

# Set up log file which should appear in the working directory.
logging.basicConfig(filename="prepare_data.log", encoding="utf-8", level=logging.DEBUG)

if not args.base_dataset_cache_directory.endswith("/"):
    logging.error("base_dataset_cache_directory must end with a / character.")

training_speech_dataset = load_dataset(
    path="MLCommons/peoples_speech",
    name="clean",
    split="train[:1%]",
    data_files="train/clean.json",
    # data_dir=os.path.dirname("/media/bigsmb/peoples_speech_clean_small"),
    cache_dir=os.path.dirname(args.base_dataset_cache_directory)
)

training_speech_dataset.save_to_disk(os.path.dirname(args.modified_dataset_cache_directory))

logging.debug("training_speech_dataset: {}".format(training_speech_dataset))

# dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
# dataset[0]["audio"]
