import os
import logging

from datasets import load_dataset, Audio

# Set up log file which should appear in the working directory.
logging.basicConfig(filename="prepare_data.log", encoding="utf-8", level=logging.DEBUG)

training_speech_dataset = load_dataset(
    path="MLCommons/peoples_speech",
    name="clean",
    split="train[:0.5%]",
    data_files="train/clean.json",
    # data_dir=os.path.dirname("/Volumes/bigsmb/peoples_speech_clean_small"),
    cache_dir=os.path.dirname("/Volumes/bigsmb/huggingface-cache/")
)

logging.debug("training_speech_dataset: {}".format(training_speech_dataset))

# dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
# dataset[0]["audio"]
