"""Runs multiple experiments, managing audio_trainer instances with multiple configurations."""

import datetime
import gc
import torch
from ..util import *
import logging

logger = logging.getLogger(__name__)

from .configs import *
from .audio_trainer import *

class TrainMultiple:
    def __init__(self, config: TrainMultipleConfig):
        self.c = config
        self.audio_trainer_config_to_trainer = {}

        pass

    def run(self):
        clear_cache_and_gc()

        pprint.pprint(self.c, width=2)

        for trainer_config in self.c.trainer_configs:
            model_config = trainer_config.model_training_config
            total_params = sum(p.numel() for p in model_config.model.parameters())
            trainer_config.log_behavior_config.writer.add_text(f"total_params_{model_config.name}", f"{total_params}")
            print(f"total_params_{model_config.name}: {total_params}")

        if self.c.should_perform_memory_test:
            for model_training_config in self.c.trainer_configs:
                self.train_rotation(audio_trainer_config=model_training_config, memory_test_run=True)

        while True:
            # torch.cuda.memory._record_memory_history(max_entries=10000000)
            #
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            #     with record_function(f"profile_full_rotation"):
            #         for model_training_config in self.c.trainer_configs:
            #             self.train_rotation(audio_trainer_config=model_training_config)
            # profiling_file_path = "/workspace/tmpprofile"
            # # Path(profiling_file_path).mkdir(parents=True, exist_ok=True)
            # torch.cuda.memory._dump_snapshot(profiling_file_path)
            # torch.cuda.memory._record_memory_history(enabled=None)
            # print(f"Wrote profiling data to {profiling_file_path}")

            for model_training_config in self.c.trainer_configs:
                self.train_rotation(audio_trainer_config=model_training_config)

        pass

    def train_rotation(self, audio_trainer_config: AudioTrainerConfig, memory_test_run=False):
        # now_str = date_str = datetime.now().strftime('%Y-%m-%d__%H-%M-%S')
        self.config_to_device(audio_trainer_config)
        trainer = self.audio_trainer_for_config(audio_trainer_config)

        if memory_test_run:
            trainer.memory_test_run()
        else:
            trainer.train_one_rotation()

        self.config_to_cpu(config=audio_trainer_config)

        # Cuda memory usage:
        if torch.cuda.is_available():
            print(f"After {audio_trainer_config.model_training_config.name} Memory allocated: {torch.cuda.memory_allocated()}")
            print(f"Max memory allocated: {torch.cuda.max_memory_allocated()}")
            print(f"Memory reserved: {torch.cuda.memory_reserved()}")
            print(f"Max memory reserved: {torch.cuda.max_memory_reserved()}")

        pass

    @staticmethod
    def config_to_device(config: AudioTrainerConfig):
        print(f"Device: {config.device}")

        config.model_training_config.model = config.model_training_config.model.to(config.device)
        for lfc in config.model_training_config.loss_function_configs:
            lfc.fn = lfc.fn.to(config.device)

    @staticmethod
    def config_to_cpu(config: AudioTrainerConfig):
        config.model_training_config.model = config.model_training_config.model.to("cpu")
        for lfc in config.model_training_config.loss_function_configs:
            lfc.fn = lfc.fn.to("cpu")

    def audio_trainer_for_config(self, config: AudioTrainerConfig):
        if config in self.audio_trainer_config_to_trainer:
            return self.audio_trainer_config_to_trainer[config]

        trainer = AudioTrainer(config)
        self.audio_trainer_config_to_trainer[config] = trainer
        return trainer

