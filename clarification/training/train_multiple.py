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



        for model_training_config in self.c.trainer_configs:
            self.train_rotation(audio_trainer_config=model_training_config, memory_test_run=True)

        while True:
            for model_training_config in self.c.trainer_configs:
                self.train_rotation(audio_trainer_config=model_training_config)

        # for model_config in models:
        #     total_params = sum(p.numel() for p in model_config.model.parameters())
        #     summary_writer.add_text(f"total_params_{model_config.name}", f"{total_params}")
        #     print(f"total_params_{model_config.name}: {total_params}")



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

        pass

    @staticmethod
    def config_to_device(config: AudioTrainerConfig):
        config.model_training_config.model = config.model_training_config.model.to(config.device)
        for lfc in config.model_training_config.loss_function_configs:
            lfc.loss_function = lfc.loss_function.to(config.device)

    @staticmethod
    def config_to_cpu(self, config: AudioTrainerConfig):
        config.model_training_config.model = config.model_training_config.model.to("cpu")
        for lfc in config.model_training_config.loss_function_configs:
            lfc.loss_function = lfc.loss_function.to("cpu")

    def audio_trainer_for_config(self, config: AudioTrainerConfig):
        if config in self.audio_trainer_config_to_trainer:
            return self.audio_trainer_config_to_trainer[config]

        trainer = AudioTrainer(config)
        self.audio_trainer_config_to_trainer[config] = trainer
        return trainer

