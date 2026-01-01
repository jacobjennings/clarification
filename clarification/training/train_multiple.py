"""Runs multiple experiments, managing audio_trainer instances with multiple configurations."""

import logging
import glob
import os

logger = logging.getLogger(__name__)

from .audio_trainer import *

class TrainMultiple:
    def __init__(self, config: TrainMultipleConfig):
        self.c = config
        self.audio_trainer_config_to_trainer = {}

        pass

    # def run(self):
    #     torch.cuda.memory._record_memory_history(max_entries=1000000)
    #
    #     with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    #         with record_function(f"profile_memory_test_rotation"):
    #             self.run2()
    #     profiling_file_path = "/workspace/tmpprofile"
    #     # Path(profiling_file_path).mkdir(parents=True, exist_ok=True)
    #     torch.cuda.memory._dump_snapshot(profiling_file_path)
    #     torch.cuda.memory._record_memory_history(enabled=None)
    #     print(f"Wrote profiling data to {profiling_file_path}")


    def run(self):
        clear_cache_and_gc()

        pprint.pprint(self.c, width=2)

        for trainer_config in self.c.trainer_configs:
            model_config = trainer_config.model_training_config
            total_params = sum(p.numel() for p in model_config.model.parameters())
            trainer_config.log_behavior_config.writer.add_text(f"total_params_{model_config.name}", f"{total_params}")

            print(f"total_params_{model_config.name}: {total_params}")

            # Weight loading logic
            if self.c.auto_resume or self.c.resume_from_run_dir:
                weights_path = self.find_weights(model_config.name, self.c.resume_from_run_dir)
                if weights_path:
                    try:
                        # Load on CPU first, then move to device later in train_rotation
                        model_config.model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
                        print(f"INFO: Loaded weights for {model_config.name} from {weights_path}")
                    except Exception as e:
                        print(f"WARNING: Could not load weights for {model_config.name} from {weights_path}: {e}")
                elif self.c.resume_from_run_dir:
                    print(f"WARNING: No weights found for {model_config.name} in specified run directory: {self.c.resume_from_run_dir}")

        if self.c.should_perform_memory_test:
            for model_training_config in self.c.trainer_configs:
                self.train_rotation(audio_trainer_config=model_training_config, memory_test_run=True)

        while True:
            previous_iterator = None
            for model_training_config in self.c.trainer_configs:
                model_training_config.state.data_loader_iter = previous_iterator
                self.train_rotation(audio_trainer_config=model_training_config)
                previous_iterator = model_training_config.state.data_loader_iter

        pass

    def train_rotation(self, audio_trainer_config: AudioTrainerConfig, memory_test_run=False):
        trainer = self.audio_trainer_for_config(audio_trainer_config)
        self.config_to_device(config=audio_trainer_config)

        if memory_test_run:
            trainer.memory_test_run()
        else:
            trainer.train_one_rotation()

        self.config_to_cpu(config=audio_trainer_config)

        # Cuda memory usage:
        if torch.cuda.is_available():
            print(f"After {audio_trainer_config.model_training_config.name} Memory allocated: {torch.cuda.memory_allocated()}")
            print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024}")
            print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024 / 1024}")
            print(f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1024 / 1024}")

        pass

    @staticmethod
    def config_to_device(config: AudioTrainerConfig):
        print(f"Device: {config.device}")

        config.model_training_config.model = config.model_training_config.model.to(config.device)
        for lfc in config.model_training_config.loss_function_configs:
            lfc.fn = lfc.fn.to(config.device)

        if config.model_training_config.model is not config.model_training_config.model_wrapper:
            config.model_training_config.model_wrapper = config.model_training_config.model_wrapper.to(config.device)

    @staticmethod
    def config_to_cpu(config: AudioTrainerConfig):
        config.model_training_config.model = config.model_training_config.model.to("cpu")
        for lfc in config.model_training_config.loss_function_configs:
            lfc.fn = lfc.fn.to("cpu")

        if config.model_training_config.model is not config.model_training_config.model_wrapper:
            config.model_training_config.model_wrapper = config.model_training_config.model_wrapper.to("cpu")

    def audio_trainer_for_config(self, config: AudioTrainerConfig):
        if config in self.audio_trainer_config_to_trainer:
            return self.audio_trainer_config_to_trainer[config]

        print(f"Creating trainer for config {config.__hash__}")
        trainer = AudioTrainer(config)
        self.audio_trainer_config_to_trainer[config] = trainer
        return trainer

    def find_weights(self, model_name: str, specific_run_dir: Optional[str] = None) -> Optional[str]:
        if specific_run_dir:
            search_pattern = os.path.join(specific_run_dir, "weights", f"weights-*-{model_name}")
            files = glob.glob(search_pattern)
        else:
            search_pattern = os.path.join(runs_dir(), f"*-{model_name}", "weights", f"weights-*-{model_name}")
            files = glob.glob(search_pattern)

        if not files:
            return None

        # Sort by modification time to get the most recent
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]
