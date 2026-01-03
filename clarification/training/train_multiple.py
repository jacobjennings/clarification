"""Runs multiple experiments, managing audio_trainer instances with multiple configurations."""

import logging
import glob
import os
import re
import json

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

        # FIRST: Handle resume logic before any logging (to avoid creating unused directories)
        for trainer_config in self.c.trainer_configs:
            model_config = trainer_config.model_training_config

            if self.c.auto_resume or self.c.resume_from_run_dir:
                weights_path = self.find_weights(
                    model_config.name, self.c.resume_from_run_dir
                )
                if weights_path:
                    try:
                        # Load on CPU first, then move to device later in train_rotation
                        model_config.model.load_state_dict(
                            torch.load(
                                weights_path, map_location="cpu", weights_only=True
                            )
                        )
                        print(
                            f"INFO: Loaded weights for {model_config.name} from {weights_path}"
                        )

                        # Extract run directory and step count from weights path
                        # Path format: .../runs/{run_name}/weights/weights-{step}-{model_name}
                        run_dir = self.extract_run_dir(weights_path)
                        step_count = self.extract_step_count(weights_path)

                        if run_dir:
                            # Point writer to the resumed run directory
                            # (writer may be None if using deferred creation)
                            old_writer = trainer_config.log_behavior_config.writer
                            if old_writer:
                                try:
                                    old_writer.close()
                                except:
                                    pass
                            
                            # Create writer pointing to the resumed run directory
                            trainer_config.log_behavior_config.writer = (
                                SummaryWriter(log_dir=run_dir)
                            )
                            trainer_config.log_behavior_config.model_weights_dir = (
                                models_dir(a_runs_dir=run_dir)
                            )
                            trainer_config.log_behavior_config.profiling_data_output_dir = profiling_data_dir(
                                a_runs_dir=run_dir
                            )

                            print(f"INFO: Continuing TensorBoard logging in {run_dir}")

                        if step_count is not None:
                            # Restore state so logging and progress metrics continue correctly
                            trainer_config.state.batches_trained = step_count

                            dataset_config = (
                                trainer_config.model_training_config.dataset_config
                            )
                            dataset_loader = (
                                trainer_config.model_training_config.dataset_loader
                            )
                            total_files = dataset_loader.total_files

                            # Try to load saved training state for accurate resume
                            state_path = self.find_state_file(
                                model_config.name, self.c.resume_from_run_dir
                            )
                            if state_path:
                                try:
                                    with open(state_path, "r", encoding="utf-8") as f:
                                        saved_state = json.load(f)

                                    trainer_config.state.data_loader_file_idx = (
                                        saved_state.get("data_loader_file_idx", 0)
                                    )
                                    trainer_config.state.epoch_count = saved_state.get(
                                        "epoch_count", 0
                                    )
                                    trainer_config.state.samples_processed = (
                                        saved_state.get(
                                            "samples_processed",
                                            step_count
                                            * dataset_config.samples_per_batch,
                                        )
                                    )

                                    print(
                                        f"INFO: Resuming {model_config.name} from step {step_count} "
                                        f"(file {trainer_config.state.data_loader_file_idx}/{total_files}, "
                                        f"epoch {trainer_config.state.epoch_count}) - loaded from state file"
                                    )
                                except Exception as e:
                                    print(
                                        f"WARNING: Could not load state file {state_path}: {e}"
                                    )
                                    state_path = None  # Fall back to estimation

                            if not state_path:
                                # Fall back to estimation if no state file found
                                trainer_config.state.samples_processed = (
                                    step_count * dataset_config.samples_per_batch
                                )

                                # iteration_count is the total number of next(loader) calls across all epochs
                                trainer_config.state.iteration_count = (
                                    step_count // dataset_config.batches_per_iteration
                                )

                                # Estimate file_idx for resume (approximate: assumes ~1 file per iteration)
                                estimated_file_idx = min(
                                    trainer_config.state.iteration_count,
                                    total_files - 1,
                                )
                                trainer_config.state.data_loader_file_idx = (
                                    estimated_file_idx
                                )

                                # Estimate epoch count from file progress
                                trainer_config.state.epoch_count = (
                                    estimated_file_idx // total_files
                                    if total_files > 0
                                    else 0
                                )

                                print(
                                    f"INFO: Resuming {model_config.name} from step {step_count} "
                                    f"(file {estimated_file_idx}/{total_files}, approx epoch {trainer_config.state.epoch_count}) "
                                    f"- estimated (no state file found)"
                                )

                    except Exception as e:
                        print(
                            f"WARNING: Could not load weights for {model_config.name} from {weights_path}: {e}"
                        )
                elif self.c.resume_from_run_dir:
                    print(
                        f"WARNING: No weights found for {model_config.name} in specified run directory: {self.c.resume_from_run_dir}"
                    )

        # THEN: Ensure writers exist and log model info
        # (after resume logic has set up the correct directories)
        for trainer_config in self.c.trainer_configs:
            model_config = trainer_config.model_training_config
            
            # Create writer now if it wasn't created by resume logic
            # This is when we actually create the log directory for fresh runs
            log_config = trainer_config.log_behavior_config
            if hasattr(log_config, 'ensure_writer'):
                log_config.ensure_writer()
            
            total_params = sum(p.numel() for p in model_config.model.parameters())

            # Log as both text and scalar for easy access
            log_config.writer.add_text(
                f"total_params_{model_config.name}", f"{total_params}"
            )
            log_config.writer.add_scalar(
                f"model_params", total_params, 0
            )

            print(f"total_params_{model_config.name}: {total_params}")

        if self.c.should_perform_memory_test:
            for model_training_config in self.c.trainer_configs:
                self.train_rotation(
                    audio_trainer_config=model_training_config, memory_test_run=True
                )

        # Handle uneven resume: if models have different batches_trained counts,
        # skip rotations for models that are ahead until all are synchronized
        models_synchronized = True
        if len(self.c.trainer_configs) > 1:
            batches_per_model = [
                (tc.state.batches_trained, tc.model_training_config.name)
                for tc in self.c.trainer_configs
            ]
            min_batches = min(b for b, _ in batches_per_model)
            max_batches = max(b for b, _ in batches_per_model)
            
            if min_batches != max_batches:
                models_synchronized = False
                print("INFO: Models have different progress after resume:")
                for batches, name in batches_per_model:
                    status = "(behind)" if batches == min_batches else "(ahead)"
                    print(f"  {name}: {batches:,} batches {status}")
                print("INFO: Will skip rotations for ahead models until synchronized")
                print("INFO: Once synchronized, data loader will reset for fair comparison")

        while True:
            # Get shared dataset loader (all models use the same one)
            first_config = self.c.trainer_configs[0]
            dataset_loader = first_config.model_training_config.dataset_loader

            # Fair comparison: all models train on the SAME data each round
            # Save position at start of round to restore for subsequent models
            round_start_file_idx = (
                dataset_loader.file_idx if self.c.fair_comparison_mode else 0
            )

            for i, model_training_config in enumerate(self.c.trainer_configs):
                # Skip models that are ahead until others catch up
                if len(self.c.trainer_configs) > 1 and not models_synchronized:
                    min_batches = min(
                        tc.state.batches_trained for tc in self.c.trainer_configs
                    )
                    max_batches = max(
                        tc.state.batches_trained for tc in self.c.trainer_configs
                    )
                    current_batches = model_training_config.state.batches_trained
                    
                    # Check if we just became synchronized
                    if min_batches == max_batches:
                        models_synchronized = True
                        print("INFO: All models now synchronized! Resetting data loader for fair comparison.")
                        dataset_loader.reset()
                        # Clear all iterators so they pick up from reset position
                        for tc in self.c.trainer_configs:
                            tc.state.data_loader_iter = None
                        round_start_file_idx = 0
                    elif current_batches > min_batches:
                        print(
                            f"Skipping {model_training_config.model_training_config.name} "
                            f"({current_batches:,} batches) - waiting for others to catch up "
                            f"(min: {min_batches:,})"
                        )
                        continue

                if self.c.fair_comparison_mode and i > 0:
                    # Restore loader position so this model sees the same data as the first
                    # Note: This re-reads data from disk (handled efficiently by preloader)
                    # For very large file indices, skip_to_file can be slow (O(N) for C++ loader)
                    print(
                        f"Fair comparison mode: restoring loader to file_idx={round_start_file_idx}"
                    )
                    if round_start_file_idx == 0:
                        dataset_loader.reset()
                    else:
                        dataset_loader.skip_to_file(round_start_file_idx)
                    # Clear the iterator so it picks up from the restored position
                    model_training_config.state.data_loader_iter = None

                self.train_rotation(audio_trainer_config=model_training_config)

        pass

    def train_rotation(
        self, audio_trainer_config: AudioTrainerConfig, memory_test_run=False
    ):
        trainer = self.audio_trainer_for_config(audio_trainer_config)
        self.config_to_device(config=audio_trainer_config)

        if memory_test_run:
            trainer.memory_test_run()
        else:
            trainer.train_one_rotation()

        self.config_to_cpu(config=audio_trainer_config)

        # Cuda memory usage:
        if torch.cuda.is_available():
            print(
                f"After {audio_trainer_config.model_training_config.name} Memory allocated: {torch.cuda.memory_allocated()}"
            )
            print(
                f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024}"
            )
            print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024 / 1024}")
            print(
                f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1024 / 1024}"
            )

        pass

    @staticmethod
    def config_to_device(config: AudioTrainerConfig):
        print(f"Device: {config.device}")

        config.model_training_config.model = config.model_training_config.model.to(
            config.device
        )
        for lfc in config.model_training_config.loss_function_configs:
            lfc.fn = lfc.fn.to(config.device)

        if (
            config.model_training_config.model
            is not config.model_training_config.model_wrapper
        ):
            config.model_training_config.model_wrapper = (
                config.model_training_config.model_wrapper.to(config.device)
            )

    @staticmethod
    def config_to_cpu(config: AudioTrainerConfig):
        config.model_training_config.model = config.model_training_config.model.to(
            "cpu"
        )
        for lfc in config.model_training_config.loss_function_configs:
            lfc.fn = lfc.fn.to("cpu")

        if (
            config.model_training_config.model
            is not config.model_training_config.model_wrapper
        ):
            config.model_training_config.model_wrapper = (
                config.model_training_config.model_wrapper.to("cpu")
            )

    def audio_trainer_for_config(self, config: AudioTrainerConfig):
        if config in self.audio_trainer_config_to_trainer:
            return self.audio_trainer_config_to_trainer[config]

        print(f"Creating trainer for config {config.__hash__}")
        trainer = AudioTrainer(config)
        self.audio_trainer_config_to_trainer[config] = trainer
        return trainer

    def find_weights(
        self, model_name: str, specific_run_dir: Optional[str] = None
    ) -> Optional[str]:
        if specific_run_dir:
            search_pattern = os.path.join(
                specific_run_dir, "weights", f"weights-*-{model_name}"
            )
            files = glob.glob(search_pattern)
        else:
            search_pattern = os.path.join(
                runs_dir(), f"*-{model_name}", "weights", f"weights-*-{model_name}"
            )
            files = glob.glob(search_pattern)

        if not files:
            return None

        # Sort by modification time to get the most recent
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]

    def find_state_file(
        self, model_name: str, specific_run_dir: Optional[str] = None
    ) -> Optional[str]:
        """Find the most recent state file for a model."""
        if specific_run_dir:
            search_pattern = os.path.join(
                specific_run_dir, "weights", f"state-*-{model_name}.json"
            )
            files = glob.glob(search_pattern)
        else:
            search_pattern = os.path.join(
                runs_dir(), f"*-{model_name}", "weights", f"state-*-{model_name}.json"
            )
            files = glob.glob(search_pattern)

        if not files:
            return None

        # Sort by modification time to get the most recent
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]

    @staticmethod
    def extract_run_dir(weights_path: str) -> Optional[str]:
        """Extract the run directory from a weights path.

        Path format: .../runs/{run_name}/weights/weights-{step}-{model_name}
        Returns: .../runs/{run_name}
        """
        # Go up two levels from weights file: weights_file -> weights/ -> run_dir/
        weights_dir = os.path.dirname(weights_path)
        if os.path.basename(weights_dir) == "weights":
            return os.path.dirname(weights_dir)
        return None

    @staticmethod
    def extract_step_count(weights_path: str) -> Optional[int]:
        """Extract the step count from a weights filename.

        Filename format: weights-{step}-{model_name}
        Returns: step count as int, or None if not found
        """
        filename = os.path.basename(weights_path)
        # Match: weights-123456-modelname
        match = re.match(r"weights-(\d+)-", filename)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _remove_empty_run_dir(dir_path: str):
        """Remove a run directory if it's empty or only contains empty subdirectories.

        This is used to clean up directories that were created but never used
        (e.g., when resuming from an existing run).
        """
        try:
            # Check if directory has any actual files (not just empty subdirs)
            has_files = False
            for root, dirs, files in os.walk(dir_path):
                if files:
                    has_files = True
                    break

            if not has_files:
                # Remove empty subdirectories first
                for root, dirs, files in os.walk(dir_path, topdown=False):
                    for d in dirs:
                        subdir = os.path.join(root, d)
                        try:
                            os.rmdir(subdir)
                        except OSError:
                            pass
                # Then remove the main directory
                os.rmdir(dir_path)
                print(f"INFO: Removed unused directory {dir_path}")
        except OSError:
            pass  # Directory not empty or other error, leave it alone
