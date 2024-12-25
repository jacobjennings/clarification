import gc
import time

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

@dataclass
class AudioModelTrainingConfig:
    name: str
    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: Optional[optim.lr_scheduler.LRScheduler]
    batches_per_model_rotation: int
    
@dataclass
class AudioLossFunctionConfig:
    name: str
    weight: float
    fn: nn.Module
    is_unary: bool
    batch_size: Optional[int]
    
@dataclass
class ValidationConfig:
    test_loader: DataLoader
    test_batches: int
    run_validation_every_batches: int

class AudioTrainer:
    def __init__(
            self,
            input_dataset_loader: DataLoader,
            models: List[AudioModelTrainingConfig],
            loss_function_configs: List[AudioLossFunctionConfig],
            sample_rate: int,
            samples_per_batch: int,
            batches_per_iteration: int,
            dataset_batch_size: int,
            device: str,
            overlap_samples: int,
            model_weights_dir: Optional[str] = None,
            model_weights_save_every_iterations: int = None,
            summary_writer: Optional[SummaryWriter] = None,
            send_audio_clip_every_iterations: int = 2000,
            dataset_batches_length: int = None,
            training_classifier: bool = False,
            validation_config: Optional[ValidationConfig] = None
    ):
        """Training loop for audio.
        
        Args:
            input_dataset_loader: DataLoader.
            models: List of AudioModelTrainingConfig.
            loss_function_configs: List of AudioLossFunctionConfig.
            sample_rate: Sample rate of audio.
            samples_per_batch: Number of samples per batch. This should match the model's expectations.
            batches_per_iteration: Number of batches to train on per iteration. Increase until you run into OOMs for training speed.
            dataset_batch_size: Batch size that comes from dataset. This will be continuous speed without interruption
            device: Device to train on.
            overlap_samples: Number of samples to overlap between batches.
            model_weights_dir: Directory to save model weights to. If not specified, weights are not saved.
            model_weights_save_every_iterations: Save model weights every n iterations. If not specified, weights are not saved.
            summary_writer: SummaryWriter for logging to tensorboard. If not specified, no logging is done.
            send_audio_clip_every_iterations: Send audio clips to tensorboard every n
            dataset_batches_length: Length of batches expected from the dataset
            training_classifier: If true, this is a classifier model. Data is expected to be a tuple of ([batches, channels, samples], [batches, labels]).

        """
        self.input_dataset_loader = input_dataset_loader
        self.models = models
        self.loss_function_configs = loss_function_configs
        self.sample_rate = sample_rate
        self.samples_per_batch = samples_per_batch
        self.batches_per_iteration = batches_per_iteration
        self.dataset_batch_size = dataset_batch_size
        self.device = device
        self.overlap_samples = overlap_samples
        self.model_weights_dir = model_weights_dir
        self.model_weights_save_every_iterations = model_weights_save_every_iterations
        self.summary_writer = summary_writer
        self.send_audio_clip_every_iterations = send_audio_clip_every_iterations
        self.dataset_batches_length = dataset_batches_length
        self.training_classifier = training_classifier

        self.samples_per_iteration = self.samples_per_batch * self.batches_per_iteration
        self.iteration_count = 0
        self.epoch_start_time = None
        self.epoch_count = 0
        self.samples_processed = 0
        self.train_start_time = None
        self.log_per_iterations = 15000 // self.batches_per_iteration
        self.last_samples_processed_log_time = time.time()
        self.samples_processed_since_last_log = 0
        self.batches_count = 0
        self.batches_since_last_validation = 0

    def train(self):
        self.train_start_time = time.time()

        gc.collect()
        torch.cuda.empty_cache()

        while True:
            self.train_epoch()
            self.epoch_count += 1
            self.summary_writer.add_scalar("epoch", self.epoch_count)

    def better_split_discard_remainder(self, tensor, split_size):
        splits = torch.split(tensor, split_size, dim=-1)

        # print(f"1 splits: {splits[0].size()}")

        # Check the size of the last group and discard if necessary
        if splits[-1].size(0) != split_size:
            splits = splits[:-1]

        # print(f"2 splits: {splits}")

        return torch.cat(splits, 0)

    def train_epoch(self):
        self.epoch_start_time = time.time()

        input_loader_iter = iter(self.input_dataset_loader)

        for model_config in self.models:
            model_batches_count = 0
            continue_training = True
            while continue_training:
                send_audio_clips = self.iteration_count % self.send_audio_clip_every_iterations == 0
                result = self.train_iteration(
                    model_config=model_config,
                    input_loader_iter=input_loader_iter,
                    should_record_audio_clips=send_audio_clips
                )
                model_batches_count += self.batches_per_iteration
                continue_training = result.continue_training and model_batches_count < model_config.batches_per_model_rotation

                if self.iteration_count % self.log_per_iterations == 0 and self.iteration_count != 0:
                    elapsed_training_time = time.time() - self.train_start_time
                    elapsed_time_since_logged_samples_processed = time.time() - self.last_samples_processed_log_time

                    self.summary_writer.add_scalar("samples_processed", self.samples_processed, self.iteration_count)
                    self.summary_writer.add_scalar("samples_processed_per_microsecond",
                                                self.samples_processed_since_last_log / elapsed_time_since_logged_samples_processed / 1000 / 1000,
                                                self.batches_count)
                    self.summary_writer.add_scalar("iterations_per_second",
                                                self.iteration_count / elapsed_training_time,
                                                self.batches_count)
                    batches_complete = self.iteration_count * self.batches_per_iteration
                    batches_per_second = batches_complete / elapsed_training_time
                    self.summary_writer.add_scalar("epoch_percentage_complete",
                                                batches_complete / self.dataset_batches_length * 100,
                                                self.batches_count)
                    self.summary_writer.add_scalar("epoch_estimated_time_per_epoch_minutes",
                                                self.dataset_batches_length / batches_per_second / 60,
                                                self.batches_count)

                    self.last_samples_processed_log_time = time.time()
                    self.samples_processed_since_last_log = 0
                    self.summary_writer.flush()

                if (self.iteration_count % self.model_weights_save_every_iterations == 0
                        and self.model_weights_save_every_iterations != 0):
                    time_string = time.strftime("%Y%m%d-%H%M%S")
                    for model_name, model, _, _ in self.models:
                        model_save_path = self.model_weights_dir + f"/{model_name}-{time_string}"
                        self.summary_writer.add_text("model_save_path_{model_name}", model_save_path, self.batches_count)
                        torch.save(model.state_dict(), model_save_path)

                self.iteration_count += 1
                self.batches_count += self.batches_per_iteration
                self.batches_since_last_validation += self.batches_per_iteration
                
        if self.batches_since_last_validation >= self.validation_config.run_validation_every_batches:
            self.run_validation()
            self.batches_since_last_validation = 0

    def train_iteration(
            self,
            model_config,
            input_loader_iter,
            should_record_audio_clips: bool):

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data preparation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        should_log_extra_stuff = self.iteration_count % self.log_per_iterations == 0
        if should_log_extra_stuff:
            perf_iteration_start = time.perf_counter()

        if self.training_classifier:
            next_input, golden_classifier_values = next(input_loader_iter, None)
            if next_input is None:
                return IterationResult(False)

            next_input = next_input.view(-1, self.samples_per_batch * self.dataset_batch_size)
            golden_classifier_values = golden_classifier_values.mean(dim=1)

        else:
            next_input = next(input_loader_iter, None)
            if next_input is None:
                return IterationResult(False)

            next_input = next_input.view(-1, 2, self.samples_per_batch)
            next_input = next_input.squeeze(0).permute(1, 0, 2)


        input_subsamples = next_input.squeeze(0)
        if not self.training_classifier:
            input_subsamples = input_subsamples[0]

        if not self.training_classifier:
            golden_subsamples = next_input.squeeze(0)[1]
            golden_reconstructed = self.reconstruct_overlapping_samples_nofade(golden_subsamples)

        del next_input
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Log prep ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if should_log_extra_stuff:
            perf_data_prep_end = time.perf_counter()
            self.summary_writer.add_scalar(
                "perf_data_prep", perf_data_prep_end - perf_iteration_start, self.batches_count)

        if should_record_audio_clips:
            noisy_audio = self.reconstruct_overlapping_samples_nofade(input_subsamples.view(-1, self.samples_per_batch)).cpu().detach()

            self.summary_writer.add_audio(f"noisy_audio", noisy_audio,
                                          sample_rate=self.sample_rate, global_step=self.batches_count)

            if not self.training_classifier:
                clear_audio = golden_reconstructed.cpu().detach()

                self.summary_writer.add_audio(f"clear_audio", clear_audio,
                                              sample_rate=self.sample_rate, global_step=self.batches_count)

        perf_memory_allocated_dict = {}
        perf_memory_reserved_dict = {}
        perf_memory_max_reserved_dict = {}
        perf_model_prediction_dict = {}
        perf_loss_name_to_model_dict = {lc.loss_name: dict() for lc in self.loss_function_configs}
        perf_loss_backward_dict = {}
        perf_optimizer_dict = {}

        total_loss_dict = {}
        loss_to_model_loss_dict = {lc.loss_name: dict() for lc in self.loss_function_configs}
        loss_to_model_loss_weighted_dict = {lc.loss_name: dict() for lc in self.loss_function_configs}

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        model_name = model_config.name
        model = model_config.model
        optimizer = model_config.optimizer
        scheduler = model_config.scheduler
        
        model.train()

        if should_log_extra_stuff:
            perf_model_train_prediction_start = time.perf_counter()

        input_unsqueezed = input_subsamples.unsqueeze(dim=1)
        # print(f"input_unsqueezed.size() = {input_unsqueezed.size()}")
        prediction_raw = model(input_unsqueezed)
        if not self.training_classifier:
            prediction_raw = prediction_raw.squeeze(dim=1)

        if should_log_extra_stuff:
            perf_model_train_prediction_end = time.perf_counter()
            perf_model_prediction_dict[model_name] = (perf_model_train_prediction_end -
                                                        perf_model_train_prediction_start)

        if not self.training_classifier:
            # print(f"prediction_raw.size() = {prediction_raw.size()}")
            prediction = self.reconstruct_overlapping_samples_nofade(prediction_raw)
            # print(f"prediction.size() AFTER RECONSTRUCT = {prediction.size()}")

            if should_record_audio_clips:
                prediction_cpu = prediction.cpu().detach()

        else:
            prediction = prediction_raw

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Loss calculation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss = None
        # print(f"self.loss_function_configs: {self.loss_function_configs}")
        for loss_config in self.loss_function_configs:
            # print(f"loss_name = {loss_name}")
            perf_loss_start = time.perf_counter()

            if self.training_classifier and prediction.size() != golden_classifier_values.size():
                print(f"Wrong golden values size! prediction.size() = {prediction.size()} golden_classifier_values.size() = {golden_classifier_values.size()}")

            goldens = golden_classifier_values if self.training_classifier else golden_reconstructed
            # print(f"prediction.size() = {prediction.size()} goldens.size(): {goldens.size()}")
            if loss_config.loss_batch_size:
                prediction = self.better_split_discard_remainder(prediction, loss_config.loss_batch_size)
                # print(f"prediction.size() after loss split = {prediction.size()}")
                goldens = self.better_split_discard_remainder(goldens, loss_config.loss_batch_size)
            if loss_config.loss_is_unary:
                loss_out = torch.mean(loss_config.loss_fn(prediction))
                # print(f"loss_out = {loss_out}")
            else:
                loss_out = loss_config.loss_fn(prediction, goldens)

            loss_out_weighted = loss_out * loss_config.loss_weight

            if should_log_extra_stuff:
                loss_to_model_loss_weighted_dict[loss_name].update({
                    model_name: loss_out_weighted
                })
                loss_to_model_loss_dict[loss_name].update({
                    model_name: loss_out
                })

            # TODO: HOW TO UNDERSTAND GRADIENTS FROM LOSSES

            if loss:
                loss = loss + loss_out_weighted
            else:
                loss = loss_out_weighted

            perf_loss_end = time.perf_counter()

            if should_log_extra_stuff:
                perf_loss_name_to_model_dict[loss_name].update({ model_name: perf_loss_end - perf_loss_start })

        if should_log_extra_stuff:
            total_loss_dict[model_name] = loss

        del prediction

        optimizer.zero_grad(set_to_none=True)

        perf_loss_backward_start = time.perf_counter()
        loss.backward()
        perf_loss_backward_end = time.perf_counter()

        if should_log_extra_stuff:
            perf_loss_backward_dict[model_name] = perf_loss_backward_end - perf_loss_backward_start
            perf_memory_allocated_dict[model_name] = torch.cuda.memory_allocated(0)
            perf_memory_reserved_dict[model_name] = torch.cuda.memory_reserved(0)
            perf_memory_max_reserved_dict[model_name] = torch.cuda.max_memory_reserved(0)

        del loss

        perf_optimizer_step_start = time.perf_counter()
        optimizer.step()
        perf_optimizer_step_end = time.perf_counter()
        perf_optimizer_dict[model_name] = perf_optimizer_step_end - perf_optimizer_step_start


        if scheduler:
            scheduler.step()

        model.eval()

        if should_record_audio_clips:
            if self.training_classifier:
                self.summary_writer.add_scalar(f"classifier_prediction_{model_name}", prediction_raw.mean(), self.batches_count),
            else:
                if prediction_cpu is not None:
                    self.summary_writer.add_audio(f"prediction_audio_{model_name}", prediction_cpu,
                                                    sample_rate=self.sample_rate, global_step=self.batches_count)
                else:
                    print("prediction_cpu is None!")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Log things to tensorboard ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if should_log_extra_stuff:
            perf_logging_start = time.time()
            for loss_name, loss_dict in loss_to_model_loss_dict.items():
                loss_dict = {model_name: loss.item() for model_name, loss in loss_dict.items()}
                self.summary_writer.add_scalars(f"loss_{loss_name}", loss_dict, self.batches_count)

            total_loss_dict = {model_name: loss.item() for model_name, loss in total_loss_dict.items()}
            self.summary_writer.add_scalars(f"loss_total", total_loss_dict, self.batches_count)

            self.summary_writer.add_scalars(
                f"perf_model_train_prediction", perf_model_prediction_dict, self.batches_count)

            self.summary_writer.add_scalars("perf_model_prediction", perf_model_prediction_dict, self.batches_count)

            for loss_name, model_dict in perf_loss_name_to_model_dict.items():
                self.summary_writer.add_scalars(
                    f"perf_loss_{loss_name}", model_dict, self.batches_count)

            self.summary_writer.add_scalars(
                f"perf_loss_backward", perf_loss_backward_dict, self.batches_count)

            self.summary_writer.add_scalars("perf_memory_allocated", perf_memory_allocated_dict, self.batches_count)
            self.summary_writer.add_scalars("perf_memory_reserved", perf_memory_reserved_dict, self.batches_count)
            self.summary_writer.add_scalars("perf_memory_max_reserved", perf_memory_max_reserved_dict, self.batches_count)
            self.summary_writer.add_scalars("perf_optimizer", perf_optimizer_dict, self.batches_count)

        self.samples_processed += self.batches_per_iteration * self.samples_per_iteration
        self.samples_processed_since_last_log += self.batches_per_iteration * self.samples_per_iteration

        perf_iteration_end = time.perf_counter()
        if should_log_extra_stuff:
            self.summary_writer.add_scalar("perf_iteration",
                                           perf_iteration_end - perf_iteration_start, self.batches_count)
            perf_logging_end = time.time()
            self.summary_writer.add_scalar(
                "perf_logging", perf_logging_end - perf_logging_start, self.batches_count)

        return IterationResult(
            continue_training=True,
        )

    def reconstruct_overlapping_samples_fade(self, samples: torch.Tensor):
        num_batches = samples.size()[0]
        non_overlapping_size = self.samples_per_batch - self.overlap_samples * 2

        total_length = (num_batches * (self.samples_per_batch - self.overlap_samples)) - self.overlap_samples * 2
        output = torch.zeros(total_length, dtype=samples.dtype, device=samples.device)

        fade = torchaudio.transforms.Fade(fade_in_len=self.overlap_samples, fade_out_len=self.overlap_samples)

        first_faded = fade(samples[0])
        output[:non_overlapping_size + self.overlap_samples] += first_faded[self.overlap_samples:]

        for idx, batch in enumerate(samples[1:-1]):
            idx = idx + 1
            start = (idx * non_overlapping_size) + ((idx - 1) * self.overlap_samples)
            end = start + non_overlapping_size + self.overlap_samples * 2
            output[start:end] += fade(batch)

        last_faded = fade(samples[-1])
        output[-(non_overlapping_size + self.overlap_samples):] += last_faded[:-self.overlap_samples]

        return output.unsqueeze(dim=0).unsqueeze(dim=0)
    
    def run_validation(self):
        pass

    def reconstruct_overlapping_samples_nofade(self, samples: torch.Tensor):
        num_batches = samples.size()[0]
        non_overlapping_size = self.samples_per_batch - self.overlap_samples * 2

        total_length = (num_batches * (self.samples_per_batch - self.overlap_samples)) - self.overlap_samples * 2
        # print(f"total_length {total_length}")
        output = torch.zeros(total_length, dtype=samples.dtype, device=samples.device)

        output[:non_overlapping_size + self.overlap_samples] += samples[0][self.overlap_samples:]

        for idx, batch in enumerate(samples[1:-1]):
            idx = idx + 1
            start = (idx * non_overlapping_size) + ((idx - 1) * self.overlap_samples)
            end = start + non_overlapping_size + self.overlap_samples * 2
            output[start:end] = batch

        output[-(non_overlapping_size + self.overlap_samples):] += samples[-1][:-self.overlap_samples]

        return output.unsqueeze(dim=0).unsqueeze(dim=0)


    def reconstruct_overlapping_samples_nofade_wrong_alignment(self, samples: torch.Tensor):
        num_batches = samples.size()[0]
        step_size = self.samples_per_batch + self.overlap_samples

        total_length = num_batches * step_size + self.overlap_samples
        output = torch.zeros(total_length, dtype=samples.dtype, device=samples.device)

        for idx, batch in enumerate(samples):
            start = idx * step_size
            end = start + self.samples_per_batch
            output[start:end] = batch

        return output.unsqueeze(dim=0).unsqueeze(dim=0)



@dataclass
class IterationResult:
    # pylint: disable=missing-class-docstring
    continue_training: bool
