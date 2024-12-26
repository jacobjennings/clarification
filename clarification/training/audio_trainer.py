import gc
import time

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..util import better_split_discard_remainder

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

@dataclass
class IterationResult:
    # pylint: disable=missing-class-docstring
    should_continue: bool

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
        self.validation_config = validation_config

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

    def train_epoch(self):
        self.epoch_start_time = time.time()

        input_loader_iter = iter(self.input_dataset_loader)

        for model_config in self.models:
            model_batches_count = 0
            continue_training = True
            while continue_training:
                send_audio_clips = self.iteration_count % self.send_audio_clip_every_iterations == 0
                result = self.run_iteration(
                    model_config=model_config,
                    input_loader_iter=input_loader_iter,
                    should_record_audio_clips=send_audio_clips,
                    is_validation=False
                )
                model_batches_count += self.batches_per_iteration
                continue_training = result.should_continue and model_batches_count < model_config.batches_per_model_rotation

                if self.iteration_count % self.log_per_iterations == 0 and self.iteration_count != 0:
                    elapsed_training_time = time.time() - self.train_start_time
                    elapsed_time_since_logged_samples_processed = time.time() - self.last_samples_processed_log_time

                    self.log_post_iteration_stuff(elapsed_time_since_logged_samples_processed, elapsed_training_time)

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

    def log_post_iteration_stuff(self, elapsed_time_since_logged_samples_processed, elapsed_training_time):
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

    def run_iteration(
            self,
            model_config,
            input_loader_iter,
            should_record_audio_clips: bool,
            is_validation: bool):

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data preparation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        writer_tag_prefix = "validation_" if is_validation else ""
        perf_iteration_start = None
        should_log_extra_stuff = self.iteration_count % self.log_per_iterations == 0
        if should_log_extra_stuff:
            perf_iteration_start = time.perf_counter()

        input_subsamples, golden_reconstructed, golden_classifier_values = self.iteration_data_prep(input_loader_iter)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Log prep ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if should_log_extra_stuff and perf_iteration_start:
            perf_data_prep_end = time.perf_counter()
            self.summary_writer.add_scalar(
                f"{writer_tag_prefix}perf_data_prep", perf_data_prep_end - perf_iteration_start, self.batches_count)

        if should_record_audio_clips:
            self.record_noisy_clear_audio_clips(golden_reconstructed, input_subsamples, writer_tag_prefix)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        model_name = model_config.name
        model = model_config.model
        optimizer = model_config.optimizer
        scheduler = model_config.scheduler

        if is_validation:
            model.eval()
        else:
            model.train()

        input_unsqueezed = input_subsamples.unsqueeze(dim=1)
        prediction_raw = model(input_unsqueezed)
        if not self.training_classifier:
            prediction_raw = prediction_raw.squeeze(dim=1)

        prediction_cpu = None
        if not self.training_classifier:
            prediction = self.reconstruct_overlapping_samples_nofade(prediction_raw)

            if should_record_audio_clips:
                prediction_cpu = prediction.cpu().detach()

        else:
            prediction = prediction_raw

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Loss calculation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        loss, loss_to_model_loss_dict, total_loss_dict = self.loss_calculation(
            golden_classifier_values,
            golden_reconstructed,
            model_name,
            prediction,
            should_log_extra_stuff)

        if should_log_extra_stuff:
            total_loss_dict[model_name] = loss

        del prediction

        if not is_validation:
            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            del loss

            optimizer.step()

            if scheduler:
                scheduler.step()

            model.eval()

        if should_record_audio_clips:
            self.record_prediction_audio_clips(model_name, prediction_cpu, prediction_raw, writer_tag_prefix)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Log things to tensorboard ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if should_log_extra_stuff:
            self.log_extra_stuff(loss_to_model_loss_dict, perf_iteration_start, total_loss_dict, writer_tag_prefix)

        self.samples_processed += self.batches_per_iteration * self.samples_per_iteration
        self.samples_processed_since_last_log += self.batches_per_iteration * self.samples_per_iteration

        return IterationResult(
            should_continue=True,
        )

    def log_extra_stuff(self, loss_to_model_loss_dict, perf_iteration_start, total_loss_dict, writer_tag_prefix):
        for loss_name, loss_dict in loss_to_model_loss_dict.items():
            loss_dict = {model_name: loss.item() for model_name, loss in loss_dict.items()}
            self.summary_writer.add_scalars(f"{writer_tag_prefix}loss_{loss_name}", loss_dict, self.batches_count)
        total_loss_dict = {model_name: loss.item() for model_name, loss in total_loss_dict.items()}
        self.summary_writer.add_scalars(f"{writer_tag_prefix}loss_total", total_loss_dict, self.batches_count)
        perf_iteration_end = time.perf_counter()
        self.summary_writer.add_scalar(f"{writer_tag_prefix}perf_iteration",
                                       perf_iteration_end - perf_iteration_start, self.batches_count)

    def record_prediction_audio_clips(self, model_name, prediction_cpu, prediction_raw, writer_tag_prefix):
        if self.training_classifier:
            self.summary_writer.add_scalar(f"{writer_tag_prefix}classifier_prediction_{model_name}", prediction_raw.mean(),
                                           self.batches_count),
        else:
            if prediction_cpu is not None:
                self.summary_writer.add_audio(f"{writer_tag_prefix}prediction_audio_{model_name}", prediction_cpu,
                                              sample_rate=self.sample_rate, global_step=self.batches_count)
            else:
                print("prediction_cpu is None!")

    def loss_calculation(self, golden_classifier_values, golden_reconstructed, model_name, prediction,
                         should_log_extra_stuff):
        total_loss_dict = {}
        loss_to_model_loss_dict = {lc.name: dict() for lc in self.loss_function_configs}
        loss_to_model_loss_weighted_dict = {lc.name: dict() for lc in self.loss_function_configs}
        loss = None
        for loss_config in self.loss_function_configs:
            if self.training_classifier and prediction.size() != golden_classifier_values.size():
                print(
                    f"Wrong golden values size! prediction.size() = {prediction.size()} golden_classifier_values.size() = {golden_classifier_values.size()}")

            goldens = golden_classifier_values if self.training_classifier else golden_reconstructed

            if loss_config.batch_size:
                prediction = better_split_discard_remainder(prediction, loss_config.batch_size)
                goldens = better_split_discard_remainder(goldens, loss_config.batch_size)

            if loss_config.is_unary:
                loss_out = torch.mean(loss_config.fn(prediction))
            else:
                loss_out = loss_config.fn(prediction, goldens)

            loss_out_weighted = loss_out * loss_config.weight

            if should_log_extra_stuff:
                loss_to_model_loss_weighted_dict[loss_config.name].update({
                    model_name: loss_out_weighted
                })
                loss_to_model_loss_dict[loss_config.name].update({
                    model_name: loss_out
                })

            if loss:
                loss = loss + loss_out_weighted
            else:
                loss = loss_out_weighted
        return loss, loss_to_model_loss_dict, total_loss_dict

    def record_noisy_clear_audio_clips(self, golden_reconstructed, input_subsamples, writer_tag_prefix):
        noisy_audio = self.reconstruct_overlapping_samples_nofade(
            input_subsamples.view(-1, self.samples_per_batch)).cpu().detach()
        self.summary_writer.add_audio(f"{writer_tag_prefix}noisy_audio", noisy_audio,
                                      sample_rate=self.sample_rate, global_step=self.batches_count)
        if not self.training_classifier:
            clear_audio = golden_reconstructed.cpu().detach()

            self.summary_writer.add_audio(f"{writer_tag_prefix}clear_audio", clear_audio,
                                          sample_rate=self.sample_rate, global_step=self.batches_count)

    def iteration_data_prep(self, loader_iter):
        golden_classifier_values = None
        if self.training_classifier:
            next_input, golden_classifier_values = next(loader_iter, None)
            if next_input is None:
                return IterationResult(False)

            next_input = next_input.view(-1, self.samples_per_batch * self.dataset_batch_size)
            golden_classifier_values = golden_classifier_values.mean(dim=1)

        else:
            next_input = next(loader_iter, None)
            if next_input is None:
                return IterationResult(False)

            next_input = next_input.view(-1, 2, self.samples_per_batch)
            next_input = next_input.squeeze(0).permute(1, 0, 2)


        input_subsamples = next_input.squeeze(0)
        if not self.training_classifier:
            input_subsamples = input_subsamples[0]

        golden_reconstructed = None
        if not self.training_classifier:
            golden_subsamples = next_input.squeeze(0)[1]
            golden_reconstructed = self.reconstruct_overlapping_samples_nofade(golden_subsamples)

        del next_input

        return input_subsamples, golden_reconstructed, golden_classifier_values

    def run_validation(self):
        for model_config in self.models:
            model_batches_count = 0
            continue_eval = True
            input_loader_iter = iter(self.validation_config.test_loader)
            while continue_eval:
                send_audio_clips = self.iteration_count % self.send_audio_clip_every_iterations == 0
                result = self.run_iteration(
                    model_config=model_config,
                    input_loader_iter=input_loader_iter,
                    should_record_audio_clips=send_audio_clips,
                    is_validation=True
                )
                model_batches_count += self.batches_per_iteration
                continue_eval = result.should_continue and model_batches_count < self.validation_config.test_batches

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
