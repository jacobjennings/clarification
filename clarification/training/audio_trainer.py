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


class AudioTrainer:
    def __init__(
            self,
            input_dataset_loader: DataLoader,
            models: List[Tuple[str, nn.Module, optim.Optimizer, Optional[optim.lr_scheduler.LRScheduler]]],
            loss_function_tuples: List[Tuple[str, float, nn.Module]],
            sample_rate: int,
            samples_per_batch: int,
            batches_per_iteration: int,
            device: str,
            overlap_samples: int,
            model_weights_dir: Optional[str] = None,
            model_weights_save_every_iterations: int = None,
            summary_writer: Optional[SummaryWriter] = None,
            send_audio_clip_every_iterations: int = 100,
            dataset_batches_length: int = None
    ):
        """Training loop for audio.
        
        Args:
            input_dataset_loader: DataLoader.
            models: List of (name, model, optimizer, scheduler) tuples to train.
            loss_function_tuples: List of loss function tuples of form (name, weight, fn), i.e. [("MyLoss", 1.2,  myloss), ...]
            sample_rate: Sample rate of audio.
            samples_per_batch: Number of samples per batch. This should match the model's expectations.
            batches_per_iteration: Number of batches to train on per iteration. Increase until you run into OOMs for training speed.
            device: Device to train on.
            overlap_samples: Number of samples to overlap between batches.
            model_weights_dir: Directory to save model weights to. If not specified, weights are not saved.
            model_weights_save_every_iterations: Save model weights every n iterations. If not specified, weights are not saved.

        """
        self.input_dataset_loader = input_dataset_loader
        self.models = models
        self.loss_function_tuples = loss_function_tuples
        self.sample_rate = sample_rate
        self.samples_per_batch = samples_per_batch
        self.batches_per_iteration = batches_per_iteration
        self.device = device
        self.overlap_samples = overlap_samples
        self.model_weights_dir = model_weights_dir
        self.model_weights_save_every_iterations = model_weights_save_every_iterations
        self.summary_writer = summary_writer
        self.send_audio_clip_every_iterations = send_audio_clip_every_iterations
        self.dataset_batches_length = dataset_batches_length

        self.samples_per_iteration = self.samples_per_batch * self.batches_per_iteration
        self.iteration_count = 0
        self.epoch_start_time = None
        self.epoch_count = 0
        self.samples_processed = 0
        self.train_start_time = None

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

        continue_training = True
        while continue_training:
            send_audio_clips = self.iteration_count % self.send_audio_clip_every_iterations == 0
            result = self.train_iteration(
                input_loader_iter=input_loader_iter,
                should_record_audio_clips=send_audio_clips
            )
            continue_training = result.continue_training

            if self.iteration_count % 15 == 0 and self.iteration_count != 0:
                elapsed_training_time = time.time() - self.train_start_time
                self.summary_writer.add_scalar("samples_processed", self.samples_processed, self.iteration_count)
                self.summary_writer.add_scalar("samples_processed_per_microsecond",
                                               self.samples_processed / elapsed_training_time / 1000 / 1000,
                                               self.iteration_count)
                self.summary_writer.add_scalar("iterations_per_second",
                                               self.iteration_count / elapsed_training_time,
                                               self.iteration_count)
                batches_complete = self.iteration_count * self.batches_per_iteration
                batches_per_second = batches_complete / elapsed_training_time
                self.summary_writer.add_scalar("epoch_percentage_complete",
                                               batches_complete / self.dataset_batches_length * 100,
                                               self.iteration_count)
                self.summary_writer.add_scalar("epoch_estimated_time_per_epoch_minutes",
                                               self.dataset_batches_length / batches_per_second / 60,
                                               self.iteration_count)

            if (self.iteration_count % self.model_weights_save_every_iterations == 0
                    and self.model_weights_save_every_iterations != 0):
                time_string = time.strftime("%Y%m%d-%H%M%S")
                for model_name, model, _, _ in self.models:
                    model_save_path = self.model_weights_dir + f"/{model_name}-{time_string}"
                    self.summary_writer.add_text("model_save_path_{model_name}", model_save_path, self.iteration_count)
                    torch.save(model.state_dict(), model_save_path)

            self.summary_writer.flush()

            self.iteration_count += 1

    def train_iteration(
            self,
            input_loader_iter,
            should_record_audio_clips: bool):

        perf_iteration_start = time.perf_counter()

        next_input = next(input_loader_iter, None).to(self.device).squeeze(0).permute(1, 0, 2)
        if next_input is None:
            return IterationResult(False, 0, None)

        input_subsamples = next_input.squeeze(0)[0]
        golden_subsamples = next_input.squeeze(0)[1]

        del next_input

        golden_reconstructed = self.reconstruct_overlapping_samples_fade(golden_subsamples)

        perf_data_prep_end = time.perf_counter()
        self.summary_writer.add_scalar(
            "perf_data_prep", perf_data_prep_end - perf_iteration_start, self.iteration_count)

        if should_record_audio_clips:
            noisy_audio = self.reconstruct_overlapping_samples_nofade(input_subsamples).cpu().detach()
            clear_audio = golden_reconstructed.cpu().detach()

            self.summary_writer.add_audio(f"noisy_audio", noisy_audio,
                                          sample_rate=self.sample_rate, global_step=self.iteration_count)
            self.summary_writer.add_audio(f"clear_audio", clear_audio,
                                          sample_rate=self.sample_rate, global_step=self.iteration_count)

        for model_idx, (model_name, model, optimizer, scheduler) in enumerate(self.models):
            model.train()

            perf_model_train_prediction_start = time.perf_counter()
            prediction_raw = model(input_subsamples.unsqueeze(dim=1)).squeeze(dim=1)
            perf_model_train_prediction_end = time.perf_counter()

            self.summary_writer.add_scalar(
                f"perf_model_train_prediction_{model_name}",
                perf_model_train_prediction_end - perf_model_train_prediction_start, self.iteration_count)

            prediction = self.reconstruct_overlapping_samples_nofade(prediction_raw)

            if should_record_audio_clips:
                prediction_cpu = prediction.cpu().detach()

            loss = None
            for loss_name, loss_weight, loss_fn in self.loss_function_tuples:
                perf_loss_start = time.perf_counter()
                loss_out = loss_fn(prediction, golden_reconstructed)
                loss_out_weighted = loss_out * loss_weight

                self.summary_writer.add_scalar(f"{loss_name}_{model_name}_weighted",
                                               loss_out_weighted.item(), self.iteration_count)
                self.summary_writer.add_scalar(f"{loss_name}_{model_name}", loss_out.item(), self.iteration_count)

                if loss:
                    loss = loss + loss_out_weighted
                else:
                    loss = loss_out_weighted

                perf_loss_end = time.perf_counter()
                self.summary_writer.add_scalar(
                    f"perf_loss_{loss_name}_{model_name}", perf_loss_end - perf_loss_start, self.iteration_count)

            self.summary_writer.add_scalar(f"total_loss_model_{model_name}", loss.item(), self.iteration_count)

            del prediction

            optimizer.zero_grad()

            perf_loss_backward_start = time.perf_counter()
            loss.backward()
            perf_loss_backward_end = time.perf_counter()
            self.summary_writer.add_scalar(
                f"perf_loss_backward_{model_name}",
                perf_loss_backward_end - perf_loss_backward_start, self.iteration_count)

            self.summary_writer.add_scalar(f"memory_post_backprop_allocated_model_{model_name}",
                                           torch.cuda.memory_allocated(0), self.iteration_count)
            self.summary_writer.add_scalar(f"memory_post_backprop_reserved_model_{model_name}",
                                           torch.cuda.memory_reserved(0), self.iteration_count)
            self.summary_writer.add_scalar(f"memory_post_backprop_max_reserved_model_{model_name}",
                                           torch.cuda.max_memory_reserved(0), self.iteration_count)

            del loss

            perf_optimizer_step_start = time.perf_counter()
            optimizer.step()
            perf_optimizer_step_end = time.perf_counter()
            self.summary_writer.add_scalar(
                f"perf_optimizer_step_{model_name}",
                perf_optimizer_step_end - perf_optimizer_step_start, self.iteration_count)

            if scheduler:
                scheduler.step()

            perf_eval_start = time.perf_counter()
            model.eval()
            perf_eval_end = time.perf_counter()
            self.summary_writer.add_scalar(
                f"perf_eval_{model_name}", perf_eval_end - perf_eval_start, self.iteration_count)

            if should_record_audio_clips:
                if prediction_cpu is not None:
                    self.summary_writer.add_audio(f"prediction_audio_{model_name}", prediction_cpu,
                                                  sample_rate=self.sample_rate, global_step=self.iteration_count)
                else:
                    print("prediction_cpu is None!")

        self.samples_processed += self.batches_per_iteration * self.samples_per_iteration

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


    def reconstruct_overlapping_samples_nofade(self, samples: torch.Tensor):
        num_batches = samples.size()[0]
        non_overlapping_size = self.samples_per_batch - self.overlap_samples * 2

        total_length = (num_batches * (self.samples_per_batch - self.overlap_samples)) - self.overlap_samples * 2
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
