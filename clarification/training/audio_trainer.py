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

        self.samples_per_iteration = self.samples_per_batch * self.batches_per_iteration
        self.iteration_count = 0
        self.epoch_start_time = None
        self.files_processed = 0

    def train(self):
        files_processed = 0

        gc.collect()
        torch.cuda.empty_cache()

        epoch_count = 0
        self.iteration_count = 0

        while True:
            self.files_processed = self.train_epoch(files_processed=files_processed)
            epoch_count += 1
            self.summary_writer.add_scalar("epoch", epoch_count)

    def train_epoch(self, files_processed: int):
        self.epoch_start_time = time.time()

        input_loader_iter = iter(self.input_dataset_loader)

        remaining_input_batches = None
        remaining_golden_batches = None

        continue_training = True
        while continue_training:
            send_audio_clips = self.iteration_count % self.send_audio_clip_every_iterations == 0
            result = self.train_iteration(
                input_loader_iter=input_loader_iter,
                remaining_input_batches=remaining_input_batches,
                remaining_golden_batches=remaining_golden_batches,
                should_record_audio_clips=send_audio_clips
            )
            continue_training = result.continue_training
            files_processed += result.files_processed_during_iteration
            remaining_input_batches = result.remaining_input_batches
            remaining_golden_batches = result.remaining_golden_batches
            self.summary_writer.add_scalar("files_processed", files_processed, self.iteration_count)
            self.summary_writer.add_scalar("files_processed_per_second",
                                           files_processed / (time.time() - self.epoch_start_time),
                                           self.iteration_count)

            if self.iteration_count % self.model_weights_save_every_iterations == 0:
                time_string = time.strftime("%Y%m%d-%H%M%S")
                for model_name, model, _, _ in self.models:
                    model_save_path = self.model_weights_dir + f"/{model_name}-{time_string}"
                    self.summary_writer.add_text("model_save_path_{model_name}", model_save_path, self.iteration_count)
                    torch.save(model.state_dict(), model_save_path)

            self.summary_writer.flush()

            self.iteration_count += 1

        return files_processed

    def train_iteration(
            self,
            input_loader_iter,
            remaining_input_batches,
            remaining_golden_batches,
            should_record_audio_clips: bool):

        files_processed = 0

        if remaining_input_batches is not None:
            input_subsamples = remaining_input_batches
            golden_subsamples = remaining_golden_batches
            remaining_input_batches = None
            remaining_golden_batches = None
        else:
            next_input = next(input_loader_iter, None)
            if next_input is None:
                return IterationResult(False, 0, None, None)

            input_subsamples = torch.stack([t[0].squeeze(0).squeeze(0).to(self.device) for t in next_input[0]])
            golden_subsamples = torch.stack([t[0].squeeze(0).squeeze(0).to(self.device) for t in next_input[1]])

            del next_input
            files_processed += 1

        while input_subsamples.size()[0] < self.batches_per_iteration:
            next_input = next(input_loader_iter, None)
            if next_input is None:
                # Discard remainder.
                return IterationResult(False, 0, None, None)

            next_input_subsamples = torch.stack([t[0].squeeze(0).squeeze(0).to(self.device) for t in next_input[0]])
            next_golden_subsamples = torch.stack([t[0].squeeze(0).squeeze(0).to(self.device) for t in next_input[1]])

            input_subsamples = torch.cat((input_subsamples, next_input_subsamples), dim=0)
            golden_subsamples = torch.cat((golden_subsamples, next_golden_subsamples), dim=0)

            del next_input
            files_processed += 1

        if input_subsamples.size()[0] > self.batches_per_iteration:
            remaining_input_batches = input_subsamples[self.batches_per_iteration:]
            remaining_golden_batches = golden_subsamples[self.batches_per_iteration:]

            input_subsamples = input_subsamples[:self.batches_per_iteration]
            golden_subsamples = golden_subsamples[:self.batches_per_iteration]

        golden_reconstructed = self.reconstruct_overlapping_samples_nofade(golden_subsamples)

        for model_idx, (model_name, model, optimizer, scheduler) in enumerate(self.models):
            model.train()

            prediction_raw = model(input_subsamples.unsqueeze(dim=1)).squeeze(dim=1)
            prediction = self.reconstruct_overlapping_samples(prediction_raw)

            if should_record_audio_clips:
                prediction_cpu = prediction.cpu().detach()

            loss = None
            for loss_name, loss_weight, loss_fn in self.loss_function_tuples:
                loss_out = loss_fn(prediction, golden_reconstructed)
                loss_out_weighted = loss_out * loss_weight

                self.summary_writer.add_scalar(f"{loss_name}_{model_name}_weighted",
                                               loss_out_weighted.item(), self.iteration_count)
                self.summary_writer.add_scalar(f"{loss_name}_{model_name}", loss_out.item(), self.iteration_count)

                if loss:
                    loss = loss + loss_out_weighted
                else:
                    loss = loss_out_weighted

            self.summary_writer.add_scalar(f"total_loss_model_{model_name}", loss.item(), self.iteration_count)

            del prediction

            optimizer.zero_grad()

            loss.backward()

            self.summary_writer.add_scalar(f"memory_post_backprop_allocated_model_{model_name}",
                                           torch.cuda.memory_allocated(0), self.iteration_count)
            self.summary_writer.add_scalar(f"memory_post_backprop_reserved_model_{model_name}",
                                           torch.cuda.memory_reserved(0), self.iteration_count)
            self.summary_writer.add_scalar(f"memory_post_backprop_max_reserved_model_{model_name}",
                                           torch.cuda.max_memory_reserved(0), self.iteration_count)

            del loss

            optimizer.step()
            if scheduler:
                scheduler.step()

            model.eval()

            if should_record_audio_clips:
                if prediction_cpu is not None:
                    noisy_audio = self.reconstruct_overlapping_samples_nofade(input_subsamples).cpu().detach()
                    clear_audio = golden_reconstructed.cpu().detach()
                    self.summary_writer.add_audio(f"noisy_audio_{model_name}", noisy_audio,
                                                  sample_rate=self.sample_rate, global_step=self.iteration_count)
                    self.summary_writer.add_audio(f"prediction_audio_{model_name}", prediction_cpu,
                                                  sample_rate=self.sample_rate, global_step=self.iteration_count)
                    self.summary_writer.add_audio(f"clear_audio_{model_name}", clear_audio,
                                                  sample_rate=self.sample_rate, global_step=self.iteration_count)
                else:
                    print("prediction_cpu is None!")

        return IterationResult(
            continue_training=True,
            files_processed_during_iteration=files_processed,
            remaining_input_batches=remaining_input_batches,
            remaining_golden_batches=remaining_golden_batches,
        )

    def reconstruct_overlapping_samples(self, samples: torch.Tensor):
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



@dataclass
class IterationResult():
    # pylint: disable=missing-class-docstring
    continue_training: bool
    files_processed_during_iteration: int
    remaining_input_batches: Optional[torch.Tensor]
    remaining_golden_batches: Optional[torch.Tensor]
