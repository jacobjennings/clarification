import gc
import time

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class AudioTrainer():
    def __init__(
        self,
        input_dataset_loader: DataLoader,
        golden_dataset_loader: DataLoader,
        models: List[nn.Module],
        loss_function_tuples: List[(str, float, nn.Module)],
        optimizer: optim.Optimizer,
        sample_rate: int,
        samples_per_batch: int,
        batches_per_iteration: int,
        device: str,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        model_weights_dir: Optional[str] = None,
        model_weights_save_every_iterations: int = None,
        summary_writer: Optional[SummaryWriter] = None,
        
    ):
        """Training loop for audio.
        
        Args:
            input_dataset_loader: DataLoader for noisy audio.
            golden_dataset_loader: DataLoader for clean audio.
            models: List of models to train.
            loss_function_tuples: List of loss function tuples of form (name, weight, fn), i.e. [("MyLoss", 1.2,  myloss), ...]
            optimizer: Optimizer to train with.
            sample_rate: Sample rate of audio.
            samples_per_batch: Number of samples per batch. This should match the model's expectations.
            batches_per_iteration: Number of batches to train on per iteration. Increase until you run into OOMs for training speed.
            device: Device to train on.
            scheduler: Learning rate scheduler. Optional.
            model_weights_dir: Directory to save model weights to. If not specified, weights are not saved.
            model_weights_save_every_iterations: Save model weights every n iterations. If not specified, weights are not saved.

        """
        self.input_dataset_loader = input_dataset_loader
        self.golden_dataset_loader = golden_dataset_loader
        self.models = models
        self.loss_function_tuples = loss_function_tuples
        self.optimizer = optimizer
        self.sample_rate = sample_rate
        self.samples_per_batch = samples_per_batch
        self.batches_per_iteration = batches_per_iteration
        self.device = device
        self.scheduler = scheduler
        self.model_weights_dir = model_weights_dir
        self.summary_writer = summary_writer
        
        self.samples_per_iteration = self.samples_per_batch * self.batches_per_iteration
        
        
    def train(self):
        files_processed = 0
        
        gc.collect()
        torch.cuda.empty_cache()
        
        epoch_count = 0
        
        while(True):
            self.files_processed = self.train_epoch(files_processed=files_processed)
            epoch_count += 1
            self.summary_writer.add_scalar("epoch", epoch_count)
            
    def train_epoch(self, files_processed: int):
        self.iteration_count = 0
        self.epoch_start_time = time.time()
        
        files_processed = files_processed

        input_loader_iter = iter(self.input_dataset_loader)
        golden_loader_iter = iter(self.golden_dataset_loader)

        remaining_input_samples = None
        remaining_golden_samples = None

        continue_training = True
        while continue_training:
            result = self.train_iteration(
                input_loader_iter=input_loader_iter,
                golden_loader_iter=golden_loader_iter,
                remaining_input_samples=remaining_input_samples,
                remaining_golden_samples=remaining_golden_samples,
            )
            continue_training = result.continue_training
            files_processed += result.files_processed_during_iteration
            remaining_input_samples = result.remaining_input_samples
            remaining_golden_samples = result.remaining_golden_samples
            self.summary_writer.add_scalar("files_processed", files_processed)
            
            model_save_path = self.model_weights_dir + f"/model-{time.strftime("%Y%m%d-%H%M%S")}"
            self.summary_writer.add_text("model_save_path", model_save_path)
            
            self.iteration_count += 1

            
    def train_iteration(
        self, 
        input_loader_iter, 
        golden_loader_iter, 
        remaining_input_samples, 
        remaining_golden_samples, 
        should_return_audio_clips: bool):
        
        files_processed = 0
        
        if remaining_input_samples is not None:
            input = remaining_input_samples
            golden = remaining_golden_samples
            remaining_input_samples = None
            remaining_golden_samples = None
        else:
            next_input = next(input_loader_iter, None)
            next_golden = next(golden_loader_iter, None)
            if next_input is None:
                return IterationResult(False, 0, None, None, None, None, None)
            
            input = next_input[0].squeeze(0).squeeze(0).to(self.device)
            golden = next_golden[0].squeeze(0).squeeze(0).to(self.device)
            del next_input, next_golden
            files_processed += 1
            
        while input.size()[0] < self.samples_per_iteration:
            next_input = next(input_loader_iter, None)
            next_golden = next(golden_loader_iter, None)
            if next_input is None:
                # Discard remainder.
                return IterationResult(False, 0, None, None, None, None, None)

            input = torch.cat((input, next_input[0].squeeze(0).squeeze(0).to(self.device)), dim=0)
            golden = torch.cat((golden, next_golden[0].squeeze(0).squeeze(0).to(self.device)), dim=0)
            
            del next_input, next_golden
            files_processed += 1
            
        input_limited = input[:self.samples_per_iteration]
        golden_limited = golden[:self.samples_per_iteration]

        if input.size()[0] > self.samples_per_iteration:
            remaining_input_samples = input[self.samples_per_iteration:]
            remaining_golden_samples = golden[self.samples_per_iteration:]

        del input, golden
                
        input_subsamples = torch.split(input_limited, self.samples_per_batch)
        golden_subsamples = torch.split(golden_limited, self.samples_per_batch)

        input_subsamples = torch.stack(input_subsamples).unsqueeze(1)
        golden_subsamples = torch.stack(golden_subsamples).unsqueeze(1)
        
        model_audio_clips = []
        for model_idx, model in enumerate(self.models):
            model.train()
            
            prediction = model(input_subsamples)
            
            if should_return_audio_clips:
                prediction_cpu = prediction.cpu().detach()

            for loss_name, loss_weight, loss_fn in self.loss_function_tuples:
                loss = loss_fn(prediction, golden_subsamples)
                loss = loss * loss_weight
                
                self.summary_writer.add_scalar(loss_name, loss.item())
                
            loss = sum([t[2] for t in self.loss_function_tuples])
            self.summary_writer.add_scalar(f"total_loss_model_{model_idx}", loss.item())

            del prediction, input_subsamples, golden_subsamples
            
            self.optimizer.zero_grad()
            
            loss.backward()
            
            self.summary_writer.add_scalar(f"memory_post_backprop_allocated_model_{model_idx}", torch.cuda.memory_allocated(0))
            self.summary_writer.add_scalar(f"memory_post_backprop_reserved_model_{model_idx}", torch.cuda.memory_reserved(0))
            self.summary_writer.add_scalar(f"memory_post_backprop_max_reserved_model_{model_idx}", torch.cuda.max_memory_reserved(0))
            
            del loss
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            self.model.eval()
        
            if should_return_audio_clips:
                if prediction_cpu is not None:
                    noisy_audio = input_limited.cpu().detach()
                    prediction_audio = prediction_cpu.view(25, -1).view(1, -1).cpu().detach()
                    clear_audio = golden_limited.cpu().detach()
                    self.summary_writer.add_audio("noisy_audio", noisy_audio, sample_rate=self.sample_rate, global_step=self.iteration_count)
                    self.summary_writer.add_audio("prediction_audio", prediction_audio, sample_rate=self.sample_rate, global_step=self.iteration_count)
                    self.summary_writer.add_audio("clear_audio", clear_audio, sample_rate=self.sample_rate, global_step=self.iteration_count)
                    model_audio_clips.append((noisy_audio, prediction_audio, clear_audio))
                else:
                    print(f"prediction_cpu is None!")
                    
        return IterationResult(
            continue_training=True,
            files_processed_during_iteration=files_processed,
            remaining_input_samples=remaining_input_samples,
            remaining_golden_samples=remaining_golden_samples,
            model_audio_clips=model_audio_clips
        )

@dataclass
class IterationResult():
    continue_training: bool
    files_processed_during_iteration: int
    remaining_input_samples: Optional[torch.Tensor]
    remaining_golden_samples: Optional[torch.Tensor]
    model_audio_clips: Optional[List[(torch.Tensor, torch.Tensor, torch.Tensor)]]
