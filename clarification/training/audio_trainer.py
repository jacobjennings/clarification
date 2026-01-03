import os
import time
import logging
import glob
import json

logger = logging.getLogger(__name__)
import torch.utils.checkpoint
from torch.cpu.amp import GradScaler
from torch.amp import autocast
from torch.profiler import profile, record_function, ProfilerActivity
from pathlib import Path

from clarification.configs import *


class AudioTrainer:
    def __init__(self, config: AudioTrainerConfig):
        """Training loop for audio.
        
        Args:
            config: Configuration for the audio trainer.
                See clarification.training.[data_classes.py].AudioTrainerConfig.
        """

        self.c = config
        # Shorthand
        self.s = config.state
        self.m = config.model_training_config
        self.d = config.model_training_config.dataset_config
        self.l = config.log_behavior_config
        self.w = config.log_behavior_config.writer
        
        # Get overlap_samples from the data loader (read from dataset's info.csv)
        self.overlap_samples = self.m.dataset_loader.overlap_size

        # Only use GradScaler for float16 - bfloat16 has same exponent range as float32
        # and doesn't need gradient scaling to avoid underflow
        if self.m.mixed_precision_config.needs_grad_scaler:
            self.s.scaler = GradScaler()
            print(f"INFO: Using GradScaler for {self.m.mixed_precision_config.use_scaler_dtype} training")
        elif self.m.mixed_precision_config.use_scaler_dtype:
            print(f"INFO: Using {self.m.mixed_precision_config.use_scaler_dtype} without GradScaler (not needed for bfloat16)")

        # model_dict_path = models_dir + "/dense4-20241222-184328"
        # model = models[0][1]
        # model.load_state_dict(torch.load(model_dict_path, weights_only=True))
        # model.eval()

    def train_one_rotation(self):
        print(f"Model training rotation {self.s.rotation_count} start for {self.m.name} @ epoch {self.s.epoch_count}")

        if self.s.data_loader_iter is None:
            print(f"Loading training iterator for {self.m.name} on device {self.m.dataset_loader.pin_memory_device}")
            
            # Fast-forward to saved file index if resuming
            if self.s.data_loader_file_idx > 0:
                print(f"INFO: Resuming {self.m.name} data loader from file index {self.s.data_loader_file_idx}...")
                self.m.dataset_loader.skip_to_file(self.s.data_loader_file_idx)
            
            self.s.data_loader_iter = iter(self.m.dataset_loader)

        # TODO WRONG TIME CALCULATIONS WHEN PAUSED
        if self.s.train_start_time is None:
            self.s.train_start_time = time.time()

        if self.s.epoch_start_time is None:
            self.s.epoch_start_time = time.time()

        if self.s.last_samples_processed_log_time is None:
            self.s.last_samples_processed_log_time = time.time()

        iterations_since_step = 0
        batch_count_this_rotation = 0

        # Reset timing right before training loop starts to exclude any overhead
        # (model loading, validation from previous rotation, config_to_device, etc.)
        # Must reset both timer AND sample counter to maintain correct metric ratio
        self.s.last_samples_processed_log_time = time.time()
        self.s.samples_processed_since_last_log = 0

        # Loop iterations until batches_per_model_rotation is reached.
        while True:
            send_audio_clips = False
            if self.s.batches_since_last_send_audio > self.l.send_audio_clip_every_batches:
                send_audio_clips = True
                self.s.batches_since_last_send_audio = 0

            should_log_extra_stuff = self.s.batches_since_last_log + self.d.batches_per_iteration >= self.l.log_info_every_batches

            should_step_optimizer = False
            if iterations_since_step == self.m.step_every_iterations or self.m.step_every_iterations == 1:
                iterations_since_step = 0
                should_step_optimizer = True

            furthest_matmul_value = "highest"
            if self.m.mixed_precision_config.matmul_batch_count_to_precision:
                for matmul_batch_limit, matmul_value in self.m.mixed_precision_config.matmul_batch_count_to_precision.items():
                    if self.s.batches_trained > matmul_batch_limit:
                        furthest_matmul_value = matmul_value

            if self.m.mixed_precision_config.matmul_batch_count_to_precision and self.s.last_matmul_value != furthest_matmul_value:
                self.s.last_matmul_value = furthest_matmul_value
                torch.set_default_tensor_type(furthest_matmul_value)

            # TODO
            # should_profile = self.l.profile_every_batches and self.s.batches_since_last_profile >= self.l.profile_every_batches
            # if should_profile:
            #     self.s.batches_since_last_profile = 0
            #     torch.cuda.memory._record_memory_history(max_entries=100000)
            #
            #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            #         with record_function(f"profile_{self.m.name}_iteration"):
            #             self.run_iteration(
            #                 should_record_audio_clips=send_audio_clips,
            #                 should_log_extra_stuff=should_log_extra_stuff,
            #                 writer_step=self.s.batches_trained,
            #                 should_step_optimizer=should_step_optimizer,
            #                 allow_mixed_precision=True,
            #                 is_validation=False,
            #             )
            #
            #     profiling_file_path = f"{self.l.profiling_data_output_dir}/profile_{self.c.training_date_str}_{self.m.name}"
            #     Path(profiling_file_path).mkdir(parents=True, exist_ok=True)
            #     torch.cuda.memory._dump_snapshot(self.l.profiling_data_output_dir + "/profile_" + self.m.name)
            #     torch.cuda.memory._record_memory_history(enabled=None)
            #
            # else:
            self.run_iteration(
                should_record_audio_clips=send_audio_clips,
                should_log_extra_stuff=should_log_extra_stuff,
                writer_step=self.s.batches_trained,
                should_step_optimizer=should_step_optimizer,
                allow_mixed_precision=True,
                is_validation=False
            )

            bpi = self.d.batches_per_iteration
            batch_count_this_rotation += bpi
            iterations_since_step += 1

            self.s.batches_trained += bpi
            self.s.batches_since_last_save += bpi
            self.s.batches_since_last_send_audio += bpi
            self.s.samples_processed += self.d.samples_per_iteration
            self.s.samples_processed_since_last_log += self.d.samples_per_iteration
            self.s.iteration_count += 1
            self.s.batches_since_last_validation += bpi
            self.s.batches_since_last_profile += bpi
            self.s.batches_since_last_log += bpi

            if self.s.batches_since_last_log > self.l.log_info_every_batches:
                elapsed_training_time = time.time() - self.s.train_start_time
                elapsed_time_since_logged_samples_processed = time.time() - self.s.last_samples_processed_log_time

                self.log_post_iteration_stuff(elapsed_time_since_logged_samples_processed, elapsed_training_time)

                self.s.last_samples_processed_log_time = time.time()
                self.s.samples_processed_since_last_log = 0
                self.s.batches_since_last_log = 0
                self.w.flush()
                pprint.pprint(self.s, width=2)

            if self.s.batches_since_last_save >= self.l.model_weights_save_every_batches:
                # Save weights with step count in filename for resume capability
                Path(self.l.model_weights_dir).mkdir(parents=True, exist_ok=True)
                model_save_path = self.l.model_weights_dir + f"/weights-{self.s.batches_trained}-{self.m.name}"
                state_save_path = self.l.model_weights_dir + f"/state-{self.s.batches_trained}-{self.m.name}.json"
                self.w.add_text(f"model_save_path_{self.m.name}", model_save_path, self.s.batches_trained)
                
                # Remove old weights and state files to avoid disk space buildup (keep only latest)
                old_weights = glob.glob(self.l.model_weights_dir + f"/weights-*-{self.m.name}")
                old_states = glob.glob(self.l.model_weights_dir + f"/state-*-{self.m.name}.json")
                for old_path in old_weights + old_states:
                    if old_path != model_save_path and old_path != state_save_path:
                        try:
                            os.remove(old_path)
                        except:
                            pass
                
                torch.save(self.m.model.state_dict(), model_save_path)
                
                # Save training state for accurate resume (independent of batches_per_iteration)
                training_state = {
                    "data_loader_file_idx": self.m.dataset_loader.file_idx,
                    "epoch_count": self.s.epoch_count,
                    "samples_processed": self.s.samples_processed,
                    "batches_trained": self.s.batches_trained,
                }
                with open(state_save_path, "w", encoding="utf-8") as f:
                    json.dump(training_state, f)
                
                self.s.batches_since_last_save = 0

            if batch_count_this_rotation >= self.m.batches_per_rotation:
                break

        self.s.rotation_count += 1

        if self.s.batches_since_last_validation >= self.m.validation_config.run_validation_every_batches:
            print(f"Running validation for {self.m.name} @ rotation {self.s.rotation_count}")

            self.run_validation()
            self.s.batches_since_last_validation = 0
            
            # Reset the samples/time tracking so validation time isn't included
            # in the next training throughput measurement
            self.s.last_samples_processed_log_time = time.time()

    def memory_test_run(self):
        print(f"Memory test run for {self.m.name} started.")
        if self.s.data_loader_iter is None:
            self.s.data_loader_iter = iter(self.m.dataset_loader)

        for i in range(1):
            self.run_iteration(
                should_record_audio_clips=False,
                should_log_extra_stuff=False,
                writer_step=0,
                should_step_optimizer=True,
                allow_mixed_precision=False,
                is_validation=False
            )

    def log_post_iteration_stuff(self, elapsed_time_since_logged_samples_processed, elapsed_training_time):
        self.w.add_scalar("samples_processed", self.s.samples_processed,
                          self.s.batches_trained)
        self.w.add_scalar("samples_processed_per_microsecond",
                          self.s.samples_processed_since_last_log / elapsed_time_since_logged_samples_processed / 1000 / 1000,
                          self.s.batches_trained)
        self.w.add_scalar("iterations_per_second",
                          self.s.iteration_count / elapsed_training_time,
                          self.s.batches_trained)
        
        # Calculate epoch percentage based on file index
        current_file_idx = self.m.dataset_loader.file_idx
        total_files = self.m.dataset_loader.total_files
        if total_files > 0:
            # Track file index within current epoch
            file_idx_in_epoch = current_file_idx % total_files
            epoch_pct = file_idx_in_epoch / total_files * 100
            self.w.add_scalar("epoch_percentage_complete", epoch_pct, self.s.batches_trained)
            self.w.add_scalar("files_processed_in_epoch", file_idx_in_epoch, self.s.batches_trained)
            
            # Estimate time per epoch based on files/second
            if elapsed_training_time > 0:
                files_per_second = current_file_idx / elapsed_training_time
                if files_per_second > 0:
                    estimated_epoch_minutes = total_files / files_per_second / 60
                    self.w.add_scalar("epoch_estimated_time_per_epoch_minutes", 
                                      estimated_epoch_minutes, self.s.batches_trained)
            
            # Save file_idx for resume
            self.s.data_loader_file_idx = current_file_idx

    def run_iteration(
            self,
            should_record_audio_clips: bool,
            should_log_extra_stuff: bool,
            writer_step: int,
            should_step_optimizer: bool,
            allow_mixed_precision: bool,
            is_validation: bool,
            input_loader_iter=None
    ):
        print(".", end="")

        if input_loader_iter is None:
            input_loader_iter = self.s.data_loader_iter

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data preparation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        writer_tag_prefix = "validation_" if is_validation else ""
        perf_iteration_start = None
        if should_log_extra_stuff:
            perf_iteration_start = time.perf_counter()

        input_subsamples, golden_reconstructed, golden_classifier_values = self.iteration_data_prep(input_loader_iter)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Log prep ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if should_log_extra_stuff and perf_iteration_start:
            perf_data_prep_end = time.perf_counter()
            self.w.add_scalar(
                f"{writer_tag_prefix}perf_data_prep", perf_data_prep_end - perf_iteration_start, writer_step)

        if should_record_audio_clips:
            # self.record_noisy_clear_audio_clips(golden_reconstructed, input_subsamples, writer_step, writer_tag_prefix)
            self.record_noisy_clear_audio_clips(
                golden_reconstructed=golden_reconstructed,
                input_subsamples=input_subsamples,
                writer_step=writer_step,
                writer_tag_prefix=writer_tag_prefix
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # torch.cuda.memory._record_memory_history(max_entries=100000)

        model = self.m.training_model()
        optimizer = self.m.optimizer
        scheduler = self.m.scheduler

        if is_validation:
            model.eval()
        else:
            model.train()

        input_unsqueezed = input_subsamples.unsqueeze(dim=1)
        use_amp = self.m.mixed_precision_config.use_scaler_dtype and allow_mixed_precision
        if use_amp:
            with autocast("cuda", dtype=self.m.mixed_precision_config.use_scaler_dtype):
                prediction_raw = self.run_model(input_unsqueezed)
        else:
            # Cast input to float32 when not using mixed precision (e.g., during validation)
            # to match model weight dtype
            input_unsqueezed = input_unsqueezed.float()
            prediction_raw = self.run_model(input_unsqueezed)

        if not self.m.training_classifier:
            prediction_raw = prediction_raw.squeeze(dim=1)

        prediction_cpu = None
        if not self.m.training_classifier:
            prediction = self.reconstruct_overlapping_samples_nofade(prediction_raw)

            if should_record_audio_clips:
                prediction_cpu = prediction.cpu().detach()

        else:
            prediction = prediction_raw

        # torch.cuda.empty_cache()

        # torch.cuda.memory._dump_snapshot(self.profile_output_dir)
        # torch.cuda.memory._record_memory_history(enabled=None)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Loss calculation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # print("After prediction")
        # print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024} MB")
        # print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024 / 1024} MB")
        # print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")

        total_loss, _ = self.loss_calculation(
            golden_classifier_values,
            golden_reconstructed,
            prediction,
            should_log_extra_stuff,
            writer_step,
            writer_tag_prefix,
            allow_mixed_precision)

        if should_log_extra_stuff:
            # Log memory at the usual worst case spot, after loss calculation
            memory_allocated_gb = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
            memory_reserved_gb = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
            memory_max_allocated_gb = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            self.w.add_scalar(f"{writer_tag_prefix}memory_allocated_gb", memory_allocated_gb, writer_step)
            self.w.add_scalar(f"{writer_tag_prefix}memory_reserved_gb", memory_reserved_gb, writer_step)
            self.w.add_scalar(f"{writer_tag_prefix}memory_max_allocated_gb", memory_max_allocated_gb, writer_step)

        del prediction

        if not is_validation:
            optimizer.zero_grad(set_to_none=True)

            # Single backward pass on total loss (much more memory efficient than
            # multiple backwards with retain_graph=True)
            use_grad_scaler = self.s.scaler is not None and allow_mixed_precision
            if use_grad_scaler:
                self.s.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            del total_loss

            if use_grad_scaler:
                self.s.scaler.unscale_(optimizer)  # Unscale gradients before clipping

            # Gradient clipping with logging
            if self.m.norm_clip:
                # clip_grad_norm_ returns the total norm BEFORE clipping
                pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.m.norm_clip)
                if should_log_extra_stuff:
                    self.w.add_scalar(f"{writer_tag_prefix}grad_norm_pre_clip", pre_clip_norm.item(), writer_step)
                    # Log whether clipping was applied (1 if clipped, 0 if not)
                    was_clipped = 1.0 if pre_clip_norm > self.m.norm_clip else 0.0
                    self.w.add_scalar(f"{writer_tag_prefix}grad_clip_applied", was_clipped, writer_step)

            if should_step_optimizer:
                if use_grad_scaler:
                    self.s.scaler.step(optimizer)
                    self.s.scaler.update()
                else:
                    optimizer.step()

                if scheduler:
                    scheduler.step()
                    # Log learning rate to TensorBoard
                    current_lr = scheduler.get_last_lr()[0]
                    self.w.add_scalar(f"learning_rate", current_lr, self.s.batches_trained)

            model.eval()

        if should_record_audio_clips:
            self.record_prediction_audio_clips(prediction_cpu, prediction_raw, writer_step, writer_tag_prefix)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Log things to tensorboard ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if should_log_extra_stuff:
            self.log_extra_stuff(perf_iteration_start, writer_step, writer_tag_prefix)

    def run_model(self, input_data: torch.Tensor):
        # def policy_fn(ctx, op, *args, **kwargs):
        #     return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
        #
        # context_fn = functools.partial(torch.utils.checkpoint.create_selective_checkpoint_contexts, policy_fn)
        #
        # x = input_data
        # outputs = None
        # initial_x = input_data
        #
        # for i in range(self.m.model.checkpoint_count()):
        #     x, initial_x, outputs = torch.utils.checkpoint.checkpoint(
        #         self.m.model.compute_checkpoint,
        #         x,
        #         initial_x,
        #         outputs,
        #         i,
        #         use_reentrant=False,
        #         context_fn=context_fn)

        x = self.m.training_model()(input_data)
        return x

    def log_extra_stuff(self, perf_iteration_start, writer_step, writer_tag_prefix):
        perf_iteration_end = time.perf_counter()
        self.w.add_scalar(f"{writer_tag_prefix}perf_iteration",
                          perf_iteration_end - perf_iteration_start, writer_step)

    def record_prediction_audio_clips(self, prediction_cpu, prediction_raw, writer_step, writer_tag_prefix):
        if self.m.training_classifier:
            self.w.add_scalar(f"{writer_tag_prefix}classifier_prediction", prediction_raw.mean(),
                              writer_step),
        else:
            if prediction_cpu is not None:
                self.w.add_audio(f"{writer_tag_prefix}prediction_audio", prediction_cpu,
                                 sample_rate=self.d.sample_rate, global_step=writer_step)
            else:
                print("prediction_cpu is None!")

    def loss_calculation(self, golden_classifier_values, golden_reconstructed, prediction,
                         should_log_extra_stuff, writer_step, writer_tag_prefix, allow_mixed_precision):
        loss = None
        loss_unweighted_total = 0.0  # Track unweighted sum for stable comparison
        weighted_losses = []
        for loss_config in self.m.loss_function_configs:
            if self.m.training_classifier and prediction.size() != golden_classifier_values.size():
                print(
                    f"Wrong golden values size! prediction.size() = {prediction.size()} golden_classifier_values.size() = {golden_classifier_values.size()}")

            goldens = golden_classifier_values if self.m.training_classifier else golden_reconstructed

            # Cast to float for loss calculation as many loss functions (like MelSTFTLoss) don't support Half
            prediction_loss = prediction.float()
            goldens_loss = goldens.float() if goldens is not None else None

            if loss_config.batch_size:
                prediction_loss = better_split_discard_remainder(prediction_loss, loss_config.batch_size)
                goldens_loss = better_split_discard_remainder(goldens_loss, loss_config.batch_size)

            if loss_config.is_unary:
                # TODO             next_input = next_input.view(-1, self.d.samples_per_batch * self.d.dataset_batch_size)
                loss_out = loss_config.fn(prediction_loss)
                loss_out = torch.mean(loss_out)
            else:
                loss_out = loss_config.fn(prediction_loss, goldens_loss)
            # Note: GradScaler.scale() is applied during backward(), not here.
            # Loss values are computed and logged without scaling.

            # Get dynamic weight based on current step (supports scheduled weights)
            current_weight = loss_config.get_weight(self.s.batches_trained)
            loss_out_weighted = loss_out * current_weight
            loss_unweighted_total += loss_out.item()

            if should_log_extra_stuff:
                self.w.add_scalar(f"{writer_tag_prefix}loss_weighted_{loss_config.name}",
                                  loss_out_weighted.item(), writer_step)
                self.w.add_scalar(f"{writer_tag_prefix}loss_{loss_config.name}",
                                  loss_out.item(), writer_step)
                # Log current weight so you can see the schedule in action
                self.w.add_scalar(f"{writer_tag_prefix}weight_{loss_config.name}",
                                  current_weight, writer_step)
            if loss:
                loss = loss + loss_out_weighted
            else:
                loss = loss_out_weighted

            weighted_losses.append(loss_out_weighted)

        if should_log_extra_stuff:
            total_norm = 0
            for p in self.m.training_model().parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)  # Calculate L2 norm
                    total_norm += param_norm.item() ** 2
                    # if should_log_extra_stuff:
                    #     model_config.writer.add_scalar(f"Gradients/{model_config.name}/{p.grad.data.norm}", param_norm,
                    #                                    writer_step)  # Log individual parameter gradient norms
            total_norm = total_norm ** 0.5
            self.w.add_scalar(f"{writer_tag_prefix}total_grad_norm", total_norm, writer_step)

            self.w.add_scalar(f"{writer_tag_prefix}loss_total", loss.item(), writer_step)
            # Unweighted total: stable metric for comparing across different weight schedules
            self.w.add_scalar(f"{writer_tag_prefix}loss_total_unweighted", loss_unweighted_total, writer_step)

        return loss, weighted_losses

    def record_noisy_clear_audio_clips(self, golden_reconstructed, input_subsamples, writer_step,
                                       writer_tag_prefix):
        noisy_audio = self.reconstruct_overlapping_samples_nofade(
            input_subsamples.view(-1, self.d.samples_per_batch)).cpu().detach()
        self.w.add_audio(f"{writer_tag_prefix}noisy_audio", noisy_audio,
                                      sample_rate=self.d.sample_rate, global_step=writer_step)
        if not self.m.training_classifier:
            clear_audio = golden_reconstructed.cpu().detach()

            self.w.add_audio(f"{writer_tag_prefix}clear_audio", clear_audio,
                                          sample_rate=self.d.sample_rate, global_step=writer_step)

    def iteration_data_prep(self, loader_iter):
        golden_classifier_values = None
        if self.m.training_classifier:
            next_input, golden_classifier_values = next(loader_iter, None)
            if next_input is None:
                self.reset_epoch()
                loader_iter = self.s.data_loader_iter
                next_input, golden_classifier_values = next(loader_iter, None)

            golden_classifier_values = golden_classifier_values.mean(dim=1)

        else:
            next_input = next(loader_iter, None)
            if next_input is None:
                self.reset_epoch()
                loader_iter = self.s.data_loader_iter
                next_input = next(loader_iter, None)

            next_input = next_input.view(-1, 2, self.d.samples_per_batch)
            next_input = next_input.squeeze(0).permute(1, 0, 2)

        input_subsamples = next_input.squeeze(0)
        if not self.m.training_classifier:
            input_subsamples = input_subsamples[0]

        golden_reconstructed = None
        if not self.m.training_classifier:
            golden_subsamples = next_input.squeeze(0)[1]
            golden_reconstructed = self.reconstruct_overlapping_samples_nofade(golden_subsamples)

        del next_input

        return input_subsamples, golden_reconstructed, golden_classifier_values

    def reset_epoch(self):
        print(f"Resetting epoch / iter for {self.m.name}")
        self.s.epoch_count += 1
        # Explicitly reset the loader to start from the beginning of the dataset
        self.m.dataset_loader.reset()
        self.s.data_loader_iter = iter(self.m.dataset_loader)

    def run_validation(self):
        validation_batch_count = 0
        # Explicitly reset validation loader to start from the beginning
        self.m.validation_config.test_loader.reset()
        input_loader_iter = iter(self.m.validation_config.test_loader)
        iterations_since_step = 0
        
        # Disable gradient computation during validation to save VRAM
        with torch.no_grad():
            while True:
                should_log_extra_stuff = (self.s.batches_since_last_log + self.d.batches_per_iteration
                                          > self.m.validation_config.log_every_batches)
                should_step_optimizer = False
                if iterations_since_step == self.m.step_every_iterations or self.m.step_every_iterations == 1:
                    iterations_since_step = 0
                    should_step_optimizer = True

                self.run_iteration(
                    input_loader_iter=input_loader_iter,
                    should_record_audio_clips=False,
                    should_log_extra_stuff=should_log_extra_stuff,
                    writer_step=self.s.batches_trained,  # Use training step so validation aligns with training in TensorBoard
                    should_step_optimizer=should_step_optimizer,
                    allow_mixed_precision=False,
                    is_validation=True
                )
                bpi = self.d.batches_per_iteration

                iterations_since_step += 1
                validation_batch_count += bpi
                self.s.batches_validated += bpi
                self.s.batches_since_last_log += bpi

                if should_log_extra_stuff:
                    self.s.batches_since_last_log = 0

                if validation_batch_count >= self.m.validation_config.test_batches:
                    break

    def reconstruct_overlapping_samples_nofade(self, samples: torch.Tensor):
        num_batches = samples.size()[0]
        non_overlapping_size = self.d.samples_per_batch - self.overlap_samples * 2

        total_length = (num_batches * (self.d.samples_per_batch - self.overlap_samples)) - self.overlap_samples * 2
        output = torch.zeros(total_length, dtype=samples.dtype, device=samples.device)

        output[:non_overlapping_size + self.overlap_samples] += samples[0][self.overlap_samples:]

        for idx, batch in enumerate(samples[1:-1]):
            idx = idx + 1
            start = (idx * non_overlapping_size) + ((idx - 1) * self.overlap_samples)
            end = start + non_overlapping_size + self.overlap_samples * 2
            output[start:end] = batch

        output[-(non_overlapping_size + self.overlap_samples):] += samples[-1][:-self.overlap_samples]

        return output.unsqueeze(dim=0).unsqueeze(dim=0)
