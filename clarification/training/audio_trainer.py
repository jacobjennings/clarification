import os
import time
import logging

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

        if self.m.mixed_precision_config.use_scaler_dtype:
            self.s.scaler = GradScaler()

        # model_dict_path = models_dir + "/dense4-20241222-184328"
        # model = models[0][1]
        # model.load_state_dict(torch.load(model_dict_path, weights_only=True))
        # model.eval()

    def train_one_rotation(self):
        print(f"Model training rotation {self.s.rotation_count} start for {self.m.name} @ epoch {self.s.epoch_count}")

        if self.s.data_loader_iter is None:
            print(f"Loading training iterator for {self.m.name} on device {self.m.dataset_loader.pin_memory_device}")
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
            self.s.samples_processed += bpi * self.d.samples_per_iteration
            self.s.samples_processed_since_last_log += bpi * self.d.samples_per_iteration
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
                # Better to overwrite to avoid stacking up disk space.
                Path(self.l.model_weights_dir).mkdir(parents=True, exist_ok=True)
                model_save_path = self.l.model_weights_dir + f"/weights-{self.c.training_date_str}-{self.m.name}"
                self.w.add_text(f"model_save_path_{self.m.name}", model_save_path, self.s.batches_trained)
                if os.path.exists(model_save_path):
                    os.remove(model_save_path)
                torch.save(self.m.model.state_dict(), model_save_path)
                self.s.batches_since_last_save = 0

            if self.s.batches_trained >= self.m.mixed_precision_config.stop_amp_after_batches:
                self.s.scaler = None
                self.m.mixed_precision_config.use_scaler_dtype = None

            if batch_count_this_rotation >= self.m.batches_per_rotation:
                break

        self.s.rotation_count += 1

        if self.s.batches_since_last_validation >= self.m.validation_config.run_validation_every_batches:
            print(f"Running validation for {self.m.name} @ rotation {self.s.rotation_count}")

            self.run_validation()
            self.s.batches_since_last_validation = 0

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
        batches_complete = self.s.iteration_count * self.d.batches_per_iteration
        batches_per_second = batches_complete / elapsed_training_time
        self.w.add_scalar("epoch_percentage_complete",
                          batches_complete / self.m.dataset_batches_total_length * 100,
                          self.s.batches_trained)
        self.w.add_scalar("epoch_estimated_time_per_epoch_minutes",
                          self.m.dataset_batches_total_length / batches_per_second / 60,
                          self.s.batches_trained)

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
        if self.s.scaler and allow_mixed_precision:
            with autocast("cuda", dtype=self.m.mixed_precision_config.use_scaler_dtype):
                prediction_raw = self.run_model(input_unsqueezed)
            prediction_raw = self.s.scaler.scale(prediction_raw)
        else:
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

        _, weighted_losses = self.loss_calculation(
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

            # Log cuda memory
            # print("Before first backward")
            # print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024} MB")
            # print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024 / 1024} MB")
            # print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")

            while weighted_losses:
                loss = weighted_losses.pop()
                loss.backward(retain_graph=len(weighted_losses) != 0)
                del loss
                # print("After backward")
                # print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024} MB")
                # print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024 / 1024} MB")
                # print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")

            if self.s.scaler and allow_mixed_precision:
                self.s.scaler.unscale_(optimizer)  # Unscale before clipping

            if self.m.norm_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.m.norm_clip)

            if should_step_optimizer:
                if self.s.scaler and allow_mixed_precision:
                    self.s.scaler.step(optimizer)
                    self.s.scaler.update()
                else:
                    optimizer.step()

                if scheduler:
                    scheduler.step()

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
                if self.s.scaler and allow_mixed_precision:
                    with autocast("cuda", dtype=self.m.mixed_precision_config.use_scaler_dtype):
                        loss_out = loss_config.fn(prediction_loss)
                    loss_out = self.s.scaler.scale(loss_out)
                else:
                    loss_out = loss_config.fn(prediction_loss)
                loss_out = torch.mean(loss_out)
            else:
                if self.s.scaler and allow_mixed_precision:
                    with autocast("cuda", dtype=self.m.mixed_precision_config.use_scaler_dtype):
                        loss_out = loss_config.fn(prediction_loss, goldens_loss)
                    loss_out = self.s.scaler.scale(loss_out)
                else:
                    loss_out = loss_config.fn(prediction_loss, goldens_loss)

            if self.s.scaler and allow_mixed_precision:
                loss_out *= self.m.mixed_precision_config.amp_loss_scalar

            loss_out_weighted = loss_out * loss_config.weight

            if should_log_extra_stuff:
                self.w.add_scalar(f"{writer_tag_prefix}loss_weighted_{loss_config.name}",
                                  loss_out_weighted.item(), writer_step)
                self.w.add_scalar(f"{writer_tag_prefix}loss_{loss_config.name}",
                                  loss_out.item(), writer_step)
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
        self.s.data_loader_iter = iter(self.m.dataset_loader)

    def run_validation(self):
        validation_batch_count = 0
        input_loader_iter = iter(self.m.validation_config.test_loader)
        iterations_since_step = 0
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
                writer_step=self.s.batches_validated,
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
