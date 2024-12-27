## Are there existing trainers that I can use as a reference to improve audio_trainer.py?

## Add spectrograms to tensorboard outputs

## Use a separate summary writer for training and validation
Different directory, make it easier to compare, less data

## TODO: HOW TO UNDERSTAND GRADIENTS FROM LOSSES

## Write custom loss functions
Try to normalize loudness of outputs
Clipping detection

*   Load a sample in a notebook, do stft and
    look for very high-frequency harmonics

## Try torchaudio.functional.rnnt_loss

## Try pytorch profiler

## Super resolution

## Get LSTM models fixed up

## Turn on cudNN benchmarking
If your model architecture remains fixed and your input size stays constant, setting torch.backends.cudnn.benchmark = True might be beneficial

## Use gradient accumulation
Another approach to increasing the batch size is to accumulate gradients across multiple .backward() passes before calling optimizer.step().
MEMORY USAGE -> BIGGER BATCH

## Other optimizers

## Use Automatic Mixed Precision (AMP)
The release of PyTorch 1.6 included a native implementation of Automatic Mixed Precision training to PyTorch. The main idea here is that certain operations can be run faster and without a loss of accuracy at semi-precision (FP16) rather than in the single-precision (FP32) used elsewhere. AMP, then, automatically decide which operation should be executed in which format. This allows both for faster training and a smaller memory footprint.

In the best case, the usage of AMP would look something like this:

```python
import torch
# Creates once at the beginning of training
scaler = torch.cuda.amp.GradScaler()

for data, label in data_iter:
   optimizer.zero_grad()
   # Casts operations to mixed precision
   with torch.cuda.amp.autocast():
      loss = model(data)

   # Scales the loss, and calls backward()
   # to create scaled gradients
   scaler.scale(loss).backward()

   # Unscales gradients and calls
   # or skips optimizer.step()
   scaler.step(optimizer)

   # Updates the scale for next iteration
   scaler.update()
   
```

## torch.set_float32_matmul_precision

## Use CUDA Graphs
At the time of using a GPU, work first must be launched from the CPU and in some cases the context switch between CPU and GPU can lead to bad resource utilization. CUDA graphs are a way to keep computation within the GPU without paying the extra cost of kernel launches and host synchronization.

It can be enabled using
torch.compile(m, "reduce-overhead")
or
torch.compile(m, "max-autotune")

## Disable bias for convolutions directly followed by a batch norm
torch.nn.Conv2d() has bias parameter which defaults to True (the same is true for Conv1d and Conv3d ).

If a nn.Conv2d layer is directly followed by a nn.BatchNorm2d layer, then the bias in the convolution is not needed, instead use nn.Conv2d(..., bias=False, ....). Bias is not needed because in the first step BatchNorm subtracts the mean, which effectively cancels out the effect of bias.

## Speech Enhancement

## Look at different data models

## Find standardized benchmarks

## Prepare for publication

## Prepare a presentation for book club

Benchmarking a number of common language and vision models on NVIDIA V100 GPUs, Huang and colleagues find that using AMP over regular FP32 training yields roughly 2x – but upto 5.5x – training speed-ups.



Validation Loop: Add a validation loop to evaluate the model's performance on a validation dataset after each epoch. This helps in monitoring overfitting and generalization.  

Learning Rate Scheduler: Ensure the learning rate scheduler is properly integrated and its step is called at the right time.  

Gradient Clipping: Implement gradient clipping to prevent exploding gradients, which can be useful for training stability.  

Mixed Precision Training: Use mixed precision training to speed up training and reduce memory usage.  

Early Stopping: Implement early stopping to halt training when the validation performance stops improving.  

Model Checkpointing: Save model checkpoints based on validation performance, not just iteration count.  

Data Augmentation: Apply data augmentation techniques to improve model robustness.  

Logging Additional Metrics: Log additional metrics such as learning rate, gradient norms, and validation loss.

