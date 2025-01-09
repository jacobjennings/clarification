
## Gradient clipping schedule?

## LSTM at smaller model sizes / fix lstm models

## Revisit other layer shapes / architectures
Lower input sample rate, higher output sample rate
Super resolution

## vast.ai cli script

### Think about improving assembly of models
Should be easier to get correct layer sizes
Visual assembly
Helper functions to get expected outputs
Explainable parameter distribution
Activation visualization

### Quantization

### Pruning
Activation visualization

## Are there existing trainers that I can use as a reference to improve audio_trainer.py?

## Add spectrograms to tensorboard outputs

## Write custom loss functions
Try to normalize loudness of outputs
Clipping detection

*   Load a sample in a notebook, do stft and
    look for very high-frequency harmonics

## Try torchaudio.functional.rnnt_loss

## Turn on cudNN benchmarking
If your model architecture remains fixed and your input size stays constant, setting torch.backends.cudnn.benchmark = True might be beneficial

## Other optimizers

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

## Set up a system for listening tests
Rank "best sounding" samples with human ears
Use data to find good loss function configs


