import logging

logger = logging.getLogger(__name__)
import torch

def better_split_discard_remainder(tensor, split_size):
    splits = torch.split(tensor, split_size, dim=-1)

    # Check the size of the last group and discard if necessary
    if splits[-1].size(0) != split_size:
        splits = splits[:-1]

    return torch.cat(splits, 0)

class SqueezeIter:
    """
    An iterator that performs a squeeze(0) operation on each element.
    """
    def __init__(self, iterable):
        self.iterable = iter(iterable)

    def __iter__(self):
        return self

    def __next__(self):
        element = next(self.iterable)
        return element.squeeze(0)

class TensorChain:
    """
    An iterator that takes an iterable of torch tensors of shape [16, 2, 7200]
    and yields a series of tensors with shape [1, 2, 7200], similar to
    itertools.chain but for torch tensors.
    """
    def __init__(self, tensor_iter):
        self.tensor_iter = tensor_iter
        self.current_tensor = None
        self.inner_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:  # Loop until a valid tensor is found
            if self.current_tensor is None:
                try:
                    self.current_tensor = next(self.tensor_iter)
                except StopIteration:
                    raise StopIteration

            try:
                result = self.current_tensor[self.inner_index].unsqueeze(0)
                self.inner_index += 1
                return result
            except IndexError:
                # Move to the next tensor in the iterable
                self.current_tensor = None
                self.inner_index = 0

class RandomTensorIter:
    """
    An iterator that generates random tensors of a specified shape.
    """
    def __init__(self, shape, dtype=torch.float32):
        self.shape = shape
        self.dtype = dtype

    def __iter__(self):
        return self

    def __next__(self):
        return torch.randn(self.shape, dtype=self.dtype)


class TensorSplitter:
    def __init__(self, tensor, sample_size=300):
        """
        Initializes the TensorSplitter with the given tensor and sample size.

        Args:
          tensor: A torch tensor of shape [_, _, 7200].
          sample_size: An integer representing the size of each segment along the last dimension.
                       Defaults to 300.
        """
        self.tensor = tensor
        self.sample_size = sample_size
        self.current_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns the next segment of the tensor.

        Returns:
          A torch tensor of shape [_, _, sample_size] where dimension 0 increases with each iteration.

        Raises:
          StopIteration: If there are no more segments in the tensor.
        """
        if self.current_idx >= self.tensor.shape[-1]:
            raise StopIteration

        # Extract the segment from the tensor
        segment = self.tensor[:, :, self.current_idx:self.current_idx + self.sample_size]

        self.current_idx += self.sample_size
        return segment
