import torch

def better_split_discard_remainder(tensor, split_size):
    splits = torch.split(tensor, split_size, dim=-1)

    # Check the size of the last group and discard if necessary
    if splits[-1].size(0) != split_size:
        splits = splits[:-1]

    return torch.cat(splits, 0)
