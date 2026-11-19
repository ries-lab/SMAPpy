import torch


def is_integer(x: torch.Tensor) -> bool:
    """Returns True if x is an integer tensor, False otherwise."""
    return x.dtype in (torch.int8, torch.uint8, torch.short, torch.int, torch.long)
