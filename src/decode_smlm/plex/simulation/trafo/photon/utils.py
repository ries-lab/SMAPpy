import torch


def dual_from_ratio(r: torch.Tensor, dim_ch: int) -> torch.Tensor:
    """
    Get choric transformation matrix from channel 0 / total phot ratio for dual channel.

    Args:
        r: ratio of channel 0 / total photons
        dim_ch: channel dimension
    """
    if r.dim() != 1:
        raise ValueError("The input must be a 1D tensor.")
    return torch.stack([r, 1 - r], dim=dim_ch)
