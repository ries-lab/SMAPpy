import torch
import torch.nn.functional as F


def place(
    rois: torch.Tensor, positions: torch.Tensor, frame_size: tuple | torch.Size
) -> torch.Tensor:
    # pad frames
    pad_size = (rois.size(1), rois.size(2))
    frames = torch.zeros(*frame_size, device=rois.device)
    frames = F.pad(frames, (pad_size[0], pad_size[0], pad_size[1], pad_size[1]))

    # adjust positions
    positions = positions.clone()
    positions[:, 1:] += pad_size[0]

    # calculate valid RoI and frame indices
    valid_indices = (
        # fmt: off
        (0 <= positions[:, 0]) & (positions[:, 0] < frames.size(0))
        & (0 <= positions[:, 1]) & (positions[:, 1] < frames.size(1) - pad_size[0])
        & (0 <= positions[:, 2]) & (positions[:, 2] < frames.size(2) - pad_size[1])
        # fmt: on
    )

    indices = torch.where(valid_indices)[0]
    batch_indices = positions[indices, 0]
    x_indices = positions[indices, 1]
    y_indices = positions[indices, 2]

    x_ranges = torch.arange(rois.size(1), device=rois.device).expand(
        len(indices), -1
    ) + x_indices.unsqueeze(-1)
    y_ranges = torch.arange(rois.size(2), device=rois.device).expand(
        len(indices), -1
    ) + y_indices.unsqueeze(-1)

    # Use index_put_ with accumulate=True to correctly sum up the overlapping ROIs
    frames.index_put_(
        (batch_indices[:, None, None], x_ranges[:, :, None], y_ranges[:, None, :]),
        rois[indices],
        accumulate=True,
    )

    # remove excess padding
    frames = frames[..., pad_size[0] : -pad_size[0], pad_size[1] : -pad_size[1]]

    return frames
