from typing import Callable, Optional, Union, Sequence

import torch

from ...generic import delay
from ...generic import indexing


def bg_samples(
    bg: torch.Tensor | Sequence[torch.Tensor],
) -> Union[torch.Tensor, delay._DelayedSlicer]:
    """
    Get bg samples. Expects bg to tensor or sequence of tensors
    Args:
        bg:

    Returns:

    """
    if not isinstance(bg, torch.Tensor):
        bg = delay._InterleavedSlicer(bg)
    else:
        bg = bg.unsqueeze(1)

    return bg


def frame_samples(
    frames: torch.Tensor | list[torch.Tensor], window: Optional[int] = None
) -> indexing._WindowDelayed:
    """
    Get frame samples. Expects frames to tensor or tuple of sequence of size
    (N, H, W) or (N, C, H, W), C being the frame window. The tuple must be of
    length of the number of channels.

    Args:
        frames: frame tensor or tuple of tensors
        window: which window to use. If None, no windowing is applied.

    Returns:
        lazy tensor or tuple of tensor like object to access the frame samples.

    """
    # this is to auto-combine the tuple elements if tuple of tensors
    # if frames are channels
    frames = (
        delay._InterleavedSlicer(frames)
        if isinstance(frames, (tuple, list))
        else frames
    )

    if window is not None:
        frames = indexing.IxWindow(window, None).attach(frames, auto_recurse=False)

    return frames


def inference_samples(samples, aux: Optional, pre_fn: Callable) -> delay._DelayedSlicer:
    s = delay._DelayedTensor(
        pre_fn,
        kwargs={
            "frame": samples,
        },
        kwargs_static={"aux": aux},
    )
    size = torch.Size([len(samples), *(s[0].size())])
    s._size = size
    return s
