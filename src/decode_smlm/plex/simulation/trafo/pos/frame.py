from abc import abstractmethod
from typing import Callable

import torch
import pydantic

from ....generic import mixin


def _pad2torch(x_low, x_high, y_low, y_high):
    """Convert decode convention to pytorch convention"""
    # return (y_low, y_high, x_low, x_high)
    raise NotImplementedError


# suboptimal name
@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=True))
def frame_shift_crop_pad(
    frame: torch.Tensor,
    x: pydantic.StrictInt,
    y: pydantic.StrictInt,
    mode: str = "constant",
    value: float = 0.0,
) -> torch.Tensor:
    """
    Apply global shift to frame. Useful when global best shift is computed and frames
    shall be shifted accordingly.

    A positive shift in x/y will pad the frame left/top, a negative shift will crop
    left/top and pad right/bottom such that the frame size is preserved.

    This function is typically used in conjunction with `Register` and global shifts
    to shift frames according to the best global shift. For example a transformation
    between two channels with a offset of (2, -3) would result (after Register) in a
    best global shift of (-2, 3) for the second channel. This function should then be
    called with x=-2 and y=3 to shift the frame accordingly.

    Args:
        frame: frame stack
        x: shift in x (e.g. by global best shift)
        y: shift in y (e.g. by global best shift)
        mode: padding mode (pytorch convention)
        value: value to pad with

    """
    crop = [None] * 2
    pad_lt = [None] * 2
    pad_rd = [None] * 2

    for i, v in enumerate((x, y)):
        if v >= 0:
            crop[i] = 0
            pad_lt[i] = v
            pad_rd[i] = 0
        else:
            crop[i] = abs(v)
            pad_lt[i] = 0
            pad_rd[i] = abs(v)

    out = torch.nn.functional.pad(
        frame[..., crop[0] :, crop[1] :],
        pad=(pad_lt[1], pad_rd[1], pad_lt[0], pad_rd[0]),
        mode=mode,
        value=value,
    )  # left,right,top,bottom in pytorch convention
    f_shape = list(frame.shape[-2:])
    out = out[..., : f_shape[-2], : f_shape[-1]]
    return out


class FrameTransformation(mixin.ForwardCallAlias, mixin.MultiDevice):
    @abstractmethod
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GenericFrameTransformation(FrameTransformation):
    def __init__(self, fn: Callable, device: str | torch.device):
        self._fn = fn
        self._device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: str | torch.device):
        raise NotImplementedError

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        return self._fn(frame)


class FrameShiftCropPad(mixin.ForwardCallAlias):
    def __init__(self, x: int, y: int, mode: str = "constant", value: float = 0.0):
        self._x = x
        self._y = y
        self._mode = mode
        self._value = value

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        return frame_shift_crop_pad(
            frame, self._x, self._y, mode=self._mode, value=self._value
        )
