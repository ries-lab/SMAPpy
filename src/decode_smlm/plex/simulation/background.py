from abc import ABC, abstractmethod  # abstract class
from typing import Callable, Iterable, Optional, Union, Sequence

import math
import torch

from ..utils import dev


class Background(ABC):
    def __init__(self, size: Union[tuple[int, ...], torch.Size], device: str = "cpu"):
        """
        Background
        """
        super().__init__()

        self._size = size
        self._device = device

    @abstractmethod
    def sample(
        self, size: Union[tuple[int, ...], torch.Size], device: str = "cpu"
    ) -> torch.Tensor:
        """
        Samples from background implementation in the specified size.

        Args:
            size: size of the sample
            device: from which device to sample from

        """
        raise NotImplementedError

    def sample_like(self, x: torch.Tensor) -> torch.Tensor:
        """
        Samples background in the shape and on the device as the input.

        Args:
            x: input

        Returns:
            background sample

        """
        return self.sample(size=x.size(), device=x.device)

    def _arg_defaults(
        self, size: Optional[Union[tuple[int, ...], torch.Size]], device: str
    ):
        """Overwrite optional args with instance defaults."""
        size = self._size if size is None else size
        device = self._device if device is None else device

        return size, device


class BackgroundUniform(Background):
    def __init__(
        self,
        bg: Union[float, tuple, Callable],
        size: Optional[Union[tuple[int, ...], torch.Size]] = None,
        device: str = "cpu",
    ):
        """
        Spatially constant background (i.e. a constant offset).

        Args:
            bg: background value, range or callable to sample from
            size:
            device:

        """
        super().__init__(size=size, device=device)

        if callable(bg):
            self._bg_dist = bg
        elif isinstance(bg, (int, float)) or len(set(bg)) == 1:
            bg = bg if isinstance(bg, (int, float)) else bg[0]
            self._bg_dist = DeltaSampler(bg)
        elif isinstance(bg, Iterable):
            self._bg_dist = torch.distributions.uniform.Uniform(
                *bg,
            ).sample
        else:
            raise NotImplementedError

    def sample(
        self,
        size: Optional[Union[tuple[int, ...], torch.Size]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        size = size if size is not None else self._size
        device = device if device is not None else self._device

        if len(size) not in (2, 3, 4):
            raise NotImplementedError("Not implemented size spec.")

        # create as many sample as there are batch-dims
        bg = self._bg_dist(
            sample_shape=[size[0]] if len(size) >= 3 else torch.Size([]),
        )

        # unsqueeze until we have enough dimensions
        if len(size) >= 3:
            bg = bg.view(-1, *((1,) * (len(size) - 1)))

        return bg.to(device) * torch.ones(size, device=device)


class PerlinBackground(Background):
    """
    Taken from https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57.
    """

    def __init__(self, img_size, perlin_scale: int, amplitude, draw_amp: bool = False):
        """
        :param img_size: size of the image
        :param perlin_scale: scale of the perlin in fraction of the img_scale
        :param amplitude: background strength
        :param draw_amp: draw the perlin amplitude from a uniform distribution
        """
        super().__init__(size=img_size)
        if img_size[0] != img_size[1]:
            raise ValueError("Currently only equal img-size supported.")

        # self.img_size = img_size
        self.perlin_scale = perlin_scale
        self.amplitude = amplitude
        self.perlin_com = None
        self.draw_amp = draw_amp

        delta = (
            self.perlin_scale / self._size[0],
            self.perlin_scale / self._size[1],
        )
        self.d = (
            self._size[0] // self.perlin_scale,
            self._size[1] // self.perlin_scale,
        )
        self.grid = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, self.perlin_scale, delta[0]),
                    torch.arange(0, self.perlin_scale, delta[1]),
                ),
                dim=-1,
            )
            % 1
        )

    @staticmethod
    def fade_f(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def calc_perlin(self, shape, res):

        if shape[0] == res[0] and shape[1] == res[1]:
            return torch.rand(*shape) * 2 - 1

        angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
        gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

        tile_grads = (
            lambda slice1, slice2: gradients[
                slice1[0] : slice1[1], slice2[0] : slice2[1]
            ]
            .repeat_interleave(self.d[0], 0)
            .repeat_interleave(self.d[1], 1)
        )
        dot = lambda grad, shift: (
            torch.stack(
                (
                    self.grid[: shape[0], : shape[1], 0] + shift[0],
                    self.grid[: shape[0], : shape[1], 1] + shift[1],
                ),
                dim=-1,
            )
            * grad[: shape[0], : shape[1]]
        ).sum(dim=-1)

        n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
        t = self.fade_f(self.grid[: shape[0], : shape[1]])
        return math.sqrt(2) * torch.lerp(
            torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1]
        )

    @dev.experimental(False)
    def sample(self, size: (torch.Size, tuple), device=torch.device("cpu")):

        if self.draw_amp:
            amp_factor = torch.rand(1)
        else:
            amp_factor = 1.0

        if not isinstance(size, torch.Size):
            size = torch.Size(size)

        assert (
            len(size) == 3
        ), f"Assuming size specification to be N x H x W (first is batch dimension)."
        assert (
            size[-2:] == self._size
        ), "Perlin background initialised with different img_shape specification."

        bg_sample = torch.empty(size)
        for s in range(size[0]):
            bg_sample[s] = (
                self.amplitude
                * amp_factor
                * (
                    self.calc_perlin(
                        self._size, [self.perlin_scale, self.perlin_scale]
                    )
                    + 1
                )
                / 2.0
            )

        return bg_sample.to(device)


class Merger:
    def __init__(self, kernel: Optional[Callable] = None):
        """
        Combine frame and background

        Args:
            kernel: kernel to combine frame and bg, if None, default is simple addition
        """
        self._kernel = kernel if kernel is not None else self._kernel_default

    def forward(
        self,
        frame: Union[torch.Tensor, Sequence[torch.Tensor]],
        bg: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]],
    ) -> torch.Tensor:
        if bg is None:
            return frame

        if isinstance(frame, Sequence) != isinstance(bg, Sequence):
            raise NotImplementedError(
                "Either none or both frame and bg must be " "Sequence."
            )
        elif isinstance(frame, Sequence):
            if len(frame) != len(bg):
                raise ValueError("Sequence of unequal length.")
            frame = [self._kernel(f, bg) for f, bg in zip(frame, bg)]
        else:
            frame = self._kernel(frame, bg)

        return frame

    @classmethod
    def _kernel_default(cls, frame: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
        return frame + bg


def _no_add(frame: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
    return frame


class MergerNoOp(Merger):
    def __init__(self):
        super().__init__(kernel=_no_add)


class DeltaSampler:
    def __init__(self, val: float):
        """Samples a constant value."""
        self._val = val

    def __call__(self, sample_shape) -> float:
        return self._val * torch.ones(sample_shape)
