import copy
from typing import Tuple, Any, Callable, Union, Optional

import numpy as np
import torch


def cum_count_per_group(arr: torch.Tensor):
    """
    Helper function that returns the cumulative sum per group.

    Example:
        [0, 0, 0, 1, 2, 2, 0] --> [0, 1, 2, 0, 0, 1, 3]
    """

    def grp_range(counts: torch.Tensor):
        # ToDo: Add docs
        assert counts.dim() == 1

        idx = counts.cumsum(0)
        id_arr = torch.ones(idx[-1], dtype=int)
        id_arr[0] = 0
        id_arr[idx[:-1]] = -counts[:-1] + 1
        return id_arr.cumsum(0)

    if arr.numel() == 0:
        return arr

    _, cnt = torch.unique(arr, return_counts=True)

    # ToDo: The following line in comment makes the test fail,
    #  replace once the torch implementation changes
    # return grp_range(cnt)[torch.argsort(arr).argsort()]
    return grp_range(cnt)[
        np.argsort(np.argsort(arr, kind="mergesort"), kind="mergesort")
    ]


def frame_grid(
    img_size: Tuple[int, int],
    xextent: Optional[Tuple[float, float]] = None,
    yextent: Optional[Tuple[float, float]] = None,
    # *,
    origin: Optional[Tuple[float, float]] = None,
    px_size: Optional[Tuple[float, float]] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get pixel center coordinates based on extent and img shape.
    Either specify extents XOR origin and px size.

    Args:
        img_size: image size in pixels
        xextent: extent in x
        yextent: extent in y
        origin: upper left corner (tuple of 2)
        px_size: size of one pixel
        device: device to perform computation on

    Returns:
        bin_x: x bins
        bin_y: y bins
        bin_ctr_x: bin centers in x
        bin_ctr_y: bin centers in y

    """

    if ((origin is not None) and (xextent is not None or yextent is not None)) or (
        (origin is None) and (xextent is None or yextent is None)
    ):
        raise ValueError("You must XOR specify extent or origin and pixel size.")

    if origin is not None:
        xextent = (origin[0], origin[0] + img_size[0] * px_size[0])
        yextent = (origin[1], origin[1] + img_size[1] * px_size[1])

    return frame_grid_kernel(img_size, xextent, yextent, device)


def extent_by_sizes(
    img_size: int | Tuple[int, int],
    px_size: float | Tuple[float, float] = (1, 1),
    origin: float | Tuple[float, float] = (-0.5, -0.5),
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Get extent by image size and pixel size. If no tuple are specified,
    the same value is used for both dimensions which
    applies to img_size, px_size and origin.

    Args:
        img_size: image size in pixels
        px_size: size of one pixel
        origin: upper left corner

    Returns:
        xextent:
        yextent:
    """
    img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
    px_size = (px_size, px_size) if isinstance(px_size, (int, float)) else px_size
    origin = (origin, origin) if isinstance(origin, (int, float)) else origin

    xextent = (origin[0], origin[0] + img_size[0] * px_size[0])
    yextent = (origin[1], origin[1] + img_size[1] * px_size[1])

    return xextent, yextent


@torch.jit.script
def frame_grid_kernel(
    img_size: Tuple[int, int],
    xextent: Tuple[float, float],
    yextent: Tuple[float, float],
    device: torch.device = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # see frame_grid for docs
    bin_x = torch.linspace(*xextent, steps=img_size[0] + 1, device=device)
    bin_y = torch.linspace(*yextent, steps=img_size[1] + 1, device=device)
    bin_ctr_x = (bin_x + (bin_x[1] - bin_x[0]) / 2)[:-1]
    bin_ctr_y = (bin_y + (bin_y[1] - bin_y[0]) / 2)[:-1]

    return bin_x, bin_y, bin_ctr_x, bin_ctr_y


class CompositeAttributeModifier:
    def __init__(self, mod_fn: dict[str, Callable]):
        """
        Modify attributes by independent callables.
        The order of the dictionary is the order in which the attributes are changed.

        Examples:
            `mod = CompositeAttributeModifier(
                {"xyz": lambda x: x/2, "phot": lambda p: p * 2}
            )`
            would divide the xyz by 2 and doubles the phot attribute.

        Args:
            mod_fn: dictionary of callables with key being the attribute to modify
        """
        self._mod_fn = mod_fn

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: Any) -> Any:
        x = copy.copy(x)
        for attr, mod_fn in self._mod_fn.items():
            v = mod_fn(getattr(x, attr))
            setattr(x, attr, v)
        return x
