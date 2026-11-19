from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Literal

import numpy as np
import scipy.spatial
import torch
from deprecated import deprecated

from .emitter import EmitterSet
from ..generic.processing import FilterXYZ


class EmitterProcess(ABC):
    def arg_forward(self, em: EmitterSet) -> torch.Tensor:
        raise NotImplementedError("EmitterProcess.arg_forward not implemented")

    @abstractmethod
    def forward(self, em: EmitterSet) -> EmitterSet:
        """
        Forwards a set of emitters through the filter implementation

        Args:
            em: emitters

        """
        return em


class EmitterProcessNoOp(EmitterProcess):
    def arg_forward(self, em: EmitterSet) -> torch.Tensor:
        return torch.ones(len(em), dtype=torch.bool)

    def forward(self, em: EmitterSet) -> EmitterSet:
        return em


class ArgForwardableMixin:
    def arg_forward(self, em: EmitterSet) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, em: EmitterSet) -> EmitterSet:
        return em[self.arg_forward(em)]


class EmitterFilterGeneric(ArgForwardableMixin, EmitterProcess):
    def __init__(
        self,
        agg: Literal["any", "all"] = "all",
        pre_agg: Literal["sum"] = None,
        **kwargs,
    ):
        """
        Generic emitter filter.

        Args:
            agg: default aggregate function to use for multi-dim attributes
            **kwargs: use emitter attribute and tuples to specify range,
             or specify callable that returns boolean

        Examples:
            # filters out emitters where color ix < 1 and photon count is <= 100.
             >>> f = EmitterFilterGeneric(color=[1, None], phot=lambda p: p > 100.)
        """
        super().__init__()

        # construct filter if tuples are specified
        for k, v in kwargs.items():
            if not callable(v):
                kwargs[k] = _RangeFilter(*v, agg=agg, pre_agg=pre_agg)

        self._attr_fn = kwargs

    def arg_forward(self, em: EmitterSet) -> torch.Tensor:
        is_okay = torch.ones(len(em), dtype=torch.bool)

        for k, fn in self._attr_fn.items():
            is_okay *= fn(getattr(em, k))

        return is_okay


class EmitterFilterFoV(ArgForwardableMixin, EmitterProcess):
    def __init__(
        self,
        xextent: Union[tuple, torch.Tensor],
        yextent: Union[tuple, torch.Tensor],
        zextent: Union[tuple, torch.Tensor] | None = None,
        xy_unit: str = "px",
        mode: Literal["any", "all"] | None = None,
    ):
        """
        Removes emitters that are outside a specified extent.
        The lower / left respective extent limits are included,
        the right / upper extent limit is excluded / open.

        Args:
            xextent: extent of allowed field in x direction
            yextent: extent of allowed field in y direction
            zextent: (optional) extent of allowed field in z direction
            xy_unit: str that specifies the unit of the data
        """
        super().__init__()
        self._kernel = FilterXYZ(xextent, yextent, zextent, mode)
        self.xy_unit = xy_unit

    def arg_forward(self, em: EmitterSet) -> torch.Tensor:
        """Removes emitters that are outside of the specified extent."""

        if self.xy_unit is None:
            em_mat = em.xyz
        elif self.xy_unit == "px":
            em_mat = em.xyz_px
        elif self.xy_unit == "nm":
            em_mat = em.xyz_nm
        else:
            raise ValueError(f"Unsupported xy unit: {self.xy_unit}")

        is_emit = self._kernel.filter(em_mat)

        return is_emit


class EmitterFilterConvexHull(ArgForwardableMixin, EmitterProcess):
    def __init__(self, points: torch.Tensor, xy_unit: str = "px", tol: float = 1e-6):
        """
        Filters emitters by convex hull.

        Args:
            points: points to construct the convex hull
            xy_unit: unit of the data
        """
        super().__init__()
        self._hull_points = points
        self._hull = scipy.spatial.ConvexHull(points.numpy())
        self._hull_equations = torch.from_numpy(self._hull.equations)
        self._tol = tol
        self.xy_unit = xy_unit

    def _points_in_hull(self, p: np.ndarray) -> np.ndarray:
        # taken from https://stackoverflow.com/a/72483841

        return np.all(
            self._hull.equations[:, :-1] @ p.T
            + np.repeat(self._hull.equations[:, -1][None, :], len(p), axis=0).T
            <= self._tol,
            0,
        )

    def arg_forward(self, em: EmitterSet) -> torch.Tensor:
        match self.xy_unit:
            case "px":
                xyz = em.xyz_px
            case "nm":
                xyz = em.xyz_nm
            case _:
                raise ValueError(f"Unsupported xy unit: {self.xy_unit}")

        if self._hull_points.size(1) == 2:
            xyz = xyz[:, :2]

        in_hull = self._points_in_hull(xyz.numpy())
        return torch.from_numpy(in_hull)


class EmitterFilterFrame(EmitterProcess):
    def __init__(self, ix_low: int, ix_high: int, shift: int):
        """
        Filter emitters by frame. Thin wrapper around `em.get_subset_frame`.

        Args:
            ix_low: lower frame ix
            ix_high: upper frame ix
            shift: shift frames by
        """
        super().__init__()
        self._ix_low = ix_low
        self._ix_high = ix_high
        self._shift = shift

    def forward(self, em: EmitterSet) -> EmitterSet:
        return em.get_subset_frame(self._ix_low, self._ix_high, self._shift)


class _RangeFilter:
    def __init__(
        self,
        low: Optional[Any],
        high: Optional[Any],
        inverse: bool = False,
        agg: Literal["any", "all"] | None = None,
        pre_agg: Literal["sum"] | None = None,
    ):
        """
        For use in conjunction with EmitterFilterGeneric.

        Args:
            low: inclusive
            high: exclusive
            inverse: invert result
            agg: aggregation of boolean outcome for multi-dim tensors
            pre_agg: aggregation before checking range
        """
        self._low = torch.as_tensor(low) if low is not None else None
        self._high = torch.as_tensor(high) if high is not None else None
        self._inverse = inverse
        self._agg = agg
        self._pre_agg = pre_agg

    def __call__(self, em_attr) -> torch.Tensor:
        if self._pre_agg is not None:
            em_attr = getattr(em_attr, self._pre_agg)(-1)

        is_ok = torch.ones_like(em_attr, dtype=torch.bool)

        if self._low is not None:
            is_ok *= em_attr >= self._low

        if self._high is not None:
            is_ok *= em_attr < self._high

        if self._inverse:
            is_ok = ~is_ok

        if em_attr.dim() >= 2 and self._agg is not None:
            is_ok = getattr(is_ok, self._agg)(-1)  # aggregate away the last dim

        return is_ok


def range_factory(mode: Literal["less", "greater"]):
    # helper function to create range filter
    def _range_filter(threshold: float) -> _RangeFilter:
        if mode == "less":
            return _RangeFilter(None, threshold, inverse=False)
        elif mode == "greater":
            return _RangeFilter(threshold, None, inverse=False)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    return _range_filter


@deprecated(reason="Use generic filter.", version="0.11.0")
class TarFrameEmitterFilter(EmitterProcess):
    pass


@deprecated(reason="Use generic filter.", version="0.11.0")
class PhotonFilter(EmitterProcess):
    pass
