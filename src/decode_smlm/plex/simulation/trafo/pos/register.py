from typing import Callable

from . import trafo
from ....neuralfitter import indicator

import torch


class Register:
    def __init__(
        self,
        t: trafo.XYZTransformation,
        roi_x: tuple[int, int],
        roi_y: tuple[int, int],
        aggregate_fn: Callable | str = "median",
    ):
        """
        Register channels by calculating a global (interger) shift, such that the
        resulting shifts in a defined region of interest (roi) vary around 0.

        Args:
            t: transformation that outputs channel-wise coordinates
            roi_x: roi in reference channel in x
            roi_y: roi in reference channel in y
        """
        self._t = t
        self._agg_fn = (
            aggregate_fn if callable(aggregate_fn) else getattr(torch, aggregate_fn)
        )
        self._roi_x = roi_x
        self._roi_y = roi_y

        x = (roi_x[0] - 0.5, roi_x[1] - 0.5)
        y = (roi_y[0] - 0.5, roi_y[1] - 0.5)
        shape = (roi_x[1] - roi_x[0], roi_y[1] - roi_y[0])

        self._ind = indicator.IndicatorChannelOffset(
            xextent=(x,),
            yextent=(y,),
            img_shape=(shape,),
            xy_trafo=t,
            device="cpu",
        )

    def forward(self):
        shifts = self._ind.forward()

        x = torch.stack([-s[0].mean() for s in shifts])
        y = torch.stack([-s[1].mean() for s in shifts])

        return x, y
