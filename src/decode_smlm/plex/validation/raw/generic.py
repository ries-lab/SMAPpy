from abc import abstractmethod
from typing import Callable

import torch

from .. import base


class TensorValidator(base.Validator):
    @abstractmethod
    def validate(self, x: torch.Tensor):
        pass


class ChanneledValidator(TensorValidator):
    def __init__(self, ch_val: dict[int, TensorValidator]):
        self.ch_val = ch_val

    def validate(self, x: torch.Tensor):
        for ch, val in self.ch_val.items():
            val.validate(x[..., ch, :, :])


class ProbabilityMassValidator(TensorValidator):
    def __init__(
        self,
        action: Callable,
        low: float,
        high: float | None,
        low_th: float = 0.05,
        high_th: float = 0.9,
    ):
        """
        Validator for typical probability outputs. Assumes that a (large) fraction of
        the probability output is usually below `low_th` and of the values that are
        above, most should be above `high_th`.

        Args:
            action:
            low: min. fraction of values below low_th
            high: min. fraction of values above high_th of all that are above low_th
            low_th: lower threshold
            high_th: higher threshold
        """
        self.action = action
        self.low = low
        self.low_th = low_th
        self.high = high
        self.high_th = high_th

    def validate(self, x: torch.Tensor):
        ix_low = x < self.low_th
        fract_low = ix_low.float().mean()
        fract_high = (x[~ix_low] > self.high_th).float().mean()

        if (fract_low < self.low) or (
            (fract_high < self.high) if self.high is not None else False
        ):
            self.action({"fract_low": fract_low, "fract_high": fract_high})


class LimitValidator(TensorValidator):
    def __init__(
        self,
        mass: dict[tuple[float, float], float],
        action: Callable,
    ):
        """
        Run action if the fraction of values in the given range is above specified
         mass limit.

        Args:
            mass: tuple of limits and mass or None for no lower/upper limit
            action: action to run
        """
        if any(l[0] >= l[1] for l in mass):
            raise ValueError("Limits must be increasing.")
        self.mass = mass
        self.action = action

    def validate(self, x: torch.Tensor):
        for limits, mass in self.mass.items():
            ix = (x > limits[0]) & (x < limits[1])
            if (m := ix.float().mean()) > mass:
                self.action({"limits": limits, "mass": m.item()})
