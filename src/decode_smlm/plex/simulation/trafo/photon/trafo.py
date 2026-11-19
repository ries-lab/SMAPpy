import copy
from abc import abstractmethod
from typing import Literal
from typing import Protocol, Optional, Union

import torch

from ....generic import protocols
from ....utils import torch as torch_utils


class ChoricTransformation(protocols.MultiDevice, Protocol):
    @abstractmethod
    def forward(self, phot: torch.Tensor, color: torch.Tensor):
        raise NotImplementedError


class MultiChoricSplitter(ChoricTransformation):
    def __init__(
        self,
        t: torch.Tensor,
        t_sig: Optional[torch.Tensor] = None,
        ix_low: Optional[int] = 0,
        channel: Literal["first", "last"] = "last",
        _optional_expand: bool = True,
    ):
        """
        Resembles a multi-choric beam splitter by a transmission matrix
        (which can be sampled).

        Args:
            t: transmission matrix of size `colour x channel`
            t_sig: standard deviation of transmission matrix of size `colour x channel`
            ix_low: lower index of the channel range
            _optional_expand: if True, the input tensor is expanded to the number of
                channels but its not forced if the photon tensor already is 2D
        """
        self._t = t
        self._t_mu = copy.copy(t)
        self._t_sig = t_sig
        self._ix_low = ix_low
        self._channel = channel
        self._optional_expand = _optional_expand

    @property
    def device(self) -> torch.device:
        return self._t.device

    def to(self, device: Union[str, torch.device]) -> "MultiChoricSplitter":
        return MultiChoricSplitter(
            t=self._t.to(device),
            t_sig=self._t_sig.to(device) if self._t_sig is not None else None,
            ix_low=self._ix_low,
        )

    def forward(
        self, phot: torch.Tensor, color: Optional[torch.LongTensor]
    ) -> torch.Tensor:
        if color is not None:
            color = color - self._ix_low
            if (phot.dim() == 1) or (not self._optional_expand):
                phot = self._expand_col_by_index(phot, color, len(self._t))

        match self._channel:
            case "first":
                phot = torch_utils.permute_dim_to_pos(phot, 0, -1)

        # phot size here `N x colour`
        phot = phot @ self._t

        match self._channel:
            case "first":
                phot = torch_utils.permute_dim_to_pos(phot, -1, 0)

        return phot

    def sample_transmission_(self):
        # inplace
        self._t = self.sample_transmission()
        return self

    def sample_transmission(self):
        """
        Samples transmission matrix and renormalizes it
        """
        t = torch.normal(self._t_mu, self._t_sig)
        # normalize twice to account for possible effect of clamp, otherwise
        # one can get nan
        t /= torch.sum(t, dim=1, keepdim=True)
        t = t.clamp(min=0.0)
        t /= torch.sum(t, dim=1, keepdim=True)
        return t

    @staticmethod
    def _expand_col_by_index(x: torch.Tensor, ix: torch.LongTensor, ix_max: int):
        """
        Expands a one dim. tensor `x` to col dimension of size `ix_max` and puts the
        value at `ix`.

        Args:
            x: tensor to be expanded
            ix: col position
            ix_max: number of cols

        Examples:
            >>> _expand_col_by_index([1, 2], [1, 0], 3)
            [
                [0, 1, 0],
                [2, 0, 0]
            ]

        """
        x_out = x.unsqueeze(1).repeat(1, ix_max)

        # create bool with True where we should expand
        ix_bool = torch.zeros_like(x_out, dtype=torch.bool)
        ix_bool[torch.ones_like(x, dtype=torch.bool), ix] = True

        x_out *= ix_bool

        return x_out
