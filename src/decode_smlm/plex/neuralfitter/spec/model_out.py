from abc import ABC, abstractmethod
from typing import runtime_checkable, Protocol, Optional, Sequence

import torch


class ModelChannelMap(ABC):
    """
    Helper to map model output to semantic channels.

    Note: we need to return list, not tuples, otherwise indexing does not work. E.g.:
    >>> x = torch.rand(5)
    >>> x[[0, 1, 2]]  # ok
    >>> x[(0, 1, 2)]  # fails
    """

    @property
    @abstractmethod
    def n(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def ix_prob(self) -> list[int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def ix_phot(self) -> list[int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def ix_xyz(self) -> list[int]:
        raise NotImplementedError

    @property
    def ix_bg(self) -> list[int]:
        raise NotImplementedError

    def split_tensor(self, t: torch.Tensor) -> dict:
        """
        Split (model output) tensor into semantic channels

        Args:
            t: tensor of size `N x C (x H x W)`

        Returns:
            dictionary with channel key and tensor
        """
        raise NotImplementedError


@runtime_checkable
class _ModelChannelMapInferredAttr(Protocol):
    # just for linting
    ix_phot_sig: list[int]
    ix_xyz_sig: list[int]


class ModelChannelMapGMM(_ModelChannelMapInferredAttr, ModelChannelMap):
    _n_xyz = 3  # x, y, z

    def __init__(
        self,
        n_codes: int,
        n_channels: int,
        n_phot: int,
        ch_map_out: Optional[Sequence[int]] = None,
    ):
        """
        Helper to map model output of gaussian mixture model to semantic channels.

        Args:
            n_codes: number of codes to predict
            n_channels: number of channels
            n_phot: number of photon channels
            ch_map_out: maps output channels of the model, if None will be inferred
        """

        self._n_codes = n_codes
        self._n_ch = n_channels
        self._n_phot = n_phot
        self._ix = list(range(self.n))

        if ch_map_out is None:
            ch_map_out = self.auto_ch_out_map()
        self.ch_map_out = ch_map_out

    @property
    def n(self) -> int:
        return self.n_prob + self.n_mu + self.n_sig + self.n_bg

    @property
    def n_prob(self) -> int:
        return self._n_codes

    @property
    def n_codes(self) -> int:
        return self._n_codes

    @property
    def n_phot(self) -> int:
        return self._n_phot

    @property
    def n_mu(self) -> int:
        return 3 + self._n_phot

    @property
    def n_sig(self) -> int:
        return 3 + self._n_phot

    @property
    def n_bg(self) -> int:
        return self._n_ch

    @property
    def ix_prob(self) -> list[int]:
        return self._ix[: self.n_prob]

    @property
    def ix_mu(self) -> list[int]:
        # phot and xyz means
        return self._ix[self.n_prob : (self.n_prob + self.n_mu)]

    @property
    def ix_sig(self) -> list[int]:
        # phot and xyz sigmas
        return self._ix[
            (self.n_prob + self.n_mu) : (self.n_prob + self.n_mu + self.n_sig)
        ]

    @property
    def ix_bg(self) -> list[int]:
        return self._ix[-self.n_bg :]

    @property
    def ix_phot(self) -> list[int]:
        return self.ix_mu[: self._n_phot]

    @property
    def ix_xyz(self) -> list[int]:
        return self.ix_mu[self._n_phot :]

    @property
    def ix_z(self) -> list[int]:
        return self.ix_xyz[-1:]

    @property
    def _delta_mu_sig(self) -> int:
        return self._n_xyz + self._n_phot

    def split_tensor(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Split (model output) tensor into semantic channels

        Args:
            x: tensor of size `N x C (x H x W)`

        Returns:
            dictionary with channel key and tensor
        """
        if x.dim() < 2:
            raise ValueError(f"Expected tensor of dim >= 2, got size {x.size()}")

        return {
            "prob": x[:, self.ix_prob],
            "phot": x[:, self.ix_phot],
            "phot_sig": x[:, self.ix_phot_sig],
            "xyz": x[:, self.ix_xyz],
            "xyz_sig": x[:, self.ix_xyz_sig],
            "bg": x[:, self.ix_bg],
        }

    def __getattr__(self, item):
        # _sig attributes are inferred by delta between mu and sigma
        if "_sig" in item and hasattr(self, item.rstrip("_sig")):
            return [
                mu + self._delta_mu_sig for mu in getattr(self, item.rstrip("_sig"))
            ]

        raise AttributeError

    def auto_ch_out_map(self) -> tuple[int, ...]:
        return self.n_prob, self.n_mu, self.n_sig, self.n_bg
