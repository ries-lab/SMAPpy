from abc import abstractmethod

import torch

from . import protocols


class ForwardCallAlias(protocols.Forwardable):
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class MultiDevice(protocols.MultiDevice):
    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @abstractmethod
    def to(self, device: torch.device) -> "MultiDevice":
        raise NotImplementedError
