from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Union

import torch


class TensorMemoryMapped(ABC, Sequence):
    """
    Minimal pseudo torch tensor, must be a sequence, implements `size` and
    returns proper torch.Tensor on __getitem__. Pseudo implements `device` and `to`.
    """

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def cpu(self):
        return self.to(torch.device("cpu"))

    def cuda(self):
        return self.to(torch.device("cuda"))

    def to(self, device: str | torch.device) -> "TensorMemoryMapped":
        # no op for better compatibility
        if device != torch.device("cpu"):
            raise NotImplementedError
        return self

    @abstractmethod
    def size(self, dim: Optional[int]) -> Union[int, torch.Size]:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item) -> torch.Tensor:
        pass
