# here we collect protocols that are used in multiple places
from abc import abstractmethod
from typing import Any, Generic, Protocol, TypeVar

import torch


class Forwardable(Protocol):
    def forward(self, *args, **kwargs) -> Any:
        ...


class Sampleable(Protocol):
    def sample(self) -> Any:
        ...


class SampleableTensor(Sampleable):
    def sample(self) -> torch.Tensor:
        ...


class MultiDevice(Protocol):
    @property
    def device(self) -> torch.device:
        ...

    @abstractmethod
    def to(self, device: torch.device) -> "MultiDevice":
        ...


T = TypeVar("T")


class TypedSequence(Protocol, Generic[T]):
    def __len__(self) -> int:
        ...

    def __getitem__(self, item: int) -> T:
        ...
