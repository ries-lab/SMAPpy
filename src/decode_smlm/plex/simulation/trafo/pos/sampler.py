from abc import abstractmethod
from typing import Protocol, Optional, Union, Sequence, List, Callable

import torch
from structlog import get_logger

from . import trafo
from ....utils import future

logger = get_logger(__name__)


class XYZTransformationSamplerProtocol(Protocol):
    @abstractmethod
    def sample(
        self, n: Optional[int] = None
    ) -> Union[trafo.XYZTransformation, Sequence[trafo.XYZTransformation]]:
        """
        Samples, a transformation or multiple transformations.
        """
        raise NotImplementedError


class TransformationSamplerNoOp(XYZTransformationSamplerProtocol):
    def __init__(self, t: trafo.XYZTransformation, n: int | None):
        self._trafo = t
        self._n = n

    def sample(self, n: int | None = None):
        n = n if n is not None else self._n
        return [self._trafo for _ in range(n)]


class TransformationOffsetSampler(XYZTransformationSamplerProtocol):
    def __init__(
        self,
        trafo: trafo.XYZTransformationMatrix,
        offset: tuple[float, ...] | Callable[[], torch.Tensor],
        rebound: bool = True,
        n: int = 1,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Samples a random offset and adds it to the xyz to fake a different
        transformation, i.e. a transformation that is similar to a random crop.

        Args:
            trafo: xyz transformation
            offset: offset limits or callable that returns an offset
            rebound: subtract offset after transformation
            rebound_t: subtract transformed offset after transformation
            n: default sample size
            device:
        """
        self._trafo = trafo
        offset = (
            torch.as_tensor(offset, device=device) if not callable(offset) else offset
        )
        offset_fn = offset if callable(offset) else self._offset_factory
        self._offset = offset
        self._offset_fn = offset_fn
        self._rebound = rebound
        self._n = n
        self._device = device

        if self._trafo.device != self._device:
            logger.info(
                f"Trafo device {self._trafo.device} does not match "
                f"sampler device {self._device}. Shipping trafo to sampler "
                f"device."
            )
            self._trafo = self._trafo.to(self._device)

    def sample(self, n: Optional[int] = None) -> List[trafo.XYZCompositTransformation]:
        n = n if n is not None else self._n
        t = [self._trafo_factory() for _ in range(n)]
        return t

    def _trafo_factory(self) -> trafo.XYZCompositTransformation:
        offset = self._offset_fn()
        return trafo.offset_trafo(self._trafo, offset, rebound=self._rebound)

    def _offset_factory(self) -> torch.Tensor:
        return torch.rand_like(self._offset) * self._offset


class TransformationRotSampler(XYZTransformationSamplerProtocol):
    def __init__(
        self,
        angle: Optional[Union[Sequence[float], Callable[[int], torch.Tensor]]],
        offset: Optional[Union[Sequence[float], Callable[[int], torch.Tensor]]],
        rebound: bool = True,
        n: int = 1,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Samples a random rotation and adds it to the xyz to fake a different
        transformation, i.e. a transformation that is similar to a random crop.

        Args:
            angle: rotation angle limits
            offset: offset limits
            rebound: subtract offset after transformation
            rebound_t: subtract transformed offset after transformation
            n: default sample size
            device:
        """
        self._angle = angle
        self._offset = offset
        self._rebound = rebound
        self._n = n
        self._device = device

    def sample(self, n: Optional[int] = None) -> List[trafo.XYZRotation]:
        n = n if n is not None else self._n

        angle = [self._angle] * n if not callable(self._angle) else self._angle(n)
        offset = [self._offset] * n if not callable(self._offset) else self._offset(n)

        t = [
            trafo.XYZRotation(
                a,
                o,
                rebound=self._rebound,
                device=self._device,
            )
            for a, o in future.zip(angle, offset, strict=True)
        ]
        return t
