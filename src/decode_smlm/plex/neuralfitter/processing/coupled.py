from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from ...emitter import emitter
from ...simulation import roi


class Coupled(ABC):
    @abstractmethod
    def forward(
        self,
        em: emitter.EmitterSet,
        frame: torch.Tensor | Sequence[torch.Tensor],
        bg: torch.Tensor | Sequence[torch.Tensor],
        aux: [torch.Tensor, Sequence[torch.Tensor]],
    ):
        raise NotImplementedError


class CoupledCrop(Coupled):
    def __init__(
        self,
        roi: roi.ROI | roi.ROISampler | None = None,
    ):
        """
        Crops frames and emitters to a given ROI.

        Args:
            roi:
        """
        super().__init__()
        self._roi = roi

    def forward(
        self,
        em: emitter.EmitterSet | None,
        frame: torch.Tensor | Sequence[torch.Tensor] | None,
        bg: torch.Tensor | Sequence[torch.Tensor] | None,
        aux: torch.Tensor | Sequence[torch.Tensor] | None,
    ):
        if self._roi is None:
            return em, frame, bg, aux

        r = self._roi.sample() if isinstance(self._roi, roi.ROISampler) else self._roi

        if em is not None:
            em = r.crop_emitter(em)
        if frame is not None:
            frame = r.crop_frames(frame)
        if bg is not None:
            bg = r.crop_frames(bg)
        if aux is not None:
            aux = r.crop_frames(aux)

        return em, frame, bg, aux
