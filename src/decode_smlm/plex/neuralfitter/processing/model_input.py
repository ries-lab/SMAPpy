from abc import abstractmethod
from typing import Any, Callable, Union, Sequence, Optional, Protocol, TypeVar

import torch

from ...generic import protocols
from ... import emitter
from .. import scaling
from ...simulation import camera, background
from ...utils import future


class ModelInput(Protocol):
    @abstractmethod
    def forward(
        self,
        frame: Union[torch.Tensor, Sequence[torch.Tensor]],
        em: emitter.EmitterSet,
        aux: dict[str, Any],
    ) -> torch.Tensor:
        raise NotImplementedError


T = TypeVar("T", bound=ModelInput)


class ModelInputPostponed(ModelInput):
    def __init__(
        self,
        *,
        frame_pre: protocols.Forwardable
        | Sequence[protocols.Forwardable]
        | None = None,
        cam: Optional[Union[camera.Camera, Sequence[camera.Camera]]] = None,
        aux: Optional[
            Callable[[T], Union[torch.Tensor, Sequence[torch.Tensor]]]
        ] = None,
        merger_bg: Optional[background.Merger] = None,
        scaler: scaling.base.ScalerModelChannel | None = None
    ):
        """
        Prepares model's input. This module is often used for lazy computation
        of the model input to put parts of the computation in the dataloader.
        It supports (all optional):
        - Combining frames with background
        - Applying camera noise
        - Auxiliary data generation (optional) or merging pre-computed auxiliary data
        - Scaling of frame and auxiliary data

        Note:
            - in inference (with static auxiliaries),
              constructor XOR forward aux should not be None

        Args:
            frame_pre: pre-processing of frame, e.g. cropping, or scaling etc.
            cam: camera module, omit if frames are already pre-computed
            aux: auxiliary generator or static auxiliary data
            merger_bg: optional background merger
            scaler: optional scaler applied before model input
        """

        # make it a list because this allows for zipping later on
        self._frame_pre = (
            [frame_pre]
            if frame_pre is not None and not isinstance(frame_pre, Sequence)
            else frame_pre
        )
        self._noise: Optional[list[cam.Camera]] = (
            [cam] if cam is not None and not isinstance(cam, Sequence) else cam
        )
        self._aux = aux
        self._merger_bg = background.Merger() if merger_bg is None else merger_bg
        self._scaler = scaler

    def forward(
        self,
        frame: Union[torch.Tensor, Sequence[torch.Tensor]],
        em: emitter.EmitterSet | None,
        bg: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
        aux: Optional[Union[torch.Tensor, Sequence[torch.Tensor], T]] = None,
    ) -> torch.Tensor:
        """
        Combines bg with frame, applies camera noise, combines with auxiliaries,
        concatenates them and scales them.

        Args:
            frame: of size T x H x W (T being the temporal window)
            em: not used
            bg: background tensor or sequence of background tensors
            aux: tensor of auxiliary model input or sequence of tensors, overwrites
             instance attribute aux if provided
        """
        if self._frame_pre is not None:
            for p in self._frame_pre:
                frame = p.forward(frame)

        if bg is not None:
            frame = self._merger_bg.forward(frame=frame, bg=bg)

        if self._noise is not None:
            frame = [
                n.forward(f)
                for n, f in future.zip(  # raises err for unequal
                    self._noise,
                    frame if isinstance(frame, Sequence) else (frame,),
                    strict=True,
                )
            ]

        if aux is not None:
            aux = self._aux(aux) if self._aux is not None else aux
        else:
            aux = self._aux

        frame = torch.cat(frame, -3) if isinstance(frame, Sequence) else frame
        if isinstance(aux, Sequence):
            aux = torch.cat(aux, -3) if aux[0].dim() == 3 else torch.stack(aux, 0)

        # list of channels and auxiliary to model input tensor
        x = torch.cat([frame, aux]) if aux is not None else frame

        if self._scaler is not None:
            x_scaled = self._scaler.forward(x)
            x = x_scaled.squeeze(0) if x.dim() < x_scaled.dim() else x_scaled

        return x
