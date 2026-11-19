from typing import Any, Optional, Iterable, Callable, Union, Sequence, Protocol

import torch
from deprecated import deprecated
from structlog import get_logger

from . import noise as noise_lib
from . import psf_kernel
from . import trafo
from ..emitter.emitter import EmitterSet
from ..generic import utils

logger = get_logger(__name__)


class MicroscopeProtocol(Protocol):
    # weak protocol for microscope to allow for non-child usage
    def forward(
        self,
        em: EmitterSet,
        bg: Optional[torch.Tensor] = None,
        ix_low: Optional[int] = None,
        ix_high: Optional[int] = None,
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        ...


class Microscope(MicroscopeProtocol):
    def __init__(
        self,
        psf: psf_kernel.PSF,
        noise: Optional[noise_lib.NoiseDistribution] = None,
        frame_range: Optional[Union[int, tuple[int, int]]] = None,
    ):
        """
        Microscope consisting of psf and noise model.

        Args:
            psf: point spread function
            noise: noise model
            frame_range: frame range in which to sample
        """
        self._psf = psf
        self._noise = noise
        # default to 0 ... frame_range if int
        self._frame_range = (
            frame_range if not isinstance(frame_range, int) else (0, frame_range)
        )

    def forward(
        self,
        em: EmitterSet,
        bg: Optional[torch.Tensor] = None,
        ix_low: Optional[int] = None,
        ix_high: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward emitter and background through microscope and return frames.

        Args:
            em: emitters
            bg: background
            ix_low: lower frame index
            ix_high: upper frame index

        """
        ix_low = ix_low if ix_low is not None else self._frame_range[0]
        ix_high = ix_high if ix_high is not None else self._frame_range[1]

        f = self._psf.forward(
            em.xyz_px, em.phot, em.frame_ix, ix_low=ix_low, ix_high=ix_high
        ).to(em.device)

        if bg is not None:
            f += bg
        if self._noise is not None:
            f = self._noise.forward(f)

        return f


class MicroscopeMultiChannel:
    def __init__(
        self,
        psf: list[psf_kernel.PSF],
        noise: list[Optional[noise_lib.NoiseDistribution]],
        frame_range: Optional[tuple[int, int]],
        ch_range: Optional[Union[int, tuple[int, int]]],
        trafo_xyz: Optional[trafo.pos.trafo.XYZTransformation] = None,
        trafo_phot: Optional[trafo.photon.trafo.ChoricTransformation] = None,
        stack: Optional[Union[str, Callable]] = None,
    ):
        """
        A microscope that has multi channels. Internally this is modelled as a list
        of individual microscopes.

        Args:
            psf: list of psf
            noise: list of noise
            frame_range: frame range among which frames are sampled
            ch_range: range of active channels
            trafo_xyz: channel-wise coordinate transformer
            trafo_phot: channel-wise photon transformer
            stack: stack function, None, `stack` or callable.
        """
        self._microscopes: list[Microscope] = [
            Microscope(psf=p, noise=n, frame_range=frame_range)
            for p, n in zip(psf, noise, strict=True)
        ]
        self._ch_range = ch_range
        self._trafo_xyz = trafo_xyz
        self._trafo_phot = trafo_phot
        self._stack_impl = stack

    def _stack(self, x: Sequence[torch.Tensor]) -> Any:
        if self._stack_impl is None:
            return x
        if self._stack_impl == "stack":
            return torch.stack(x, dim=1)
        raise ValueError("Unsupported stack implementation.")

    def forward(
        self,
        em: EmitterSet,
        bg: Optional[Iterable[torch.Tensor]] = None,
        ix_low: Optional[int] = None,
        ix_high: Optional[int] = None,
    ) -> Any:
        """
        Forward emitters through multichannel microscope

        Args:
            em: emitters
            bg: list of bg, length equals to number of channels
            ix_low: lower frame index
            ix_high: upper frame index

        Returns:
            frames
        """
        em = em.clone()

        if self._trafo_xyz is not None:
            em.xyz_px = self._trafo_xyz.forward(em.xyz_px)

        if self._trafo_phot is not None:
            em.phot = self._trafo_phot.forward(em.phot, em.code)

        if self._trafo_xyz is not None or self._trafo_phot is not None:
            em.code = em.infer_code() + self._ch_range[0]
            em = em.linearize()

        if em.phot.dim() == 2:
            logger.warning("Overwriting/Inferring code from phot instead of em.code.")
            em.code = em.infer_code() + self._ch_range[0]
            em = em.linearize()

        em_by_channel = [em.icode[c] for c in range(*self._ch_range)]
        bg = [None] * len(em_by_channel) if bg is None else bg

        frames = [
            m.forward(e, bg=b, ix_low=ix_low, ix_high=ix_high)
            for m, e, b in zip(self._microscopes, em_by_channel, bg, strict=True)
        ]
        return self._stack(frames)


@deprecated(reason="Not necessary", version="0.11.1dev1")
class MicroscopeChannelSplitter:
    def __init__(self):
        raise NotImplementedError


@deprecated(reason="Not necessary", version="0.11.1dev1")
class MicroscopeChannelModifier:
    def __init__(self, ch_fn: list[Callable]):
        """
        Used to apply a transformation per channel on an EmitterSet.

        Warnings:
            - this treats channels as independent


        Args:
            ch_fn: list of callables taking and outputting an EmitterSet.
        """
        raise NotImplementedError


class EmitterCompositeAttributeModifier(utils.CompositeAttributeModifier):
    # lazy alias for modifying emitter attributes
    pass
