import copy
from typing import TypeVar, Optional, Union, Sequence, Iterable

import torch
from structlog import get_logger

from . import process
from .data import dataset
from ..emitter import emitter
from ..generic import indexing, protocols
from ..generic import lazy
from ..generic.delay import _DelayedSlicer, _DelayedTensor
from ..simulation import trafo, sampler as em_sampler
from ..simulation.microscope import Microscope, MicroscopeMultiChannel
from ..simulation.trafo.pos import trafo
from ..utils import future
from .sample import samplers

logger = get_logger()

T = TypeVar("T", bound="_TypedSequence")


class SamplerMicroscope:
    def __init__(
        self,
        mic_common: Microscope,
        win: int,
    ):
        """
        A class to generate training samples that allow for sample-wise
        xyz transformations and a shared microscope.
        This is not straight-forward since the xyz transformations need to be applied
        individually and only then frames can be computed.

        Note:
            The shared microscope should not have xyz transformations.

        Args:
            mic_common: common microscope
            win: window of sample
        """
        self._mic_common = mic_common
        self._win = win
        self._n_pad = win // 2
        self._win_limit = win // 2

        if win % 2 != 1:
            raise ValueError("Window must be odd.")

        if (
            isinstance(mic_common, MicroscopeMultiChannel)
            and self._mic_common._trafo_xyz is not None
        ):
            logger.warning(
                "Common microscope and sampler have positional (xyz)"
                "transformations. Know what you are doing."
            )

    def forward(
        self,
        em: Union[emitter.EmitterSet, Sequence[emitter.EmitterSet]],
        bg: Optional[Iterable[torch.Tensor]],
        trafo_xyz: Optional[Sequence[trafo.XYZTransformation]],
        ix_low: Optional[int] = None,
        ix_high: Optional[int] = None,
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """
        Forward emitters through sampler and return frames.
        Output will be a sequence of frames per channel. Per channel the frames will
        have batch dimension equivalent to the length of the emitter sequence (limited
        by ix_low and ix_high) or the frame_range of the emitters if a EmitterSet is
        directly passed.

        Args:
            em: EmitterSet which will be windowed or Sequence of EmitterSet. In case of a
                sequence, all EmitterSets are expected to reference the same frame range,
                otherwise the mapping will be wrong. I.e. for window size 3, the
                frame_ix need to be between -1 and 1 (inlcuding).
            bg: sequence of bg frames per channel
            trafo_xyz: sample-wise xyz transformations
            ix_low: lower frame ix
            ix_high: upper frame ix

        """

        # here, frame_ix is meant, not index in EmitterSet (rather EmitterSet.frame_ix)
        ix_low = ix_low if ix_low is not None else 0
        ix_high = (
            ix_high
            if ix_high is not None
            else (len(em) if isinstance(em, list) else em.frame_ix.max().item() + 1)
        )

        em = self._window_emitters(em, ix_low, ix_high)

        if trafo_xyz is not None:
            for i, (e, t_xyz) in enumerate(future.zip(em, trafo_xyz, strict=True)):
                e.xyz = t_xyz.forward(e.xyz)
        else:
            logger.warning(
                f"Using `SamplerMicroscope` without xyz transformations is "
                f"possible but without point."
            )

        # background needs to be repeated to account for linearization
        if bg is not None:
            bg = [bg] if not isinstance(bg, Sequence) else bg
            bg = [b[ix_low:ix_high] for b in bg]
            bg = self._linearize_bg(bg)

        # linearize emitters, we do that because microscopes expect one EmitterSet
        # and batched psf computation is much faster
        em_lin = emitter.EmitterSet.cat(em)
        frames = self._mic_common.forward(
            em=em_lin, bg=bg, ix_low=0, ix_high=len(em) * self._win
        )
        frames = self._unstack(frames)
        return frames

    def _window_emitters(
        self,
        em: Union[emitter.EmitterSet, Sequence[emitter.EmitterSet]],
        ix_low: int,
        ix_high,
    ) -> Sequence[emitter.EmitterSet]:
        # take relevant subset on specified frames. Shift frame_ix to start its
        # reference at 0 because we ultimately want to compute the frames
        # note that this does not mean that we actually have emitters at frame_ix 0,
        # it just means that this is the reference

        if not isinstance(em, emitter.EmitterSet):
            em = em[ix_low:ix_high]
            em = copy.deepcopy(em)
            for i, e in enumerate(em):
                if len(e) > 0:
                    if (
                        e.frame_ix.min() < -self._win_limit
                        or e.frame_ix.max() > self._win_limit
                    ):
                        raise ValueError(
                            f"EmitterSet.frame_ix must be between -win_limit and "
                            f"win_limit (including)."
                        )
                e.frame_ix += self._win // 2 + i * self._win
            return em

        # window emitter by frames, and shift frame_ix to start its reference at 0
        em = em.get_subset_frame(ix_low, ix_high)
        em.frame_ix -= ix_low
        ix_high -= ix_low
        ix_low = 0

        # pad emitters at both ends
        em_first = em.iframe[ix_low].repeat(self._n_pad, 1)
        em_first.frame_ix -= self._n_pad
        em_last = em.iframe[ix_high - 1].repeat(self._n_pad, 1)
        em_last.frame_ix += 1  # append at end

        em = emitter.EmitterSet.cat([em_first, em, em_last])
        em.frame_ix += self._n_pad

        em_split = em.split_in_frames(ix_low, ix_high + self._n_pad * 2)

        ix_low += self._n_pad
        ix_high += self._n_pad
        # we omit n because we do it via range
        win = indexing.IxWindow(win=self._win, n=None)
        em = [
            emitter.EmitterSet.cat([em_split[ww] for ww in win[ix]], sanity_check=False)
            for ix in range(ix_low, ix_high)
        ]

        for i, e in enumerate(em):
            e.frame_ix += i * self._n_pad * 2

        return em

    def _linearize_bg(self, bg: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        # only applies to sample-wise bg, not to single bg for all
        if bg[0].ndim >= 3:
            bg = [b.repeat_interleave(self._win, dim=0) for b in bg]

        return bg

    def _unstack(
        self, frames: Union[torch.Tensor, Sequence[torch.Tensor]]
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        # could be list of tensors (channel-wise) or tensor with channels (in dim 1)
        if isinstance(frames, torch.Tensor):
            frames = frames.unfold(0, self._win, self._win).permute(0, -1, 1, 2, 3)

        if isinstance(frames, Sequence):
            frames = [
                f.unfold(0, self._win, self._win).permute(0, -1, 1, 2) for f in frames
            ]

        return frames


class SamplerIndependents:  # ToDo: Naming
    def __init__(
        self,
        em: [em_sampler.EmitterSampler],
        bg: [Union[protocols.SampleableTensor, Sequence[protocols.SampleableTensor]]],
        trafo: Optional[protocols.Sampleable] = None,
    ):
        """
        This samples the `independent` variables of a physical experiment.

        Args:
            em:
            bg:
            trafo:
        """
        self._em = em
        self._bg = bg
        self._trafo = trafo

    def sample(
        self,
    ) -> tuple[
        emitter.EmitterSet,
        Union[torch.Tensor, Sequence[torch.Tensor]],
        Optional[Sequence[trafo.XYZTransformation]],
    ]:
        em = self._em.sample()
        bg = (
            [bg.sample() for bg in self._bg]
            if isinstance(self._bg, Sequence)
            else self._bg.sample()
        )
        trafo = self._trafo.sample() if self._trafo is not None else None

        return em, bg, trafo


class SamplerTraining:
    def __init__(
        self,
        proc: process.Processing,
        sampler_physical: Optional[SamplerIndependents] = None,
        sampler_microscope: Optional[SamplerMicroscope] = None,
        window: Optional[int] = 1,
        ix_low: Optional[int] = None,
        ix_high: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        _device_storage: Union[str, torch.device] = "cpu",
    ):
        """
        The purpose of this is to attach processing to an experiment to generate input
        and target.

        Args:
            proc: processing that is able to produce input and target
            sampler_physical: physical sampler, i.e. returning emitters, bg, frames and
             trafo
            window: win size for input
        """
        super().__init__()

        self._proc = proc
        self._ix_low = ix_low
        self._ix_high = ix_high
        self._sampler = sampler_physical
        self._sampler_mic = sampler_microscope
        self._win = window
        self._device = device
        self._device_to = lazy.To(device)
        self._device_storage = _device_storage
        self._device_storage_to = lazy.To(_device_storage)

        self.em: Optional[emitter.EmitterSet] = None
        self.bg: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None
        self.frame: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None
        self.trafo: Optional[Sequence[trafo.XYZTransformation]] = None
        self.indicator: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None

    def __len__(self) -> int:
        if self.frame is None:
            raise ValueError("Unable to return length because frame was not set.")
        if not isinstance(self.frame, Sequence):
            return len(self.frame)
        else:
            if not all(len(f) == len(self.frame[0]) for f in self.frame):
                raise ValueError(
                    "All channels need to have the same frame length, but got "
                    f"{[len(f) for f in self.frame]}"
                )
        return len(self.frame[0])

    # ToDo: Get rid of this and seperate into other class
    def _sample_kwargs(self, frame, em, bg, trafo) -> tuple[dict, dict]:
        win = self._win

        raise_win = False
        if self._win is not None:
            if isinstance(frame, torch.Tensor) and frame.dim() == 4:
                raise_win = True if frame.size(1) != self._win else False
                win = None
            elif isinstance(frame, Sequence) and frame[0].dim() == 4:
                raise_win = True if frame[0].size(1) != self._win else False
                win = None

            if raise_win:
                raise ValueError(
                    f"Frames have window dimension already but "
                    f"frame_samples should still apply windowing (win: {self._win})."
                )
        if win != self._win:
            logger.info(
                "Windowing automatically changed because of frame dimensions.",
                win_new=win,
                win_old=self._win,
            )

        kwargs = {
            "frame": samplers.frame_samples(frame, window=win),
            "em": em.iframe,
            "bg": samplers.bg_samples(bg),
        }
        kwargs_static = dict()
        if trafo is not None:
            kwargs["aux"] = self.trafo
        else:
            kwargs_static["aux"] = None

        return kwargs, kwargs_static

    def train_samples(self) -> _DelayedSlicer:
        kwargs, kwargs_static = self._sample_kwargs(
            frame=self.frame, em=self.em, bg=self.bg, trafo=self.trafo
        )

        return _DelayedSlicer(
            self._proc.pre_train,
            kwargs=kwargs,
            kwargs_static=kwargs_static,
            indexer=None,
            # dataset.IxShifter(None, window=self._win, n=len(self))
            # if self._win
            # else None,
        )

    def val_samples(self) -> _DelayedSlicer:
        kwargs, kwargs_static = self._sample_kwargs(
            frame=self.frame, em=self.em, bg=self.bg, trafo=self.trafo
        )

        return _DelayedSlicer(
            self._proc.pre_val,
            kwargs=kwargs,
            kwargs_static=kwargs_static,
            indexer=None,
            # dataset.IxShifter(None, window=self._win, n=len(self))
            # if self._win
            # else None,
        )

    def inference_samples(self, frame, trafo: Optional[int] = None) -> _DelayedSlicer:
        return samplers.inference_samples(
            samples=samplers.frame_samples(frame),
            aux=trafo,
            pre_fn=self._proc.pre_inference,
        )

    def register(
        self,
        em: emitter.EmitterSet,
        bg: torch.Tensor,
        frame: torch.Tensor,
        trafo: Optional[Sequence[trafo.XYZTransformation]] = None,
    ) -> None:  # ToDo: This should be obsolete
        """
        Register the data that is used for generating the samples.

        Args:
            em:
            bg:
            frame:
            trafo:

        """
        self.em = self._device_storage_to(em)
        self.bg = self._device_storage_to(bg)
        self.frame = self._device_storage_to(frame)
        self.trafo = self._device_storage_to(trafo)

    def sample(self, em=None, bg=None, trafo=None) -> None:
        # ToDo: make em, bg, trafo arguments to function and remove sampler attributes

        if self._ix_low is None or self._ix_high is None:
            raise ValueError(f"ix_low and ix_high need to be set for sampling.")

        em, bg, trafo = self._sampler.sample()

        if trafo is not None:
            frame = self._sampler_mic.forward(
                em=em.to(self._device),
                bg=bg,
                trafo_xyz=trafo,
                ix_low=self._ix_low,
                ix_high=self._ix_high,
            )
        else:
            frame = self._sampler_mic.forward(
                em=em.to(self._device),
                bg=bg,
                ix_low=self._ix_low,
                ix_high=self._ix_high,
            )

        self.register(em, bg, frame, trafo)
