from abc import ABC
from typing import Union, Literal, Optional, Sequence, TypeVar

import torch

from .. import emitter
from ..generic import utils
from ..utils import dev
from ..utils import torch as torch_utils

_T_Frame = TypeVar("_T_Frame", torch.Tensor, Sequence[torch.Tensor])
_T_Emitter = TypeVar("_T_Emitter", emitter.EmitterSet, Sequence[emitter.EmitterSet])


class ROI(ABC):
    def crop_frames(self, frame: _T_Frame) -> _T_Frame:
        raise NotImplementedError

    def crop_emitter(self, em: _T_Emitter) -> _T_Emitter:
        raise NotImplementedError


class ROIBlock(ROI):
    def __init__(
        self,
        ix_x: tuple[int, int],
        ix_y: tuple[int, int],
        xextent: Optional[tuple[float, float]] = None,
        yextent: Optional[tuple[float, float]] = None,
        img_shape: Optional[tuple[int, int]] = None,
        rref_pos: bool = False,
    ):
        """
        ROI

        Args:
            ix_x: lower and upper index of roi in x
            ix_y: lower and upper index of roi in y
            xextent: extent in x of frame
            yextent: extent in y of frame
            img_shape: shape of frame
            rref_pos: if True, emitter are re-referenced to ROI after cropping
        """
        self._ix_x = ix_x
        self._ix_y = ix_y
        self._xextent = xextent
        self._yextent = yextent
        self._img_shape = img_shape
        self._fov = None
        self._rref_pos = rref_pos

        if xextent is not None and yextent is not None and img_shape is not None:
            # get x/y extent of roi by extent and indices of frame
            bin_x, bin_y, *_ = utils.frame_grid(
                img_size=img_shape, xextent=xextent, yextent=yextent
            )
            xextent_roi = (bin_x[self._ix_x[0]], bin_x[self._ix_x[1]])
            yextent_roi = (bin_y[self._ix_y[0]], bin_y[self._ix_y[1]])
            self._fov = emitter.process.EmitterFilterFoV(
                xextent=xextent_roi, yextent=yextent_roi
            )

    def crop_frames(self, frames: _T_Frame) -> _T_Frame:
        f = [frames] if isinstance(frames, torch.Tensor) else frames
        f = [
            ff[..., self._ix_x[0] : self._ix_x[1], self._ix_y[0] : self._ix_y[1]]
            for ff in f
        ]
        return f[0] if isinstance(frames, torch.Tensor) else f

    def crop_emitter(self, em: _T_Emitter) -> _T_Emitter:
        em_ = [em] if isinstance(em, emitter.EmitterSet) else em
        em_ = [self._fov.forward(e) for e in em_]
        if self._rref_pos:
            for i, e in enumerate(em_):
                em_[i].xyz_px = e.xyz_px - torch.tensor(
                    [self._ix_x[0], self._ix_y[0], 0.0]
                )

        return em_[0] if isinstance(em, emitter.EmitterSet) else em_


roi_sampler_mode = Literal["inner"]
roi_sampler_strategy = Literal["uniform"]


class ROISampler:
    def __init__(
        self,
        sampler: torch.distributions.Distribution,
        roi_shape: tuple[int, int],
        img_shape: tuple[int, int],
        xextent: Optional[tuple[float, float]] = None,
        yextent: Optional[tuple[float, float]] = None,
        rref_pos: bool = False,
    ):
        """Sample ROIs from an image.

        Args:
            sampler (torch.distributions.Distribution): Distribution to sample from.
            roi_shape: roi shape
            img_shape (tuple): Shape of the frame.
            xextent: extent in x of frame
            yextent: extent in y of frame
            rref_pos: if True, emitter are re-referenced to ROI after cropping
        """
        self._img_shape = img_shape
        self._sampler = sampler
        self._roi_shape = roi_shape
        self._xextent = xextent
        self._yextent = yextent
        self._rref_pos = rref_pos

    def sample(self, n: int = None) -> Union[ROI, list[ROI]]:
        """Sample ROIs.

        Args:
            n (int): Number of ROIs to sample.

        Returns:
            ROI | list[ROI]: ROI or list of ROIs.

        """
        # sample lower left corner of roi
        ix_low = self._sampler.sample((n,) if n is not None else (1,))
        ix_high = ix_low + torch.tensor(self._roi_shape)
        r = [
            ROIBlock(
                ix_x=(low[0].item(), high[0].item()),
                ix_y=(low[1].item(), high[1].item()),
                img_shape=self._img_shape,
                xextent=self._xextent,
                yextent=self._yextent,
                rref_pos=self._rref_pos,
            )
            for low, high in zip(ix_low, ix_high)
        ]
        return r[0] if n is None else r

    @classmethod
    def factory(
        cls,
        roi_shape: tuple[int, int],
        img_shape: tuple[int, int],
        xextent: Optional[tuple[float, float]] = None,
        yextent: Optional[tuple[float, float]] = None,
        mode: roi_sampler_mode = "inner",
        strategy: roi_sampler_strategy = "uniform",
        rref_pos: bool = False,
    ) -> "ROISampler":
        """Factory method to create a ROISampler."""
        return cls(
            sampler=cls.get_sampler(
                roi_shape=roi_shape, img_shape=img_shape, mode=mode, strategy=strategy
            ),
            roi_shape=roi_shape,
            img_shape=img_shape,
            xextent=xextent,
            yextent=yextent,
            rref_pos=rref_pos,
        )

    @classmethod
    def get_sampler(
        cls,
        roi_shape: tuple[int, int],
        img_shape: tuple[int, int],
        mode: roi_sampler_mode,
        strategy: roi_sampler_strategy,
    ) -> torch.distributions.Distribution:
        if mode not in ["inner"]:
            raise NotImplementedError(f"Mode {mode} not yet supported.")
        if strategy not in ["uniform"]:
            raise NotImplementedError(f"Strategy {strategy} not yet supported.")

        high = torch.tensor(img_shape) - torch.tensor(roi_shape)
        sampler = torch_utils.UniformInt(torch.zeros(2), high, dtype=torch.long)

        return sampler


@dev.experimental(tested=False, level="warning")
def roi_mirr_shift(frame_size: int, roi_size: int, roi_pos: int) -> int:
    """
    Calculate shift induced by cropping and de-mirroring. This calculates it along
    the dimension which is cropped and mirrored.

    Args:
        frame_size:
        roi_size:
        roi_pos:

    Returns:

    """
    return 2 * roi_pos + roi_size - frame_size
