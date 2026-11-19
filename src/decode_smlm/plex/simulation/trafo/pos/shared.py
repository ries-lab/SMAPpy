# shared frame and positional transformation

import torch

from . import trafo
from . import frame
from ... import psf_kernel


class PipedTransformation:
    def __init__(
        self,
        trafo_pos: trafo.XYZTransformation,
        psf: psf_kernel.PSF,
        trafo_frame: frame.FrameTransformation,
    ):
        """
        Transformation to wrap positional transformation, frame computation by psf
        and frame transformation. Useful e.g. for flipping coordinates and then the
        frame when the PSF is not symmetric.

        Args:
            trafo_pos:
            psf:
            trafo_frame:
        """
        self._trafo_pos = trafo_pos
        self._psf = psf
        self._trafo_frame = trafo_frame

    def forward(
        self,
        xyz: torch.Tensor,
        weight: torch.Tensor,
        frame_ix: torch.Tensor | None = None,
        ix_low: int | None = None,
        ix_high: int | None = None,
    ):
        xyz_m = self._trafo_pos.forward(xyz)
        f = self._psf.forward(xyz_m, weight, frame_ix, ix_low, ix_high)
        f = self._trafo_frame.forward(f)

        return f
