"""SMAPpy's spline calibration as DECODE's CubicSplinePSF takes it.

DECODE read a SMAP ``_3dcal.mat`` itself (``SMAPSplineCoefficient``): the raw ``cspline.coeff``,
``(n1, n2, nz, 64)`` in MATLAB's order, with ``ref0 = (x0 - 1, x0 - 1, z0)``.  Here the file comes
in through SMAPpy's ``load_spline_calibration``, which reads native ``.h5`` calibrations too and
hands back a spline in the camera's orientation (a spline of x-mirrored EM beads is un-mirrored,
``smappy.io.calibration._unmirror_x``), as ``(64, nz, ny, nx)`` with the monomial index
``16 pz + 4 py + px``.

**The axes.**  DECODE's x is the frame's first axis, its rows: SMAPpy's y (docs/decode/poc-plan.md
§6, item 4).  Its coefficients are ``coeff[x_D, y_D, z, k]`` with ``k = 16 pz + 4 p_yD + p_xD`` --
the fastest digit its x -- so SMAPpy's are transposed to ``(ny, nx, nz, 64)`` and their two lateral
digits exchanged.  Without a mirror this is a permutation of the file's values: the result is
DECODE's own reading, bitwise (tests/decode_tests/test_calibration_io.py).

**The reference** is SMAP's lateral centre ``x0`` (1-based, one value for both axes) less one, and
``z0`` -- DECODE's own.

**A mirrored calibration.**  SMAP's spline of x-mirrored EM beads, which DECODE drew as it was
(and the Fit notebook mirrored the frames to match), comes un-mirrored from SMAPpy: ``S'(u) =
S(n - u)`` along the columns, with ``n - x0`` as its ``x0``.  DECODE's ROI is not mirror-symmetric
-- an emitter's window reaches voxels 0 to n - 2 of the spline along an axis, so mirrored it would
reach 1 to n - 1 -- and drawn at the mirrored reference the PSFs differed by up to 16 % of the peak
in an edge column at 800 nm defocus (the example package's calibration, saved as mirrored).  So the
un-mirrored coefficients are moved one voxel along the columns (the first dropped, a zero column
appended), with SMAPpy's ``x0`` itself as the reference (``n - r`` less the voxel): then the PSF an emitter draws is the
file's PSF mirrored, to float32 rounding (1.5e-7 of the peak, every z).  The exception is an emitter
exactly on a pixel centre in y, where DECODE's own window jumps to all n voxels (its fractional part
0); uniform float32 positions meet it about once in 8 million emitters.  The rows keep the file's
``x0 - 1``, which is ``n - x0`` of SMAPpy's, the grid being square.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# k_D = 16 pz + 4 p_yD + p_xD reads SMAPpy's 16 pz + 4 py + px with p_xD = py and p_yD = px
_k = np.arange(64)
_DIGIT_SWAP = 16 * (_k // 16) + 4 * (_k % 4) + (_k // 4) % 4


def decode_coefficients(cal) -> tuple[torch.Tensor, tuple[float, float, float], float]:
    """``(coeff, ref0, dz)`` for CubicSplinePSF from a `smappy.io.calibration.SplineCalibration`.

    ``coeff`` is ``(ny, nx, nz, 64)`` float32, DECODE's x the rows; ``ref0`` in DECODE's axis order.
    """
    c = np.asarray(cal.coeff, dtype=np.float32)
    if c.ndim != 4 or c.shape[0] != 64:
        raise ValueError(f"a spline calibration's coefficients are (64, nz, ny, nx), not {c.shape}")
    _, nz, ny, nx = c.shape
    if cal.x0 is None:
        raise ValueError("the calibration has no lateral centre x0; DECODE's reference needs one")
    coeff = np.ascontiguousarray(c.transpose(2, 3, 1, 0)[..., _DIGIT_SWAP])      # (ny, nx, nz, 64)
    flipped = "smappy_conversion" in (cal.parameters or {})
    if flipped:
        if nx not in (ny, ny + 1):
            raise ValueError(f"an un-mirrored spline of {ny} x {nx} voxels: SMAP's grids are square")
        # one voxel along the columns, so DECODE's window reaches the voxels it reached in the file
        coeff = np.concatenate([coeff[:, 1:], np.zeros_like(coeff[:, :1])], axis=1)
        ref_rows = (ny - float(cal.x0)) - 1.0      # the file's x0 - 1: SMAPpy's x0 is n - x0_file
        # S(c - p + r) mirrored is S(n - u) at r' = n - r = x0 + 1; one voxel along, x0
        ref_cols = float(cal.x0)
    else:
        ref_rows = ref_cols = float(cal.x0) - 1.0
    ref0 = (ref_rows, ref_cols, float(cal.z0))
    return torch.from_numpy(coeff), ref0, float(cal.dz)


def spline_psf(calibration, xextent, yextent, img_shape, device="cpu", **kwargs):
    """DECODE's CubicSplinePSF for a calibration file (``_3dcal.mat`` or SMAPpy ``.h5``) or a loaded one.

    In DECODE's coordinates: x the frame's rows, in pixels; z in nm.
    """
    from decode_smlm.simulation.psf_kernel import CubicSplinePSF

    if isinstance(calibration, (str, Path)):
        from smappy.io.calibration import load_spline_calibration
        calibration = load_spline_calibration(calibration)
    coeff, ref0, dz = decode_coefficients(calibration)
    return CubicSplinePSF(xextent=xextent, yextent=yextent, img_shape=img_shape, ref0=ref0, coeff=coeff,
                          vx_size=(1.0, 1.0, dz), device=device, **kwargs)
