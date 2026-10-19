"""Single-slice localisation with SMAPpy's cubic-spline fitter: the fast localiser of bead scoring.

Bead scoring localises every slice of every bead on its own to measure the z bias (`scoring`): on
Zenodo, 84 beads × 40 slices.  Fitted on the pupil model that took 86 s (`scoring.slice_fits`).  Here
the model is sampled once into a voxel PSF -- the whole forward model, bead size and blur included --
and SMAPpy makes the spline and fits every frame in C++.  The z bias then is what a user localising
with that spline in SMAPpy will see, as v1 measured it with its spline (``msez``).

uipsf does not fit or write splines itself (CLAUDE.md): `spline_coefficients` and the fitter are
SMAPpy's, imported when used.  Without SMAPpy, `scoring.slice_fits` remains the localiser.

Conventions (SMAPpy's `io.calibration`, checked in tests/uipsf_tests/test_spline.py):
- the volume's planes run with the objective, emitter z = −(index − z0)·dz, z0 its middle plane;
- the ROI sits centred in the larger spline grid, and for an odd ROI x_roi = (X − 1)/2 is its centre
  pixel, x = 0 here;
- the fitter is Poisson: the read noise is handled as the shifted Poisson does, the variance s added
  to the data and absorbed by the fitted background.
"""
from __future__ import annotations

import numpy as np
import torch

from ..model import VectorPSF


class SplineFrames:
    def __init__(self, dz_nm: float = 10.0, margin_px: int = 3, z_margin_nm: float = 500.0,
                 iterations: int = 50):
        """``dz_nm`` the spline's z step; ``margin_px`` pixels around the ROI in the spline grid (the
        fitter needs room for the spot's shift); ``z_margin_nm`` beyond the slices' z range."""
        self.dz_nm, self.margin_px = float(dz_nm), int(margin_px)
        self.z_margin_nm, self.iterations = float(z_margin_nm), int(iterations)

    def calibration(self, model, shared: dict, roi_shape, z_range_nm: tuple[float, float]):
        """SMAPpy's `SplineCalibration` of ``model`` on ``shared``, sampled over ``z_range_nm``."""
        from smappy.calibrate.core import spline_coefficients
        from smappy.io.calibration import SplineCalibration
        if model.n_channels > 1:
            raise NotImplementedError("SplineFrames fits one channel; SMAPpy's dual fitter is not wired here")
        Y, X = (int(n) + 2 * self.margin_px for n in roi_shape)
        psf0 = model.psf
        psf = VectorPSF(psf0.grid, psf0.pupil, (Y, X), upsample=psf0.upsample,
                        bead_radius_nm=psf0.bead_radius_nm)
        lo, hi = z_range_nm[0] - self.z_margin_nm, z_range_nm[1] + self.z_margin_nm
        half = int(np.ceil(max(abs(lo), abs(hi)) / self.dz_nm))
        n = 2 * half + 1
        z0 = float(half)
        z = -(torch.arange(n, dtype=torch.float32) - z0) * self.dz_nm          # emitter z of each plane
        xyz = torch.zeros(1, n, 3)
        xyz[0, :, 2] = z
        dev = psf0.grid.device
        p = {k: v.detach().to(dev) for k, v in shared.items()}
        with torch.no_grad():
            vol = psf(p, xyz.to(dev, psf0.grid.dtype), z_step_nm=self.dz_nm)[0].cpu().double().numpy()   # [n, Y, X]
        vol = np.maximum(vol, 0)
        vol = (vol / vol.sum((1, 2)).max()).astype(np.float32)               # SMAPpy: max plane sum 1
        return SplineCalibration(spline_coefficients(vol).astype(np.float32), self.dz_nm, z0,
                                 x0=(X - 1) / 2, psf=vol)

    def __call__(self, model, shared: dict, batch, noise, stack_fit):
        """`scoring.slice_fits`' contract: (xyz [E, K, 3], CRLB [E, K, 3], stack-fit xyz [E, K, 3]), nm."""
        from smappy import _fit3d
        E, C, K, Y, X = batch.data.shape
        stack = model.geometry(stack_fit.physical(), batch).detach().double().cpu().numpy()   # [E, K, 3]
        cal = self.calibration(model, shared, (Y, X), (float(stack[..., 2].min()), float(stack[..., 2].max())))
        s = batch.s
        s = float(s) if not isinstance(s, torch.Tensor) or s.dim() == 0 else s.expand_as(batch.data).cpu().numpy()
        data = (batch.data + 0).double().cpu().numpy() + s                     # shifted Poisson
        data = np.maximum(data[:, 0], 0).astype(np.float32)                     # [E, K, Y, X]
        px, py = model.pixel_size_nm
        found = np.zeros((E, K, 3))
        crlb = np.zeros((E, K, 3))
        for k in range(K):                                                     # one z start per slice
            z_start = float(cal.z_nm_to_index(np.median(stack[:, k, 2])))
            theta, var, *_ = _fit3d.fit_cspline(np.ascontiguousarray(data[:, k]), cal.coeff, z_start,
                                                self.iterations, 0)
            theta, var = np.asarray(theta), np.clip(np.asarray(var), 0, None)
            found[:, k, 0] = (theta[:, 0] - (X - 1) / 2) * px
            found[:, k, 1] = (theta[:, 1] - (Y - 1) / 2) * py
            found[:, k, 2] = cal.z_index_to_nm(theta[:, 4])
            crlb[:, k] = np.sqrt(var[:, [0, 1, 4]]) * np.array([px, py, self.dz_nm])
        return found, crlb, stack
