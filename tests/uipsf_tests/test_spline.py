"""The spline localiser of bead scoring (`workflows.spline.SplineFrames`): SMAPpy's cubic-spline fitter on
a voxel PSF sampled from the model.  Skips without SMAPpy."""
import numpy as np
import pytest
import torch

import uipsf
from uipsf.noise import Poisson, ShiftedPoisson
from uipsf.workflows.scoring import slice_fits
from uipsf_tests.conftest import small_beads

pytest.importorskip("smappy._fit3d")
from uipsf.workflows.spline import SplineFrames  # noqa: E402


@pytest.mark.parametrize("read_noise", [0.0, 25.0])
def test_the_spline_localises_every_slice_where_the_pupil_model_does(optics, read_noise):
    # Conventions (z sign, the ROI's centre, the shifted Poisson) are what this pins: a wrong sign or a
    # half-pixel offset would show as a bias of tens of nm against the pupil model's own fits.
    model, truth, batch = small_beads(optics, n=4, dtype=torch.float32, roi=(13, 13), photons=20000.0,
                                      background=20.0)
    if read_noise:
        g = torch.Generator().manual_seed(3)
        batch = uipsf.Batch(data=batch.data + torch.randn(batch.data.shape, generator=g) * read_noise ** 0.5,
                            z_nm=batch.z_nm, s=read_noise)
    noise = ShiftedPoisson() if read_noise else Poisson()
    shared = {k: v for k, v in truth.physical().items() if truth.specs[k].scope == "shared"}
    a, crlb, stack = slice_fits(model, shared, batch, noise, truth)
    b, crlb_b, stack_b = SplineFrames()(model, shared, batch, noise, truth)
    assert np.allclose(stack, stack_b)
    central = np.abs(stack[..., 2]) < 450
    for i in range(3):
        diff = (b - a)[..., i][central]
        # 0.5 nm in y (0.004 px, the spline on non-square pixels): a convention error would be tens of nm
        assert abs(diff.mean()) < 0.3 * np.median(crlb[..., i][central]), "xyz"[i]
        assert np.sqrt((diff ** 2).mean()) < 0.5 * np.median(crlb[..., i][central]), "xyz"[i]    # spline interpolation
        assert np.allclose(np.median(crlb_b[..., i][central]), np.median(crlb[..., i][central]), rtol=0.1)
