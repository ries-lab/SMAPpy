"""A SMAP calibration through SMAPpy's loader draws what DECODE drew from the file itself.

DECODE read ``SXY.cspline`` raw (``SMAPSplineCoefficient``: the coefficients as stored, ``ref0 =
(x0 - 1, x0 - 1, z0)``); the port reads it with ``smappy.io.calibration.load_spline_calibration``
and converts (``decode_smlm.utils.calibration_io``).  Without a mirror the two must be the same
numbers; with one, the un-mirrored spline drawn at an emitter must be the mirror image of the
file's spline drawn at the emitter's mirror image.  And SMAPpy's own ``evaluate_spline`` draws
the same PSF inside the ROI; on its first row and column it extrapolates where DECODE's kernel
gives 1e-10 (docs/decode/poc-plan.md §6, item 6).
"""
import os
import pathlib

import numpy as np
import pytest
import scipy.io as sio
import torch

from decode_smlm.simulation.psf_kernel import CubicSplinePSF
from decode_smlm.utils.calibration_io import decode_coefficients, spline_psf
from smappy.io.calibration import evaluate_spline, load_spline_calibration

REAL = pathlib.Path(os.environ.get("DECODE_SMLM_DATA", pathlib.Path.home() / ".cache" / "decode_smlm")) \
    / "experimental_data_workflow" / "spline_calibration_3dcal.mat"
FRAME = dict(xextent=(-0.5, 39.5), yextent=(-0.5, 31.5), img_shape=(40, 32))   # not square: x and y show


def _write_smap(path, coeff, x0, z0, dz, mirrored):
    """A SMAP _3dcal.mat as calibrate3D_g writes it, as far as the readers look."""
    sio.savemat(path, {"SXY": {"cspline": {"coeff": coeff, "x0": x0, "z0": z0, "dz": dz},
                               "EMon": int(mirrored)},
                       "parameters": {"emmirror": 3 if mirrored else 2}})


def _decode_raw(path, device="cpu"):
    """DECODE v0.10.2's SMAPSplineCoefficient(...).init_spline(...), which the port does not keep."""
    cs = sio.loadmat(path, struct_as_record=False, squeeze_me=True)["SXY"].cspline
    return CubicSplinePSF(ref0=(cs.x0 - 1, cs.x0 - 1, cs.z0), coeff=torch.from_numpy(np.asarray(cs.coeff, np.float32)),
                          vx_size=(1.0, 1.0, float(cs.dz)), device=device, **FRAME)


@pytest.fixture
def synthetic(tmp_path):
    rng = np.random.default_rng(3)
    n, nz = 14, 21
    env = np.exp(-0.5 * ((np.arange(n) - 6.5) / 3) ** 2)
    coeff = (rng.normal(0, 1, (n, n, nz, 64)) * (env[:, None] * env[None, :])[..., None, None]).astype(np.float32)
    coeff[..., 0] += 2 * (env[:, None] * env[None, :])[..., None]
    return coeff


def _emitters(n=300, seed=4):
    g = torch.Generator().manual_seed(seed)
    xyz = torch.stack([torch.rand(n, generator=g) * 40 - 0.5, torch.rand(n, generator=g) * 32 - 0.5,
                       torch.rand(n, generator=g) * 1600 - 800], 1)
    return xyz, torch.rand(n, generator=g) * 5000 + 500, torch.randint(0, 6, (n,), generator=g)


def test_without_a_mirror_the_coefficients_and_reference_are_decodes_own(tmp_path, synthetic):
    path = tmp_path / "beads_3dcal.mat"
    _write_smap(path, synthetic, x0=7, z0=10, dz=50.0, mirrored=False)
    coeff, ref0, dz = decode_coefficients(load_spline_calibration(path))
    assert torch.equal(coeff, torch.from_numpy(synthetic))          # a permutation of the file's values
    assert ref0 == (6.0, 6.0, 10.0) and dz == 50.0


def test_an_unmirrored_spline_draws_the_mirror_image_of_the_files(tmp_path, synthetic):
    """EM beads were x-mirrored: the file's PSF lives in the mirrored frame, SMAPpy's in the camera's."""
    path = tmp_path / "em_beads_3dcal.mat"
    _write_smap(path, synthetic, x0=7, z0=10, dz=50.0, mirrored=True)
    raw, ours = _decode_raw(path), spline_psf(path, **FRAME)
    xyz, phot, frame = _emitters()
    mirrored = xyz.clone()
    mirrored[:, 1] = (FRAME["img_shape"][1] - 1) - xyz[:, 1]          # DECODE's y is SMAPpy's x, the columns
    a = raw.forward(xyz, phot, frame, 0, 5).flip(-1)
    b = ours.forward(mirrored, phot, frame, 0, 5)
    assert (a - b).abs().max() <= 1e-6 * a.abs().max()


def test_an_emitter_exactly_on_a_pixel_centre_is_where_the_mirror_cannot_be_exact(tmp_path, synthetic):
    """DECODE's window reaches all n voxels at a fractional part of 0 and n - 1 otherwise: mirrored, the
    other edge.  Pinned so that a change to it is seen; uniform positions almost never meet it."""
    path = tmp_path / "em_beads_3dcal.mat"
    _write_smap(path, synthetic, x0=7, z0=10, dz=50.0, mirrored=True)
    raw, ours = _decode_raw(path), spline_psf(path, **FRAME)
    one = torch.tensor([1.0]), torch.tensor([0]), 0, 0
    a = raw.forward(torch.tensor([[20.4, 15.0, 100.0]]), *one)[0].flip(-1)
    b = ours.forward(torch.tensor([[20.4, 31 - 15.0, 100.0]]), *one)[0]
    assert (a - b).abs().max() > 1e-3 * a.max()
    a = raw.forward(torch.tensor([[20.4, 15.25, 100.0]]), *one)[0].flip(-1)
    b = ours.forward(torch.tensor([[20.4, 31 - 15.25, 100.0]]), *one)[0]
    assert (a - b).abs().max() <= 1e-6 * a.max()


@pytest.mark.skipif(not REAL.exists(), reason=f"the example package's calibration is not in {REAL.parent}")
def test_the_example_packages_calibration_draws_decodes_frames():
    raw, ours = _decode_raw(REAL), spline_psf(REAL, **FRAME)
    xyz, phot, frame = _emitters()
    a, b = raw.forward(xyz, phot, frame, 0, 5), ours.forward(xyz, phot, frame, 0, 5)
    assert torch.equal(a, b)


@pytest.mark.skipif(not REAL.exists(), reason=f"the example package's calibration is not in {REAL.parent}")
def test_evaluate_spline_draws_the_same_psf_inside_the_roi_and_not_on_its_first_row_and_column():
    cal = load_spline_calibration(REAL)
    psf = spline_psf(cal, **FRAME)
    n = cal.coeff.shape[-1]
    rng = np.random.default_rng(0)
    inside, edge = 0.0, 0.0
    for _ in range(50):
        fx, fy, z = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(-900, 900)
        d = psf.forward_rois(torch.tensor([[fx, fy, z]]), torch.tensor([1.0]))[0].numpy().astype(np.float64)
        # SMAPpy's ROI coordinates: x the columns (DECODE's y), the centre at (R - 1) / 2; DECODE's
        # reference x0 - 1 puts the emitter half a pixel further: 13 + f against 12.5 + f for R = 26
        s = evaluate_spline(cal, n / 2 - 0.5 + fy, n / 2 - 0.5 + fx, cal.z_nm_to_index(z), n).astype(np.float64)
        diff = np.abs(d - s) / d.max()
        inside, edge = max(inside, diff[1:, 1:].max()), max(edge, max(diff[0].max(), diff[:, 0].max()))
    assert inside < 1e-6
    assert edge > 1e-3   # the edge differs: DECODE's 1e-10 against SMAPpy's clamped extrapolation
