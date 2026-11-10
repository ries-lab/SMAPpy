"""The torch spline renderer against DECODE v0.10.2's own: milestone (c) of docs/decode/poc-plan.md.

``fixtures/v0102_renderer.npz`` holds what CubicSplinePSF drew with the original's compiled ``spline``
(``scripts/decode/reference_v0102.py oracles``): 1032 emitters on a 32 x 24 frame -- inside it, on
and across its border, outside the calibration's z range, exactly on pixel centres and voxel
boundaries -- from a synthetic calibration.  The port draws them again with the same calibration
and the same calls, and must agree to 1e-6 of the largest value, on the CPU and on the GPU there is.
"""
import pathlib

import numpy as np
import pytest
import torch

from decode_smlm.simulation.psf_kernel import CubicSplinePSF
from decode_smlm.simulation.spline_torch import ROI_OUT_EPS, SplineTorch, monomials
from decode_smlm.utils.hardware import default_device

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
DEVICES = ["cpu"] + ([default_device()] if default_device() != "cpu" else [])
REL = 1e-6


@pytest.fixture(scope="module")
def oracle():
    return dict(np.load(FIXTURES / "v0102_renderer.npz"))


def _psf(o, device):
    return CubicSplinePSF(xextent=tuple(o["xextent"]), yextent=tuple(o["yextent"]),
                          img_shape=tuple(int(v) for v in o["frame_shape"]), ref0=tuple(o["ref0"]),
                          coeff=torch.from_numpy(o["coeff"]), vx_size=(1.0, 1.0, float(o["dz"])), device=device)


def _close(port, original, rel=REL):
    port, original = np.asarray(port, np.float64), np.asarray(original, np.float64)
    assert port.shape == original.shape
    err = np.abs(port - original).max() / np.abs(original).max()
    assert err <= rel, err


@pytest.mark.parametrize("device", DEVICES)
def test_the_rois_are_the_originals(oracle, device):
    rois = _psf(oracle, device).forward_rois(torch.from_numpy(oracle["xyz"]), torch.from_numpy(oracle["phot"]))
    _close(rois, oracle["rois"])


@pytest.mark.parametrize("device", DEVICES)
def test_the_frames_are_the_originals(oracle, device):
    frames = _psf(oracle, device).forward(torch.from_numpy(oracle["xyz"]), torch.from_numpy(oracle["phot"]),
                                          torch.from_numpy(oracle["frame_ix"]), 0, int(oracle["n_frames"]) - 1)
    _close(frames, oracle["frames"])


@pytest.mark.parametrize("device", DEVICES)
def test_the_derivatives_are_the_originals(oracle, device):
    d = slice(0, None, int(oracle["drv_step"]))
    drv, rois_bg = _psf(oracle, device).derivative(torch.from_numpy(oracle["xyz"][d]),
                                                   torch.from_numpy(oracle["phot"][d]), torch.from_numpy(oracle["bg"][d]))
    for k in range(5):  # each derivative on its own scale: x, y, z, phot, bg
        _close(drv[:, k], oracle["drv"][:, k])
    _close(rois_bg, oracle["rois_bg"])


@pytest.mark.parametrize("kind,name", [(1, "on or across the frame's border"), (2, "outside the z range"),
                                       (3, "on pixel centres and voxel boundaries")])
def test_the_edge_cases_agree_on_their_own(oracle, kind, name):
    """The kinds of emitter where an implementation can differ, each against its own largest value."""
    sel = oracle["kind"] == kind
    rois = _psf(oracle, "cpu").forward_rois(torch.from_numpy(oracle["xyz"][sel]), torch.from_numpy(oracle["phot"][sel]))
    _close(rois, oracle["rois"][sel])


def test_a_pixel_outside_the_grid_is_the_kernels_epsilon_not_times_the_photons():
    coeff = torch.ones(5, 5, 4, 64)
    impl = SplineTorch(coeff, (5, 5))
    # an emitter two voxels off in x: the last two pixel rows read outside the 5-voxel grid
    rois = impl.forward_rois(torch.tensor([2.0]), torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([1000.0]))
    assert torch.all(rois[0, 3:] == torch.tensor(ROI_OUT_EPS, dtype=torch.float32))
    assert torch.all(rois[0, :3] == 1000.0)          # inside: phot * sum of monomials at delta 0 = phot * 1


def test_z_beyond_the_grid_clamps_the_voxel_and_keeps_the_delta():
    torch.manual_seed(0)
    coeff = torch.rand(6, 6, 4, 64)
    impl = SplineTorch(coeff, (6, 6))
    z = torch.tensor([5.25])                          # voxel 5, beyond the last (3): clamped to 3, delta 0.25
    roi = impl.forward_rois(torch.tensor([0.0]), torch.tensor([0.0]), z, torch.tensor([1.0]))[0]
    m = monomials(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.25]))[0]
    assert torch.allclose(roi, coeff[:, :, 3] @ m, rtol=1e-6)


def test_coordinates_of_many_kinds_draw_the_same_as_grouped():
    """Grouping the emitters by voxel and gathering each one's own slab are the same computation."""
    torch.manual_seed(1)
    impl = SplineTorch(torch.rand(9, 9, 10, 64), (9, 9))
    n = 600
    x, y, z = torch.rand(n) * 4 - 2, torch.rand(n) * 4 - 2, torch.rand(n) * 12 - 1
    x0, y0, z0, dx, dy, dz = impl._split(x, y, z)
    m = monomials(dx, dy, dz)[:, None]
    grouped, inside_g = impl._contract(m, x0, y0, z0)
    gathered, inside = impl._contract_gathered(m, x0, y0, z0)
    assert torch.equal(inside, inside_g)
    assert torch.allclose(grouped, gathered, rtol=1e-5, atol=1e-6)
