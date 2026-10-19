"""Several channels (M4): one emitter, each channel with its own pupil, a projective transform to the main
channel, a photon share and a defocus offset (plan §3, §9 M4)."""
import os
import sys

import numpy as np
import pytest
import torch

import uipsf
from uipsf.model import BeadStackGeometry
from uipsf.model.channels import map_points, unpack_transform
from uipsf.noise import Poisson
from uipsf.params import Params
from uipsf.solve import LevenbergMarquardt, Problem
from uipsf_tests.conftest import SPLIT_TRANSFORM, make_psf, stack, two_channel_beads

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "uipsf", "synthetic"))
from beads import rescale  # noqa: E402

f64 = torch.float64


@pytest.mark.parametrize("mirrored", [False, True])
def test_the_secondary_channel_sees_the_emitter_where_the_transform_maps_it_on_the_chip(optics, mirrored):
    model, truth, batch = two_channel_beads(optics, mirrored=mirrored)
    p = truth.physical()
    xyz = model.geometry(p, batch)
    px = torch.tensor(optics.pixel_size_nm, dtype=f64)
    main = batch.roi_xy[:, None, 0] + xyz[..., :2] / px                 # chip pixels in the main half
    second = batch.roi_xy[:, None, 1] + model.positions(p, batch, xyz, 1)[..., :2] / px
    H = unpack_transform(p["ch1.transform"])
    assert torch.allclose(map_points(H, second), main, atol=1e-9)      # secondary → main, SMAPpy's direction
    assert torch.allclose(model.positions(p, batch, xyz, 1)[..., 2], xyz[..., 2] + p["ch1.dz"])
    # A mirrored splitter turns a step along +y in the main half into one along −y in the secondary.
    step = xyz.clone()
    step[..., 1] += 10.0
    dx = model.positions(p, batch, step, 1)[..., 1] - model.positions(p, batch, xyz, 1)[..., 1]
    assert ((dx < 0) if mirrored else (dx > 0)).all()


def test_the_photons_are_the_emitters_total_split_by_the_shares(optics):
    # Each channel's PSF holds one photon in focus, so the share is the fraction of the emitter's photons.
    model, truth, batch = two_channel_beads(optics, kind="ratiometric", n=2, roi=(31, 31), P=48, z_range=0.0,
                                            K=3)
    p = truth.physical()
    mu = model(p, batch) - model.background(p, batch)
    light = mu.sum((-2, -1))[:, :, 1] / p["emitter.photons"][:, None]       # the slice at the main focus
    assert torch.allclose(light[:, 1] / light[:, 0], torch.tensor(0.3 / 0.7, dtype=f64).expand(2), rtol=0.03)
    assert float(model.share(p, 0)) == pytest.approx(0.7)


def test_each_emitter_sees_only_its_own_parameters_in_every_channel(optics):
    model, truth, batch = two_channel_beads(optics, n=3)
    ue = truth.emitter_matrix()
    tangent = torch.zeros_like(ue)
    tangent[1] = 1.0
    _, dmu = torch.func.jvp(lambda v: model(truth.replace(emitter=v).physical(), batch), (ue,), (tangent,))
    assert dmu[1, 0].abs().max() > 0 and dmu[1, 1].abs().max() > 0
    assert dmu[0].abs().max() == 0 and dmu[2].abs().max() == 0


@pytest.mark.parametrize("case", ["plain", "bead kernel, drift", "mirrored, planar"])
def test_the_linearised_jacobians_are_the_jvps_in_every_channel(optics, case):
    # The M7 speed paths per channel: pupil columns, emitter columns, and the transform, share and offset
    # from the emitter linearisation; with a bead kernel the drift is not stack-uniform and falls back.
    kw = {"plain": {}, "bead kernel, drift": dict(bead_radius_nm=50.0, geometry=BeadStackGeometry(drift=True)),
          "mirrored, planar": dict(mirrored=True)}[case]
    model, truth, batch = two_channel_beads(optics, n=3, **kw)
    if case == "mirrored, planar":
        from uipsf.model import Planar
        model.background = Planar(optics.pixel_size_nm)
        truth = Params(model.specs(batch), 3, dtype=f64, values=truth.physical())
    p = truth.replace(shared=truth.shared_vector() * 1.001 + 0.01, emitter=truth.emitter_matrix() * 1.02 + 1e-3)
    a = Problem(model, Poisson(), batch, linearise=False).normal_equations(p)
    b = Problem(model, Poisson(), batch, tangents=3).normal_equations(p)
    for k in ("H_ss", "H_se", "H_ee", "g_s", "g_e"):
        assert torch.allclose(getattr(b, k), getattr(a, k), rtol=1e-10, atol=1e-7), k


def test_chunks_and_units_change_nothing_with_two_channels(optics):
    model, truth, batch = two_channel_beads(optics, n=4)
    start = truth.replace(shared=truth.shared_vector() * 1.001 + 0.01)
    ref = Problem(model, Poisson(), batch, chunk_size=4).normal_equations(start)
    for size in (1, 3):
        ne = Problem(model, Poisson(), batch, chunk_size=size, tangents=size).normal_equations(start)
        for k in ("H_ss", "H_se", "H_ee", "g_s", "g_e"):
            assert torch.allclose(getattr(ne, k), getattr(ref, k), rtol=1e-10, atol=1e-10), k
    problem = Problem(model, Poisson(), batch)
    lm = LevenbergMarquardt(max_iter=4)
    a = lm.solve(problem, start)
    b = lm.solve(problem, rescale(start, {"ch1.transform": 1e-3, "ch1.share": 10.0, "ch1.dz": 1e2,
                                          "ch1.pupil.phase": 0.1}))
    for h0, h1 in zip(a.history, b.history, strict=True):
        assert abs(h0["objective"] - h1["objective"]) < 1e-8 * abs(h0["objective"])
    for k, v in a.params.physical().items():
        assert torch.allclose(v, b.params.physical()[k], rtol=1e-6, atol=1e-8), k


def test_the_rough_start_finds_the_shares_and_the_biplane_offset(optics):
    model, truth, batch = two_channel_beads(optics, n=8, roi=(17, 17), P=32, K=21, z_range=1000.0)
    start = uipsf.model.rough_start(batch, model.pixel_size_nm)
    assert start["shares"][1] == pytest.approx(0.5, abs=0.1)
    assert start["dz"][1] == pytest.approx(350.0, abs=150.0)          # three slices of 100 nm
    assert start["background"].shape == (8, 2)


def test_a_channel_is_unpolarised_unless_it_says_otherwise_and_x_plus_y_is_unpolarised(optics):
    # Review F1: v1 put two-channel vector fits behind x and y polarisers by default.
    model, _, _ = two_channel_beads(optics, n=1)
    assert [ch.psf.analyser for ch in model.channels] == ["unpolarised", "unpolarised"]
    xyz = stack([-300.0, 0.0, 400.0], x=20.0)
    whole, p = make_psf(optics, P=32)
    x, _ = make_psf(optics, P=32, analyser="x")
    y, _ = make_psf(optics, P=32, analyser="y")
    p["pupil.phase"][1] = 0.4
    # Each analyser channel holds one photon in focus, like any channel; weighted by the light each
    # polariser passes, the two add up to the unpolarised PSF.
    fx, fy = x.energy / whole.energy, y.energy / whole.energy
    assert float(fx + fy) == pytest.approx(1.0, abs=1e-12)
    assert torch.allclose(fx * x(p, xyz) + fy * y(p, xyz), whole(p, xyz), rtol=1e-10, atol=1e-15)
    with pytest.raises(ValueError):
        make_psf(optics, P=32, analyser="z")


def test_a_one_channel_model_has_the_m3_parameters_and_a_two_channel_one_adds_the_secondarys(optics):
    model, truth, _ = two_channel_beads(optics, n=1)
    names = set(truth.specs)
    assert {"pupil.phase", "psf.sigma", "ch1.pupil.phase", "ch1.psf.sigma", "ch1.transform", "ch1.share",
            "ch1.dz"} <= names
    assert not any(n.startswith("ch0.") for n in names)
    assert np.allclose(unpack_transform(truth.physical()["ch1.transform"]).numpy(), SPLIT_TRANSFORM)
