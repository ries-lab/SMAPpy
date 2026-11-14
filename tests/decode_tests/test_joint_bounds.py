"""The test set's joint bounds (`decode_smlm.training.test_crlbs`, `smappy.training.joint_crlb`)
on DECODE's own spline: the spots are where the renderer draws them, the derivatives are the
spline's, the single bound is the one the training reported before, a crowded molecule's joint
bound is above it and its neighbouring frames bring it down, and localizations scattered by the
joint bound read 1 against it in the Follow metrics."""
import json
import types

import numpy as np
import pytest
import torch
import yaml

from decode_smlm import prefit
from decode_smlm import training as T
from decode_smlm.generic.emitter import EmitterSet
from decode_smlm.neuralfitter.train.random_simulation import setup_random_simulation
from smappy.calibrate.core import spline_coefficients
from smappy.io.calibration import SplineCalibration
from smappy.training import job as jobs

NZ, BOX, FRAME = 41, 17, 40


def astigmatic() -> SplineCalibration:
    z = np.linspace(-1, 1, NZ)[:, None, None]
    y, x = np.mgrid[:BOX, :BOX] - (BOX - 1) / 2
    sx = 1.3 + 0.8 * (z - 0.4) ** 2
    sy = 1.3 + 0.8 * (z + 0.4) ** 2
    psf = np.exp(-x ** 2 / (2 * sx ** 2) - y ** 2 / (2 * sy ** 2)) / (2 * np.pi * sx * sy)
    return SplineCalibration(spline_coefficients(psf.astype(np.float32)), 50.0, NZ // 2, x0=(BOX - 1) / 2)


@pytest.fixture(scope="module")
def test_set(tmp_path_factory):
    """A test set of 24 frames of 40 x 40 pixels at the HD density, molecules on for about two
    frames, as a job's training draws it; with its parameters and PSF."""
    folder = tmp_path_factory.mktemp("job")
    est = prefit.Estimates(on_time_frames=2.0, lifetime_frames=2.0, intensity_mu_sig=(3000.0, 600.0),
                           intensity_from="fully-on frames", n_fully_on=500, background=(20.0, 60.0),
                           z_nm=(-400.0, 400.0), emitters_per_frame=25.0, n_blinks=100, n_localizations=300)
    params = prefit.method_params(est, prefit.camera(0.5, 100.0, 100.0), "HD", frame_px=FRAME)
    params["TestSet"] = {"test_size": 24}
    job = jobs.Job.new(method="decode-v0.10.2", ranges={"density": "HD"}, camera={}, calibration={}, seed=0,
                       device="cpu")
    folder = jobs.write(folder / "job", job, params, astigmatic())
    param = T.parameters(folder, yaml.safe_load((folder / "method.yaml").read_text()), "cpu", None, 0)
    T.seed_all(0)
    _, sim = setup_random_simulation(param)
    em = sim.em_sampler()
    em = em[(em.frame_ix >= 0) & (em.frame_ix < 24)]
    bg = torch.as_tensor(np.random.default_rng(0).uniform(20, 60, 24), dtype=torch.float32)
    target = bg[:, None, None].expand(24, FRAME, FRAME) / float(param.Scaling.bg_max)
    ds = types.SimpleNamespace(_emitter=em, _target=(None, None, target))
    return ds, param, T._psf(param), bg.numpy()


def test_the_spots_lie_where_the_renderer_draws_them_with_the_splines_derivatives(test_set):
    ds, param, psf, bg = test_set
    em = ds._emitter
    spots = T.test_spots(em, psf, bg)
    h, w = spots.model.shape[1:]
    canvas = np.zeros((24, FRAME + 2 * h, FRAME + 2 * w))
    for f, (r, c), m in zip(spots.frame, spots.corner, spots.model):
        canvas[f, r + h:r + 2 * h, c + w:c + 2 * w] += m
    drawn = psf.forward(em.xyz, em.phot, em.frame_ix, 0, 23).numpy()
    assert canvas[:, h:h + FRAME, w:w + FRAME] == pytest.approx(drawn, abs=1e-3 * float(drawn.max()))
    # the derivatives by x, y (per pixel), z (per nm) and photons, against differences of the patches
    xyz, phot = em.xyz[:5].double().float(), em.phot[:5].float()
    for k, step in enumerate((1e-2, 1e-2, 1.0)):
        e = torch.zeros(3)
        e[k] = step
        # each emitter moved by the step keeps its patch where it was only within its pixel: moved
        # back to the same corner, the difference is the derivative
        hi = psf.forward_rois(xyz + e, phot)
        lo = psf.forward_rois(xyz - e, phot)
        same = (psf.frame2roi_coord(xyz + e)[1] == psf.frame2roi_coord(xyz - e)[1]).all(1)
        numeric = ((hi - lo) / (2 * step)).numpy()[same.numpy()]
        assert np.abs(spots.grads[:5][same.numpy(), k]) == pytest.approx(np.abs(numeric), abs=2e-3 * phot.max().item())
    assert spots.grads[:5, 3] * phot[:5, None, None].numpy() == pytest.approx(spots.model[:5], rel=1e-4, abs=1e-3)


def test_the_single_bound_is_the_one_reported_before_and_crowding_raises_the_joint_one(test_set):
    ds, param, psf, bg = test_set
    b = T.test_crlbs(ds, param, psf)
    em = ds._emitter
    assert set(b) == {"index", "single", "joint", "context", "nn_nm", "frames_on"} and len(b["index"]) == len(em)
    before, _ = psf.crlb_sq(em.xyz.float(), em.phot.float(), torch.as_tensor(bg)[em.frame_ix].float())
    px = float(param.Camera.px_size[0])
    # the same matrix: up to half a percent apart, which is DECODE's float32 inverse's error
    assert b["single"][:, :2] == pytest.approx(before[:, :2].numpy() * px, rel=1e-2)
    assert b["single"][:, 2] == pytest.approx(before[:, 2].numpy(), rel=1e-2)
    ratio = b["joint"][:, 0] / b["single"][:, 0]
    alone = b["nn_nm"] > 1000
    inside = np.all((em.xyz[:, :2].numpy() > 9) & (em.xyz[:, :2].numpy() < FRAME - 10), axis=1)
    assert np.median(ratio[alone & inside]) == pytest.approx(1.0, abs=0.02)
    assert np.median(ratio[b["nn_nm"] < 300]) > 1.5
    # the frames read around it: never worse, and better for a molecule on in all three
    assert np.all(b["context"][:, :2] <= b["joint"][:, :2] * (1 + 1e-6))
    three = b["frames_on"] == 3
    assert three.sum() > 20
    assert np.median(b["context"][three, 0] / b["joint"][three, 0]) < 0.75
    # a pair on top of each other is bounded by about the prior, never infinitely
    assert np.all(np.isfinite(b["joint"])) and b["joint"][:, :2].max() < 1.5 * T.PRIOR_PX * px


def test_localizations_scattered_by_the_context_bound_read_one_against_it(test_set):
    ds, param, psf, _ = test_set
    b = T.test_crlbs(ds, param, psf)
    em = ds._emitter
    px = float(param.Camera.px_size[0])
    rng = np.random.default_rng(2)
    noise = rng.standard_normal((len(em), 3)) * b["context"] / np.array([px, px, 1.0])
    fit = EmitterSet(xyz=em.xyz + torch.as_tensor(noise, dtype=torch.float32), phot=em.phot, frame_ix=em.frame_ix,
                     xy_unit="px")
    m = T._crlb_and_pixel_metrics(None, fit, em, b, param.Camera.px_size)
    json.dumps(m, allow_nan=False)                       # a progress line as it is written
    assert m["crlb_width_lat_context"] == pytest.approx(1.0, abs=0.08)
    assert m["crlb_width_ax_context"] == pytest.approx(1.0, abs=0.12)
    # the frame alone allows less than its neighbours do: against its bound the same errors are smaller
    assert m["crlb_width_lat_joint"] < 0.7 and m["crlb_width_lat_joint"] < m["crlb_width_lat"]
    split = m["crlb_by_neighbour"]
    assert split["edges_nm"] == list(T.NEIGHBOUR_EDGES_NM) and sum(split["n"]) == len(em)
    near = split["single_lat"][0]
    assert near is not None and near > 1.2 and near > m["crlb_width_lat"]   # most off for close pairs
    by_frames = m["crlb_by_frames_on"]
    assert by_frames["frames"] == [1, 2, 3] and sum(by_frames["n"]) == len(em)
    # against the single frame's bound, a molecule on in three frames looks better than the bound
    assert by_frames["joint_lat"][2] < 0.85


def test_without_the_joint_bounds_only_the_single_one_is_computed_and_reported(test_set):
    ds, param, psf, _ = test_set
    b = T.test_crlbs(ds, param, psf, joint=False)
    assert set(b) == {"index", "single", "nn_nm", "frames_on"}
    assert b["single"] == pytest.approx(T.test_crlbs(ds, param, psf)["single"])
    em = ds._emitter
    m = T._crlb_and_pixel_metrics(None, em, em, b, param.Camera.px_size)
    assert "crlb_width_lat" in m and not any(k.endswith(("_joint", "_context")) for k in m)
    assert set(m["crlb_by_neighbour"]) == {"edges_nm", "n", "single_lat", "single_ax"}
