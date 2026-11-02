"""One PSF model: one description of the choices (`PSFModelSettings`), one builder (`model.build`), one point-emitter
view (`VectorPSF.for_emitters`) and one hand-over between models (`VectorPSF.describe`, `held_from`)."""
import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from smappy.calibrate.pupil import BeadSettings, PSFModelSettings, model_shown

import uipsf
from uipsf.export import sample_psf
from uipsf.model import (
    Held,
    InSituGeometry,
    MinimalPupil,
    ZernikePupil,
    held_from,
    make_psf,
    make_pupil,
    pupil_grid,
    start_from,
)
from uipsf.model.fourpi import FourPiModel
from uipsf.optics.zernike import noll_to_nm
from uipsf.workflows import bead_model, insitu_model

f64 = torch.float64
# below the NA: a supercritical ring, damped by the beads' h
OPTICS = uipsf.Optics(NA=1.5, wavelength_nm=600, n_imm=1.518, n_med=1.33, pixel_size_nm=(105.0, 105.0))
SPEC = PSFModelSettings(zernike_order=4)
GAMMA, RING_H, SIGMA = 0.35, 80.0, 20.0


def shared_of(psf, phase=(0.4, -0.2, 0.15, 0.1)):
    p = {"pupil.phase": torch.zeros(len(psf.pupil.phase_modes), dtype=f64),
         "psf.sigma": torch.tensor([SIGMA], dtype=f64),
         "pupil.magnitude": torch.zeros(len(psf.pupil.magnitude_modes), dtype=f64)}
    p["pupil.phase"][:len(phase)] = torch.tensor(phase, dtype=f64)
    if isinstance(psf.pupil, MinimalPupil):
        p["pupil.gamma"] = torch.tensor(GAMMA, dtype=f64)
        if psf.pupil.ring:
            p["pupil.ring_h"] = torch.tensor(RING_H, dtype=f64)
    return p


def test_the_builder_puts_a_minimal_pupil_on_a_grid_beyond_the_na_and_a_zernike_one_at_it():
    minimal = make_pupil(OPTICS, SPEC, device="cpu")
    assert isinstance(minimal, MinimalPupil) and minimal.na == 1.5 and minimal.grid_na == pytest.approx(1.51)
    zernike = make_pupil(OPTICS, replace(SPEC, magnitude="symmetric"), device="cpu")
    assert isinstance(zernike, ZernikePupil) and zernike.grid.optics.NA == 1.5 and zernike.magnitude_modes == [4, 11]
    assert pupil_grid(OPTICS, SPEC, size=32).size == 32


def test_the_builder_owns_the_blur_the_pixel_blur_and_the_saf_damping():
    psf = make_psf(OPTICS, SPEC, (13, 13), device="cpu")
    assert psf.round_sigma and psf.fit_sigma and psf.pixel_blur and psf.pupil.ring
    held = make_psf(OPTICS, replace(SPEC, blur="fixed", blur_nm=30.0, pixel_blur=False, saf_damping=False), (13, 13),
                    device="cpu")
    assert not held.fit_sigma and held.sigma_init == (30.0, 30.0) and not held.pixel_blur and not held.pupil.ring


def test_a_minimal_pupil_refuses_held_zernike_magnitude_and_a_zernike_one_a_held_gamma():
    with pytest.raises(ValueError, match="γ"):
        make_pupil(OPTICS, replace(SPEC, magnitude="symmetric"), Held(gamma=0.3), device="cpu")
    with pytest.raises(ValueError, match="magnitude"):
        make_pupil(OPTICS, SPEC, Held(magnitude={4: 0.1}), device="cpu")


def test_a_held_gamma_and_h_are_no_parameters_and_draw_the_same_pupil_as_learnt_ones():
    learnt = make_pupil(OPTICS, SPEC, device="cpu", dtype=f64)
    held = MinimalPupil(learnt.grid, n_max=4, na=1.5, held_gamma=GAMMA, held_ring_h=RING_H)
    assert {s.name for s in held.specs()} == {"pupil.phase", "pupil.magnitude"}
    p = {"pupil.phase": torch.zeros(len(learnt.phase_modes), dtype=f64), "pupil.magnitude": torch.zeros(0, dtype=f64)}
    a, _ = learnt(dict(p, **{"pupil.gamma": torch.tensor(GAMMA, dtype=f64), "pupil.ring_h": torch.tensor(RING_H)}))
    b, _ = held(p)
    assert torch.allclose(a, b) and held.values(p) == pytest.approx({"gamma": GAMMA, "ring_h_nm": RING_H})
    assert held.truncated(2).held_gamma == GAMMA


def test_the_emitter_view_drops_the_bead_and_the_ring_damping_and_keeps_the_rest():
    beads = bead_model(OPTICS, BeadSettings(psf=SPEC, bead_diameter_nm=100.0), (13, 13), z_max_nm=1000,
                       device="cpu", dtype=f64)
    base = beads.psf
    view = base.for_emitters(17, upsample=3)
    assert view.roi_shape == (17, 17) and view.upsample == 3 and view.bead_radius_nm == 0.0
    assert view.pixel_blur and view.round_sigma and (view.kind, view.analyser) == (base.kind, base.analyser)
    assert base.pupil.ring and not view.pupil.ring                    # the ring at h = 0: undamped
    p = shared_of(base)
    xyz = torch.tensor([[[10.0, -20.0, 150.0]]], dtype=f64)
    point = base.resized(17, upsample=3, bead_radius_nm=0.0)(dict(p, **{"pupil.ring_h": torch.tensor(0.0, dtype=f64)}),
                                                             xyz)
    assert torch.allclose(view(p, xyz), point, rtol=1e-10, atol=1e-14)


def test_a_movie_draws_its_molecules_with_the_learnt_models_point_emitter_psf():
    # Would have caught `movie.render`'s own rebuild, which dropped the pixel's blur and kept the beads' ring
    # damping: a spot cut from the movie is `sample_psf`'s, the one view of a learnt model for molecules.
    from uipsf.movie import MovieTruth, render
    psf = make_psf(OPTICS, SPEC, (15, 15), device="cpu", dtype=f64)
    model = uipsf.ForwardModel(psf, geometry=InSituGeometry(500.0), intensity=uipsf.model.Bleaching(order=0))
    shared = dict(shared_of(psf), **{"geometry.focus": torch.tensor(500.0, dtype=f64)})
    h = 300.0
    px = OPTICS.pixel_size_nm[0]
    truth = MovieTruth(np.array([0]), np.array([0]), np.array([0]), np.array([20 * px]), np.array([20 * px]),
                       np.array([h]), np.array([1.0]))
    image = np.zeros((1, 41, 41))
    render(model, shared, truth, image, 15)
    expected = sample_psf(model, shared, [[0.0, 0.0, -500.0]], 15, depth_nm=h)[0].double().numpy()
    assert np.allclose(image[0, 13:28, 13:28], expected, rtol=1e-6, atol=1e-12)
    # and that is not what the beads' model draws: the pixel's blur and the ring's damping both matter
    sharp = replace(SPEC, pixel_blur=False)
    without = make_psf(OPTICS, sharp, (15, 15), device="cpu", dtype=f64)
    damped = without(shared, torch.tensor([[[0.0, 0.0, -500.0]]], dtype=f64), torch.tensor([h], dtype=f64))[0, 0]
    assert np.abs(damped.numpy() - expected).max() > 0.01 * expected.max()


def test_the_bead_scoring_spline_is_the_beads_model_with_its_pixel_blur():
    # Would have caught `SplineFrames.calibration`'s rebuild, which dropped the pixel's blur: its voxels are the
    # model's own forward pass of a bead stack, centre on centre.
    pytest.importorskip("smappy._fit3d")
    from uipsf.workflows.spline import SplineFrames
    model = bead_model(OPTICS, BeadSettings(psf=SPEC), (13, 13), z_max_nm=1000, device="cpu", dtype=f64)
    shared = shared_of(model.psf)
    frames = SplineFrames(dz_nm=50.0, z_margin_nm=0.0)
    cal = frames.calibration(model, shared, (13, 13), (-200.0, 200.0))
    n = cal.psf.shape[0]
    z = -(torch.arange(n, dtype=f64) - cal.z0) * 50.0
    xyz = torch.zeros(1, n, 3, dtype=f64)
    xyz[0, :, 2] = z
    direct = model.psf(shared, xyz, z_step_nm=50.0)[0].numpy()               # [n, 13, 13]
    m = frames.margin_px
    vol = cal.psf[:, m:m + 13, m:m + 13] * (direct.sum((1, 2)).max() / cal.psf[:, m:m + 13, m:m + 13].sum((1, 2)).max())
    centre = (slice(None), slice(4, 9), slice(4, 9))
    assert np.allclose(vol[centre], direct[centre], rtol=0, atol=2e-3 * direct.max())


def test_a_psf_describes_itself_for_the_hand_over():
    model = bead_model(OPTICS, BeadSettings(psf=SPEC), (13, 13), z_max_nm=1000, device="cpu", dtype=f64)
    d = model.psf.describe(shared_of(model.psf))
    assert d["kind"] == "minimal" and d["gamma"] == GAMMA and "ring_h_nm" not in d and "magnitude" not in d
    assert d["phase_nm"]["5"] == pytest.approx(0.4 * 600 / (2 * math.pi), abs=1e-3)
    assert d["sigma_nm"] == [SIGMA, SIGMA] and d["pixel_blur"] and d["NA"] == 1.5 and d["wavelength_nm"] == 600
    zernike = bead_model(OPTICS, BeadSettings(psf=replace(SPEC, magnitude="symmetric")), (13, 13), device="cpu")
    p = dict(shared_of(zernike.psf), **{"pupil.magnitude": torch.tensor([-0.1, 0.05])})
    assert zernike.psf.describe(p)["kind"] == "zernike"
    assert zernike.psf.describe(p)["magnitude"] == pytest.approx({"4": -0.1, "11": 0.05})


def test_a_description_is_held_and_started_from_by_the_next_model():
    beads = bead_model(OPTICS, BeadSettings(psf=replace(SPEC, zernike_order=6)), (13, 13), device="cpu", dtype=f64)
    phase = torch.linspace(0.3, -0.2, len(beads.psf.pupil.phase_modes))
    d = beads.psf.describe(dict(shared_of(beads.psf), **{"pupil.phase": phase.to(f64)}))
    # at another emission: the same wavefront in other radians
    held = held_from(d, 4, 1200.0, source="beads")
    above = {j for j in beads.psf.pupil.phase_modes if noll_to_nm(j)[0] > 4}               # Noll 16-28
    assert held.gamma == GAMMA and held.kind == "minimal" and set(held.phase) == above
    j = beads.psf.pupil.phase_modes.index(22)
    assert held.phase[22] == pytest.approx(float(phase[j]) / 2, abs=1e-4)
    situ = insitu_model(OPTICS, PSFModelSettings(zernike_order=4), (13, 13), 500.0, device="cpu", held=held)
    assert situ.psf.pupil.held_gamma == GAMMA and not situ.psf.pupil.ring          # in situ: no SAF damping
    start = start_from(situ.psf.pupil, d, 1200.0)
    assert set(start) == {"pupil.phase"}                                          # γ held: nothing to start
    learns = insitu_model(OPTICS, PSFModelSettings(zernike_order=4), (13, 13), 500.0, device="cpu")
    assert float(start_from(learns.psf.pupil, d, 600.0)["pupil.gamma"]) == pytest.approx(GAMMA)
    assert held_from(None, 4, 600.0) == Held() and not Held()


def test_each_channel_of_a_dual_calibration_learns_its_own_gamma():
    red = replace(OPTICS, wavelength_nm=680.0)
    model = bead_model([OPTICS, red], BeadSettings(psf=SPEC), (13, 13), device="cpu")
    main, secondary = (ch.psf.pupil for ch in model.channels)
    assert main is not secondary and main.held_gamma is None and secondary.held_gamma is None
    assert all("pupil.gamma" in {s.name for s in ch.psf.specs()} for ch in model.channels)


def test_twin_objectives_share_one_gamma_under_same_magnitude():
    grid = pupil_grid(OPTICS, SPEC, size=32)
    pupils = [make_pupil(OPTICS, SPEC, grid=grid) for _ in range(2)]
    shared = FourPiModel(grid, (13, 13), pupils=pupils, same_magnitude=True, pixel_blur=True)
    own = FourPiModel(grid, (13, 13), pupils=pupils)
    assert "pupil.gamma" in shared.psf_names() and "lo.pupil.gamma" not in shared.psf_names()
    assert {"pupil.gamma", "lo.pupil.gamma"} <= own.psf_names()
    assert shared.lower_name("pupil.gamma") == "pupil.gamma" and shared.psf_lo.pixel_blur


def test_the_form_shows_what_each_method_reads():
    beads = model_shown("beads", "minimal")
    assert beads["psf.magnitude"] and beads["psf.saf_damping"] and beads["drift"] and beads["field_dependence"]
    assert not beads["fit_tilt"] and model_shown("beads", layered=True)["fit_tilt"]
    assert not model_shown("beads", "free")["psf.saf_damping"]                # a minimal magnitude's only
    assert not model_shown("beads", "minimal", ring=False)["psf.saf_damping"]  # nothing beyond the critical angle
    assert not model_shown("beads", dual=True)["field_dependence"]
    situ = model_shown("insitu")
    assert situ["fit_tilt"] and not any(situ[k] for k in ("psf.magnitude", "psf.saf_damping", "bead_diameter_nm",
                                                           "drift", "field_dependence"))
