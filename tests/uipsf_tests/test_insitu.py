"""In situ: emitters in the sample, a focus above the coverslip, and the sign of the even aberrations (M5)."""
import numpy as np
import pytest
import torch

import uipsf
from uipsf.model import Bleaching, InSituGeometry
from uipsf.noise import Poisson
from uipsf.optics.zernike import noll_to_nm
from uipsf.simulate import parameters, simulate, template
from uipsf.solve import Problem
from uipsf.workflows.insitu import TWIN_DECISIVE_NATS, learn_insitu, localise_from_heights, twin

f64 = torch.float64


def insitu_emitters(optics, n=12, focus=600.0, true_focus=None, P=24, roi=(11, 11), seed=0, photons=3000.0,
                    phase=(0.1, 0.7, -0.2, 0.15, 0.1), kind="vector"):
    """(model, truth Params, Batch): ``n`` emitters at heights 0–1000 nm, one frame each, seen with the focus
    at ``true_focus`` (default ``focus``, the model's start); astigmatism, coma and spherical (Noll 5–9)."""
    grid = uipsf.PupilGrid.build(optics, P, device="cpu", dtype=f64)
    pupil = uipsf.ZernikePupil(grid, n_max=4)
    psf = uipsf.VectorPSF(grid, pupil, roi, sigma_nm=(30.0, 30.0), kind=kind)
    model = uipsf.ForwardModel(psf, geometry=InSituGeometry(focus), intensity=Bleaching(order=0))
    g = torch.Generator().manual_seed(seed)
    xy = (torch.rand(n, 2, generator=g, dtype=f64) - 0.5) * 150
    h = torch.rand(n, generator=g, dtype=f64) * 1000.0
    coefficients = torch.zeros(len(pupil.phase_modes), dtype=f64)
    coefficients[:len(phase)] = torch.tensor(phase, dtype=f64)
    values = {"pupil.phase": coefficients, "psf.sigma": torch.tensor([35.0, 35.0]),
              "geometry.focus": torch.tensor(focus if true_focus is None else true_focus),
              "emitter.xyz": torch.cat([xy, h[:, None]], 1),
              "emitter.photons": photons * (1 + torch.rand(n, generator=g, dtype=f64)),
              "emitter.background": torch.full((n,), 10.0, dtype=f64)}
    b0 = template(n, torch.zeros(1, dtype=f64), roi, dtype=f64)
    truth = parameters(model, b0, values, dtype=f64)
    return model, truth, simulate(model, truth, b0, seed=seed + 1)


def test_the_focus_moves_the_defocus_and_the_height_stays_in_the_sample():
    batch = template(2, torch.zeros(1, dtype=f64), (5, 5), dtype=f64)
    geo = InSituGeometry(focus_nm=800.0)
    p = uipsf.Params(geo.specs(batch, {"xyz": torch.tensor([[1.0, 2.0, 300.0], [0.0, 0.0, 900.0]])}), 2,
                     dtype=f64).physical()
    pos = geo(p, batch)
    assert pos.shape == (2, 1, 4)
    assert pos[0, 0].tolist() == [1.0, 2.0, -800.0, 300.0]
    assert pos[1, 0, 2:].tolist() == [-800.0, 900.0]


def test_the_focus_has_a_loose_prior_and_is_fitted_unless_fixed():
    batch = template(1, torch.zeros(1), (5, 5))
    spec = {s.name: s for s in InSituGeometry(500.0).specs(batch, {"xyz": torch.zeros(1, 3)})}["geometry.focus"]
    assert not spec.fixed and spec.prior.mean == 500.0 and spec.prior.sigma == 500.0
    fixed = {s.name: s for s in InSituGeometry(500.0, fit_focus=False).specs(batch, {"xyz": torch.zeros(1, 3)})}
    assert fixed["geometry.focus"].fixed


def test_with_matched_indices_an_emitter_in_the_sample_is_a_bead_at_its_height_above_the_focus():
    optics = uipsf.Optics(NA=1.2, wavelength_nm=680, n_imm=1.33, n_med=1.33, pixel_size_nm=(110, 110))
    model, truth, batch = insitu_emitters(optics, n=4, focus=400.0)
    p = truth.physical()
    xyz = p["emitter.xyz"].clone()
    xyz[:, 2] -= p["geometry.focus"]
    bead = model.psf(p, xyz[:, None, :])
    insitu = model.psf(p, model.geometry(p, batch)[..., :3], model.geometry(p, batch)[..., 3])
    assert torch.allclose(insitu, bead, rtol=1e-9, atol=1e-12)


def test_the_emitter_linearisation_gives_the_jvps_normal_equations_in_situ(optics):
    model, truth, batch = insitu_emitters(optics, n=3, P=16, roi=(7, 7))
    p = truth.replace(shared=truth.shared_vector() * 1.02 + 0.01, emitter=truth.emitter_matrix() * 1.02 + 1e-3)
    a = Problem(model, Poisson(), batch, linearise=False).normal_equations(p)
    b = Problem(model, Poisson(), batch).normal_equations(p)
    for k in ("H_ss", "H_se", "H_ee", "g_s", "g_e"):
        assert torch.allclose(getattr(b, k), getattr(a, k), rtol=1e-10, atol=1e-8), k


def test_the_twin_flips_astigmatism_and_spherical_but_not_coma_or_trefoil(optics):
    model, truth, _ = insitu_emitters(optics, n=1)
    pupil = model.psf.pupil
    shared = {"pupil.phase": torch.arange(1.0, len(pupil.phase_modes) + 1, dtype=f64),
              "pupil.magnitude": torch.arange(1.0, len(pupil.magnitude_modes) + 1, dtype=f64)}
    t = twin(model, shared)
    for j, before, after in zip(pupil.phase_modes, shared["pupil.phase"], t["pupil.phase"], strict=True):
        assert float(after) == (-float(before) if noll_to_nm(j)[1] % 2 == 0 else float(before)), j
    for j, before, after in zip(pupil.magnitude_modes, shared["pupil.magnitude"], t["pupil.magnitude"],
                                strict=True):
        assert float(after) == (float(before) if noll_to_nm(j)[1] % 2 == 0 else -float(before)), j


def test_with_matched_indices_the_twin_images_the_mirrored_height_exactly():
    # Scalar light, matched indices, no supercritical angles: A*(−k) at −z is A at z, pixel for pixel.
    optics = uipsf.Optics(NA=1.2, wavelength_nm=680, n_imm=1.33, n_med=1.33, pixel_size_nm=(110, 110))
    model, truth, batch = insitu_emitters(optics, n=1, kind="scalar")
    p = truth.physical()
    p["pupil.magnitude"] = torch.linspace(-0.1, 0.1, len(model.psf.pupil.magnitude_modes), dtype=f64)
    q = dict(p, **{k: v for k, v in twin(model, p).items() if k.startswith("pupil.")})
    for dz in (-400.0, 250.0):
        xyz = torch.tensor([[[0.0, 0.0, dz]]], dtype=f64)
        assert torch.allclose(model.psf(q, -xyz), model.psf(p, xyz), rtol=1e-9, atol=1e-12)


def test_with_mismatched_indices_the_twin_does_not_image_the_mirrored_height(optics):
    model, truth, batch = insitu_emitters(optics, n=1, kind="scalar")
    p = truth.physical()
    q = dict(p, **{k: v for k, v in twin(model, p).items() if k.startswith("pupil.")})
    pos = model.geometry(p, batch)
    h_focus = float(p["geometry.focus"]) * optics.n_med / optics.n_imm
    mirrored = pos.clone()
    mirrored[..., 3] = 2 * h_focus - pos[..., 3]
    a, b = model.psf(p, pos[..., :3], pos[..., 3]), model.psf(q, mirrored[..., :3], mirrored[..., 3])
    assert float((a - b).abs().max()) > 0.01 * float(a.max())


def test_every_emitter_is_found_on_the_true_pupil_from_the_heights_tried(optics):
    model, truth, batch = insitu_emitters(optics, n=8, P=16, roi=(9, 9), photons=4000.0)
    shared = {n: v for n, v in truth.physical().items() if truth.specs[n].scope == "shared"}
    found, objectives = localise_from_heights(model, shared, batch, Poisson(), np.arange(0.0, 1200.0, 150.0))
    h = truth.physical()["emitter.xyz"][:, 2]
    assert objectives.shape == (8,)
    assert float((found["xyz"][:, 2] - h).abs().max()) < 60.0


@pytest.mark.slow
def test_the_pupil_and_focus_are_learnt_from_emitters_in_the_sample_and_the_twin_test_agrees(optics):
    # The start: the right sign of astigmatism at half its size, the focus 120 nm off.  ~10 s on 4 cores.
    model, truth, batch = insitu_emitters(optics, n=40, focus=600.0, true_focus=720.0)
    phase = torch.zeros(len(model.psf.pupil.phase_modes), dtype=f64)
    phase[1] = 0.4
    result = learn_insitu(model, batch, start={"pupil.phase": phase})
    fit, sd = result.calibration.result, result.calibration.result.std
    shared = result.calibration.shared()
    true = truth.physical()
    assert len(result.calibration.beads.used) >= 36
    assert torch.all((shared["pupil.phase"] - true["pupil.phase"]).abs() < 4 * sd["pupil.phase"])
    assert abs(float(shared["geometry.focus"]) - 720.0) < 3 * float(sd["geometry.focus"])
    used = torch.tensor(result.calibration.beads.used)
    dh = fit.params.physical()["emitter.xyz"][:, 2] - true["emitter.xyz"][used, 2]
    # Focus and heights move together (the focal shift n_med/n_imm): the heights relative to each other are
    # what the data fix.
    assert float((dh - dh.mean()).std()) < 40.0
    assert result.twin.verdict == "agree" and result.twin.delta_nats > TWIN_DECISIVE_NATS
    assert result.warnings == []


@pytest.mark.slow
def test_a_start_of_the_wrong_sign_learns_the_twin_and_the_twin_test_warns(optics):
    # Astigmatism started with the wrong sign: the fit settles in the twin (heights mirrored about the focus,
    # spherical bent to make up for the mismatch), and only the index mismatch tells the two apart.
    model, truth, batch = insitu_emitters(optics, n=40, focus=600.0, true_focus=720.0)
    phase = torch.zeros(len(model.psf.pupil.phase_modes), dtype=f64)
    phase[1] = -0.4
    result = learn_insitu(model, batch, start={"pupil.phase": phase})
    assert float(result.calibration.shared()["pupil.phase"][1]) < -0.5          # the sign it was given
    assert result.twin.verdict == "prefer twin"
    assert len(result.warnings) == 1 and "mirrored" in result.warnings[0]
