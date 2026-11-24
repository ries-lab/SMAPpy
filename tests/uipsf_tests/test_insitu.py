"""In situ: emitters in the sample, a focus above the coverslip, and the sign of the even aberrations (M5)."""
from dataclasses import replace

import numpy as np
import pytest
import torch

import uipsf
from uipsf.model import Bleaching, InSituGeometry, PerObservation
from uipsf.noise import Poisson
from uipsf.optics.zernike import noll_to_nm
from uipsf.simulate import parameters, simulate, template
from uipsf.solve import Problem
from uipsf.workflows.insitu import TWIN_DECISIVE_NATS, learn_insitu, localise_from_heights, twin
from uipsf.workflows.scoring import localise

f64 = torch.float64


def insitu_emitters(optics, n=12, focus=600.0, true_focus=None, P=24, roi=(11, 11), seed=0, photons=3000.0,
                    phase=(0.1, 0.7, -0.2, 0.15, 0.1), kind="vector", frames=1, frame_factors=None,
                    heights=(0.0, 1000.0), truth_field=None, chip=1000.0):
    """(model, truth Params, Batch): ``n`` emitters at heights in ``heights`` (nm, uniform), seen with the
    focus at ``true_focus`` (default ``focus``, the model's start); astigmatism, coma and spherical (Noll 5–9).

    With ``frames`` > 1 each emitter is a linked blink: one position in all its frames, photons per frame
    (`PerObservation`), its count times ``frame_factors`` [frames] (default 1 in every frame).

    ``truth_field`` ({(Noll, i, j): rad}): a field-dependent pupil (`FieldPolynomial` about the middle of a
    ``chip`` px square), the emitters spread over the chip."""
    from uipsf.model import Channel, FieldPolynomial
    grid = uipsf.PupilGrid.build(optics, P, device="cpu", dtype=f64)
    pupil = uipsf.ZernikePupil(grid, n_max=4)
    psf = uipsf.VectorPSF(grid, pupil, roi, sigma_nm=(30.0, 30.0), kind=kind)
    intensity = Bleaching(order=0) if frames == 1 else PerObservation()
    field = None if truth_field is None else FieldPolynomial((chip / 2, chip / 2), optics.pixel_size_nm)
    model = uipsf.ForwardModel([Channel(psf, field=field)], geometry=InSituGeometry(focus), intensity=intensity)
    g = torch.Generator().manual_seed(seed)
    xy = (torch.rand(n, 2, generator=g, dtype=f64) - 0.5) * 150
    h = heights[0] + torch.rand(n, generator=g, dtype=f64) * (heights[1] - heights[0])
    coefficients = torch.zeros(len(pupil.phase_modes), dtype=f64)
    coefficients[:len(phase)] = torch.tensor(phase, dtype=f64)
    N = photons * (1 + torch.rand(n, generator=g, dtype=f64))
    values = {"pupil.phase": coefficients, "psf.sigma": torch.tensor([35.0, 35.0]),
              "geometry.focus": torch.tensor(focus if true_focus is None else true_focus),
              "emitter.xyz": torch.cat([xy, h[:, None]], 1), "emitter.background": torch.full((n,), 10.0, dtype=f64)}
    if frames == 1:
        values["emitter.photons"] = N
    else:
        factors = torch.ones(frames, dtype=f64) if frame_factors is None else torch.as_tensor(frame_factors, dtype=f64)
        values["emitter.photons_per_slice"] = N[:, None] * factors
    b0 = template(n, torch.zeros(frames, dtype=f64), roi, dtype=f64)
    if field is not None:
        values["field.phase"] = torch.tensor([truth_field.get(t, 0.0) for t in field.layout], dtype=f64)
        b0 = replace(b0, field_xy=torch.rand(n, 2, generator=g, dtype=f64) * chip)
    truth = parameters(model, b0, values, dtype=f64)
    return model, truth, simulate(model, truth, b0, seed=seed + 1)


def unlinked(model, batch):
    """The same frames, each its own emitter: (a one-frame model, the batch [E·K, 1, 1, Y, X])."""
    E, C, K, Y, X = batch.data.shape
    single = uipsf.ForwardModel(model.psf, geometry=model.geometry, intensity=Bleaching(order=0))
    return single, uipsf.Batch(data=batch.data.transpose(1, 2).reshape(E * K, C, 1, Y, X),
                               z_nm=torch.zeros(1, dtype=f64))


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


# A learning problem small enough for a few seconds on one core: 16 molecules, a pupil of 16 samples, 9-pixel
# regions, started from heights 300 nm apart.  It keeps the verdicts of 40 molecules at 24 samples.
SMALL = dict(n=16, P=16, roi=(9, 9), photons=4000.0, focus=600.0, true_focus=720.0)
HEIGHTS = np.arange(0.0, 1300.0, 300.0)


@pytest.mark.slow
def test_the_pupil_and_focus_are_learnt_from_emitters_in_the_sample_and_the_twin_test_agrees(optics):
    # The start: the right sign of astigmatism at half its size, the focus 120 nm off.  ~4 s on one core.
    model, truth, batch = insitu_emitters(optics, **SMALL)
    phase = torch.zeros(len(model.psf.pupil.phase_modes), dtype=f64)
    phase[1] = 0.4
    result = learn_insitu(model, batch, start={"pupil.phase": phase}, heights=HEIGHTS)
    fit, sd = result.calibration.result, result.calibration.result.std
    shared = result.calibration.shared()
    true = truth.physical()
    assert len(result.calibration.beads.used) >= 14
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
    model, truth, batch = insitu_emitters(optics, **SMALL)
    phase = torch.zeros(len(model.psf.pupil.phase_modes), dtype=f64)
    phase[1] = -0.4
    result = learn_insitu(model, batch, start={"pupil.phase": phase}, heights=HEIGHTS)
    assert float(result.calibration.shared()["pupil.phase"][1]) < -0.5          # the sign it was given
    assert result.twin.verdict == "prefer twin"
    assert len(result.warnings) == 1 and "mirrored" in result.warnings[0]


def test_a_linked_blink_has_one_position_in_all_its_frames_and_photons_per_frame(optics):
    model, truth, batch = insitu_emitters(optics, n=2, P=16, roi=(7, 7), frames=3, frame_factors=(1.0, 0.5, 2.0))
    p = truth.physical()
    pos = model.geometry(p, batch)
    assert pos.shape == (2, 3, 4) and torch.equal(pos[:, 0], pos[:, 2])
    N = model.intensity(p, batch)
    assert torch.allclose(N[:, 1] / N[:, 0], torch.full((2,), 0.5, dtype=f64))


def _shared(truth):
    return {n: v for n, v in truth.physical().items() if truth.specs[n].scope == "shared"}


@pytest.mark.slow
def test_linking_four_frames_halves_the_crlb_of_the_height(optics):
    # Four frames of the same photons: four times the information, so the height's CRLB is halved.  Both
    # localised on the true pupil and focus, from the true heights' neighbourhood.
    model, truth, batch = insitu_emitters(optics, n=4, P=16, roi=(9, 9), frames=4, photons=1500.0)
    shared = _shared(truth)
    heights = np.arange(0.0, 1100.0, 150.0)
    linked, _ = localise_from_heights(model, shared, batch, Poisson(), heights)
    fit_l, _ = localise(model, shared, replace(batch, start=linked), Poisson())
    single, frames = unlinked(model, batch)
    alone, _ = localise_from_heights(single, shared, frames, Poisson(), heights)
    fit_s, _ = localise(single, shared, replace(frames, start=alone), Poisson())
    crlb_l = fit_l.std["emitter.xyz"][:, 2]
    crlb_s = fit_s.std["emitter.xyz"][:, 2].reshape(4, 4).mean(1)
    assert torch.allclose(crlb_l / crlb_s, torch.full((4,), 0.5, dtype=f64), atol=0.05)


def test_the_photons_of_each_frame_of_a_blink_come_back(optics):
    model, truth, batch = insitu_emitters(optics, n=6, P=16, roi=(9, 9), frames=3, photons=3000.0,
                                          frame_factors=(1.0, 0.4, 1.6))
    shared = _shared(truth)
    # starts every 300 nm find the same optimum as every 150 nm, in a third of the time
    found, _ = localise_from_heights(model, shared, batch, Poisson(), np.arange(0.0, 1100.0, 300.0))
    fit, _ = localise(model, shared, replace(batch, start=found), Poisson())
    N, sd = fit.params.physical()["emitter.photons_per_slice"], fit.std["emitter.photons_per_slice"]
    assert torch.all((N - truth.physical()["emitter.photons_per_slice"]).abs() < 4 * sd)


@pytest.mark.slow
def test_the_pupil_is_learnt_from_linked_blinks_with_photons_per_frame(optics):
    model, truth, batch = insitu_emitters(optics, **dict(SMALL, n=8), frames=3, frame_factors=(1.0, 0.6, 1.3))
    phase = torch.zeros(len(model.psf.pupil.phase_modes), dtype=f64)
    phase[1] = 0.4
    result = learn_insitu(model, batch, start={"pupil.phase": phase}, heights=HEIGHTS)
    sd = result.calibration.result.std["pupil.phase"]
    assert len(result.calibration.beads.used) >= 7
    assert torch.all((result.calibration.shared()["pupil.phase"] - truth.physical()["pupil.phase"]).abs() < 4 * sd)
    assert result.twin.verdict == "agree"


def test_an_emitters_widths_start_it_near_its_height_on_the_true_model(optics):
    from smappy.calibrate.insitu import gaussian

    from uipsf.workflows.insitu import width_starts
    model, truth, batch = insitu_emitters(optics, n=12, P=16, roi=(15, 15), photons=4000.0)
    g = gaussian(batch.data[:, 0, 0].numpy().astype(np.float32))
    starts = width_starts(model, _shared(truth), np.c_[g["sigma_x_pix"] ** 2, g["sigma_y_pix"] ** 2], 15)
    h = truth.physical()["emitter.xyz"][:, 2].numpy()
    nearest = np.nanmin(np.abs(starts - h[:, None]), axis=1)
    assert starts.shape == (12, 3) and np.mean(nearest < 150) >= 0.9


def test_starting_every_emitter_at_the_same_heights_is_localising_from_those_heights(optics):
    from uipsf.workflows.insitu import localise_from_starts
    model, truth, batch = insitu_emitters(optics, n=4, P=16, roi=(9, 9))
    heights = np.array([200.0, 600.0])
    a, oa = localise_from_heights(model, _shared(truth), batch, Poisson(), heights)
    b, ob = localise_from_starts(model, _shared(truth), batch, Poisson(), np.tile(heights, (4, 1)))
    assert torch.allclose(a["xyz"], b["xyz"]) and torch.allclose(oa, ob)


def test_the_compiled_kernels_shapes_are_powers_of_two():
    from uipsf.model.psf import _bucket, _pad_dim0
    assert [_bucket(n) for n in (1, 3, 4, 5, 300, 513, 1284)] == [1, 4, 4, 8, 512, 1024, 1536]
    padded = _pad_dim0(torch.ones(3, 2), 4)
    assert padded.shape == (4, 2) and float(padded[3].abs().sum()) == 0.0


# ------------------------------------------------------------------ field dependence (field-dependence.md §12.3)
def test_the_twin_of_a_field_flips_its_astigmatism_terms_and_keeps_its_coma_terms(optics):
    model, truth, _ = insitu_emitters(optics, n=1, truth_field={(6, 1, 0): 0.3})
    field = model.channels[0].field
    shared = dict(truth.physical(), **{"field.phase": torch.arange(1.0, len(field.layout) + 1, dtype=f64)})
    t = twin(model, shared)
    for (j, _, _), before, after in zip(field.layout, shared["field.phase"], t["field.phase"], strict=True):
        assert float(after) == (-float(before) if j in (5, 6) else float(before)), j


def test_the_field_twin_mirrors_only_the_fields_astigmatism_and_keeps_the_pupil(optics):
    from uipsf.workflows.insitu import field_twin
    model, truth, _ = insitu_emitters(optics, n=1, truth_field={(6, 1, 0): 0.3})
    field = model.channels[0].field
    shared = dict(truth.physical(), **{"field.phase": torch.arange(1.0, len(field.layout) + 1, dtype=f64)})
    t = field_twin(model, shared)
    assert torch.equal(t["pupil.phase"], shared["pupil.phase"])
    for (j, _, _), before, after in zip(field.layout, shared["field.phase"], t["field.phase"], strict=True):
        assert float(after) == (-float(before) if j in (5, 6) else float(before)), j


def test_with_matched_indices_the_twin_of_a_field_images_the_mirrored_height_at_every_place():
    optics = uipsf.Optics(NA=1.2, wavelength_nm=680, n_imm=1.33, n_med=1.33, pixel_size_nm=(110, 110))
    model, truth, batch = insitu_emitters(optics, n=3, kind="scalar",
                                          truth_field={(6, 1, 0): 0.4, (5, 1, 1): 0.3, (8, 0, 1): 0.2})
    p = truth.physical()
    q = dict(p, **{k: v for k, v in twin(model, p).items() if k.startswith(("pupil.", "field."))})
    pa, pb = model.channel_params(p, batch, 0), model.channel_params(q, batch, 0)
    for dz in (-400.0, 250.0):
        xyz = torch.full((3, 1, 3), 0.0, dtype=f64)
        xyz[..., 2] = dz
        assert torch.allclose(model.psf(pb, -xyz), model.psf(pa, xyz), rtol=1e-9, atol=1e-12)


@pytest.mark.long          # ~30 s: a learning study and the field's mirror fitted
def test_molecules_in_the_sample_spread_over_the_chip_give_back_the_field(optics):
    """In situ with *field dependence*: 160 emitters over a 1000 px chip (127 × 116 µm), astigmatism and coma
    changing over it, learnt from a start of the right sign without the field: the field comes back at the
    chip's corners, the twin test still agrees, and the report calls the field significant."""
    from uipsf.diagnostics import field_report
    truth_field = {(6, 1, 0): 0.25, (5, 0, 1): 0.2, (6, 0, 2): 0.1, (8, 1, 0): 0.12}
    model, truth, batch = insitu_emitters(optics, n=160, P=16, roi=(9, 9), photons=5000.0, focus=600.0,
                                          true_focus=650.0, truth_field=truth_field)
    phase = torch.zeros(len(model.psf.pupil.phase_modes), dtype=f64)
    phase[1] = 0.4
    result = learn_insitu(model, batch, start={"pupil.phase": phase}, heights=HEIGHTS)
    field = model.channels[0].field
    fit = result.calibration.result
    learnt = fit.params.physical()["field.phase"].detach().double()
    true = truth.physical()["field.phase"]
    lo, hi = next((a, b) for n, a, b in fit.params.layout("shared") if n == "field.phase")
    cov = fit.covariance_shared[lo:hi, lo:hi].detach().double()
    # each mode's change at the chip's corners and edges against its own CRLB there: 160 molecules pin a
    # corner's astigmatism to about 0.02 rad, a third of what the field does there
    corners = torch.tensor([[x, y] for x in (0.0, 500.0, 1000.0) for y in (0.0, 500.0, 1000.0)], dtype=f64)
    for j, _ in field.terms:
        rows = [i for i, (m, _, _) in enumerate(field.layout) if m == j]
        g = field.basis(corners)[:, rows]
        error = g @ (learnt - true)[rows]
        sd = torch.sqrt(torch.einsum("ei,ij,ej->e", g, cov[rows][:, rows], g))
        assert torch.all(error.abs() < 4 * sd + 0.005) and float(sd.max()) < 0.04, (j, error, sd)
    assert result.twin.verdict == "agree"
    assert field_report(result.calibration).significant
    # the field's own mirror (the centre kept, its astigmatism flipped), which the rounds try after round 1:
    # the data refuse it here, where the field was learnt with the right sign
    from uipsf.workflows.insitu import TWIN_DECISIVE_NATS, field_twin, twin_fit
    flip = twin_fit(model, result.calibration, replace(batch, start=None), Poisson(), HEIGHTS, mirror=field_twin)
    assert flip.delta_nats > TWIN_DECISIVE_NATS
