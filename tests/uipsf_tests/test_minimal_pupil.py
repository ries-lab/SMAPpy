"""The minimal pupil (cos θ^γ, the ring, the aperture), the pixel's blur under the learnt one and the stage's
linear drift: the bead calibration's defaults since November 2026."""
import math
from dataclasses import replace

import pytest
import torch

import uipsf
from uipsf.export.model_file import build, describe
from uipsf.model import MinimalPupil, ZernikePupil
from uipsf.optics import PupilGrid
from uipsf.simulate import parameters, simulate, template
from uipsf.workflows import BeadSettings, PSFModelSettings, bead_model, calibrate_beads, start_values

OPTICS = uipsf.Optics(NA=1.5, wavelength_nm=600, n_imm=1.518, n_med=1.33, pixel_size_nm=(105.0, 105.0))
MATCHED = replace(OPTICS, n_med=1.518)
PHASE = torch.tensor([0.5, -0.3, 0.2, 0.15])          # Noll 5–8: astigmatism and coma
GAMMA, RING_H, SIGMA = 0.35, 60.0, 20.0
ROI, K = (13, 13), 21


def simulated(model, n=10, extra=None, z_stack=None, seed=3):
    g = torch.Generator().manual_seed(1)
    phase = torch.zeros(len(model.psf.pupil.phase_modes))
    phase[:4] = PHASE
    values = {"pupil.phase": phase, "psf.sigma": torch.tensor([SIGMA]),
              "emitter.xyz": (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([150.0, 150.0, 200.0]),
              "emitter.photons": 20000 * (1 + torch.rand(n, generator=g)),
              "emitter.background": torch.full((n,), 20.0)} | (extra or {})
    b0 = template(n, uipsf.Batch.z_from_objective(torch.linspace(-1000, 1000, K)), ROI, device="cpu")
    if z_stack is not None:
        b0 = z_stack(b0)
    return simulate(model, parameters(model, b0, values, device="cpu"), b0, seed=seed)


def test_the_defaults_are_the_minimal_model_with_a_round_blur_over_the_pixel_and_a_linear_drift():
    s = BeadSettings()
    assert (s.psf.magnitude, s.psf.blur, s.psf.pixel_blur, s.drift, s.fit_tilt) == ("minimal", "round", True, "linear",
                                                                                     False)
    model = bead_model(OPTICS, s, ROI, z_max_nm=1000, device="cpu")
    pupil, psf = model.psf.pupil, model.psf
    assert isinstance(pupil, MinimalPupil) and pupil.magnitude_modes == []
    assert pupil.na == 1.5 and pupil.grid_na == pytest.approx(1.51)          # the grid reaches beyond the NA
    assert psf.pixel_blur and psf.pixel_sigma == pytest.approx((105 / math.sqrt(12),) * 2)
    names = {s.name for s in psf.specs()}
    assert {"pupil.gamma", "pupil.ring_h"} <= names and model.geometry.drift == "linear"


def test_a_matched_sample_has_no_ring_and_the_aperture_cuts_at_the_objectives_na():
    model = bead_model(MATCHED, BeadSettings(psf=PSFModelSettings(zernike_order=4)), ROI, z_max_nm=1000, device="cpu")
    pupil = model.psf.pupil
    assert not pupil.ring and "pupil.ring_h" not in {s.name for s in pupil.specs()}
    rho = pupil.grid.rho.real if pupil.grid.rho.is_complex() else pupil.grid.rho
    edge = pupil.na / pupil.grid_na                                   # 0.993: the NA inside the grid's edge
    assert pupil.cutoff[rho < edge - 0.04].min() > 0.95
    assert pupil.cutoff[(rho > edge) & (rho <= 1)].max() < 0.5


def test_gamma_bends_the_magnitude_as_cos_theta_to_its_power():
    model = bead_model(MATCHED, BeadSettings(psf=PSFModelSettings(zernike_order=4)), ROI, z_max_nm=1000, device="cpu")
    pupil = model.psf.pupil
    p = {"pupil.phase": torch.zeros(len(pupil.phase_modes)), "pupil.magnitude": torch.zeros(0),
         "pupil.gamma": torch.tensor(0.5)}
    magnitude, _ = pupil(p)
    expected = pupil.cos_imm ** 0.5 * pupil.cutoff
    assert torch.allclose(magnitude, expected)


def test_the_pixel_blur_adds_the_pixels_own_under_the_learnt_one():
    model = bead_model(OPTICS, BeadSettings(psf=PSFModelSettings(zernike_order=4)), ROI, z_max_nm=1000, device="cpu")
    psf = model.psf
    sigma = psf.image_sigma({"psf.sigma": torch.tensor([20.0])})
    assert torch.allclose(sigma, torch.sqrt(20.0 ** 2 + torch.tensor(psf.pixel_sigma) ** 2))
    without = bead_model(OPTICS, BeadSettings(psf=PSFModelSettings(zernike_order=4, pixel_blur=False)), ROI,
                         z_max_nm=1000, device="cpu")
    assert torch.allclose(without.psf.image_sigma({"psf.sigma": torch.tensor([20.0])}), torch.tensor([20.0, 20.0]))


@pytest.mark.slow
def test_a_calibration_recovers_gamma_the_saf_damping_and_the_blur_beyond_the_pixel():
    model = bead_model(OPTICS, BeadSettings(psf=PSFModelSettings(zernike_order=4)), ROI, z_max_nm=1000, device="cpu")
    beads = simulated(model, extra={"pupil.gamma": torch.tensor(GAMMA), "pupil.ring_h": torch.tensor(RING_H)})
    cal = calibrate_beads(beads, OPTICS, BeadSettings(psf=PSFModelSettings(zernike_order=4)), device="cpu")
    shared = cal.shared()
    assert float(shared["pupil.gamma"]) == pytest.approx(GAMMA, abs=0.04)
    assert float(shared["pupil.ring_h"]) == pytest.approx(RING_H, abs=15.0)
    assert float(shared["psf.sigma"]) == pytest.approx(SIGMA, abs=4.0)
    assert torch.allclose(shared["pupil.phase"][:4].cpu(), PHASE, atol=0.02)


@pytest.mark.slow
def test_the_linear_drift_recovers_each_stacks_rate():
    settings = BeadSettings(psf=PSFModelSettings(zernike_order=4), drift="linear")
    model = bead_model(MATCHED, settings, ROI, z_max_nm=1000, device="cpu")
    rates = torch.tensor([[30.0, 10.0], [-15.0, 20.0]])                         # nm per µm, two stacks
    stacks = lambda b: replace(b, stack_id=torch.arange(b.n_emitters) % 2)      # noqa: E731
    beads = simulated(model, n=12, extra={"geometry.drift_rate": rates}, z_stack=stacks)
    cal = calibrate_beads(beads, MATCHED, settings, device="cpu")
    assert torch.allclose(cal.shared()["geometry.drift_rate"].cpu(), rates, atol=3.0)


def test_a_minimal_model_describes_and_builds_again():
    model = bead_model(OPTICS, BeadSettings(psf=PSFModelSettings(zernike_order=4), drift="linear"), ROI, z_max_nm=1000,
                       device="cpu")
    again = build(describe(model), device="cpu")
    pupil = again.psf.pupil
    assert isinstance(pupil, MinimalPupil) and pupil.na == model.psf.pupil.na and pupil.ring
    assert pupil.phase_modes == model.psf.pupil.phase_modes
    assert again.psf.pixel_blur and again.geometry.drift == "linear" and again.geometry.drift_axes == "xy"
    assert torch.equal(pupil.cutoff, model.psf.pupil.cutoff)
    zernike = bead_model(OPTICS, BeadSettings(psf=PSFModelSettings(zernike_order=4, magnitude="free",
                                                                   pixel_blur=False)), ROI,
                         z_max_nm=1000, device="cpu")
    again = build(describe(zernike), device="cpu")
    assert isinstance(again.psf.pupil, ZernikePupil) and not again.psf.pixel_blur


def test_a_start_from_a_model_without_the_pixel_blur_keeps_the_images_blur():
    old = bead_model(OPTICS, BeadSettings(psf=PSFModelSettings(zernike_order=4, magnitude="free", pixel_blur=False)),
                     ROI,
                     z_max_nm=1000, device="cpu")
    new = bead_model(OPTICS, BeadSettings(psf=PSFModelSettings(zernike_order=4)), ROI, z_max_nm=1000, device="cpu")

    class Previous:
        model, shared = old, {"psf.sigma": torch.tensor([40.0, 40.0])}
    sigma = start_values(new, Previous)["psf.sigma"]
    pixel = torch.tensor(new.psf.pixel_sigma, dtype=torch.float64)
    assert torch.allclose(sigma.double(), torch.sqrt(40.0 ** 2 - pixel ** 2).mean().reshape(1), atol=1e-6)


def test_a_learnt_na_cuts_where_a_held_one_would_and_travels_with_the_model():
    """``learn_na``: the aperture at ``pupil.na`` is the held aperture of a pupil built at that NA; the model file
    keeps the choice, the description the NA, and a model holding γ (in situ) holds the NA with it."""
    from uipsf.model import held_from, make_pupil
    learnt = bead_model(MATCHED, BeadSettings(psf=PSFModelSettings(zernike_order=4, learn_na=True)), ROI,
                        z_max_nm=1000, device="cpu")
    pupil = learnt.psf.pupil
    spec = {s.name: s for s in pupil.specs()}["pupil.na"]
    assert pupil.learn_na and float(spec.init) == pytest.approx(1.5)                     # the typed NA
    held = make_pupil(replace(MATCHED, NA=1.45), PSFModelSettings(zernike_order=4), grid=pupil.grid)
    p = {"pupil.phase": torch.zeros(len(pupil.phase_modes)), "pupil.magnitude": torch.zeros(0),
         "pupil.gamma": torch.tensor(0.3), "pupil.na": torch.tensor(1.45)}
    assert torch.allclose(pupil(p)[0], held(p)[0], atol=1e-6)
    assert pupil.values(p)["na"] == pytest.approx(1.45) and "pupil.na" in {s.name for s in pupil.truncated(2).specs()}
    assert build(describe(learnt), device="cpu").psf.pupil.learn_na
    description = learnt.psf.describe(p | {"psf.sigma": torch.tensor([20.0])})
    assert description["na"] == pytest.approx(1.45) and description["NA"] == pytest.approx(1.45)
    insitu = make_pupil(MATCHED, PSFModelSettings(zernike_order=4, learn_na=True),
                        held_from(description, 4, 600.0))
    assert not insitu.learn_na and insitu.na == pytest.approx(1.45) and insitu.held_gamma == pytest.approx(0.3)
    nominal = bead_model(MATCHED, BeadSettings(psf=PSFModelSettings(zernike_order=4)), ROI, z_max_nm=1000,
                         device="cpu")
    assert "na" not in nominal.psf.describe(p | {"psf.sigma": torch.tensor([20.0])})


@pytest.mark.slow
def test_a_calibration_learns_an_effective_na_below_the_objectives():
    """Beads drawn with the aperture at 1.42 behind a typed 1.5 give the 1.42 back with *learn NA*, and γ, within
    three CRLBs."""
    settings = BeadSettings(psf=PSFModelSettings(zernike_order=4, learn_na=True))
    model = bead_model(MATCHED, settings, ROI, z_max_nm=1000, device="cpu")
    beads = simulated(model, extra={"pupil.gamma": torch.tensor(GAMMA), "pupil.na": torch.tensor(1.42)})
    cal = calibrate_beads(beads, MATCHED, settings, device="cpu")
    shared, crlb = cal.shared(), cal.result.std
    assert abs(float(shared["pupil.na"]) - 1.42) < 3 * float(crlb["pupil.na"])
    assert abs(float(shared["pupil.gamma"]) - GAMMA) < 3 * float(crlb["pupil.gamma"])


def test_the_minimal_pupil_survives_the_first_stages_truncation():
    model = bead_model(OPTICS, BeadSettings(psf=PSFModelSettings(zernike_order=6)), ROI, z_max_nm=1000, device="cpu")
    low = model.psf.pupil.truncated(4)
    assert isinstance(low, MinimalPupil) and low.ring and len(low.phase_modes) < len(model.psf.pupil.phase_modes)
    assert {"pupil.gamma", "pupil.ring_h"} <= {s.name for s in low.specs()}


def test_a_grid_for_the_minimal_pupil_stays_short_of_the_immersion_index():
    from uipsf.model import minimal_grid_optics
    assert minimal_grid_optics(OPTICS).NA == pytest.approx(1.51)
    assert minimal_grid_optics(replace(OPTICS, n_imm=1.512)).NA == pytest.approx(1.507)
    PupilGrid.build(minimal_grid_optics(OPTICS), 64)                  # a valid optics


def test_the_output_page_reports_gamma_the_ring_the_blur_the_drift_and_the_tilt():
    from types import SimpleNamespace

    from uipsf.diagnostics.figures import model_text
    model = bead_model(OPTICS, BeadSettings(psf=PSFModelSettings(zernike_order=4), drift="linear", fit_tilt=True), ROI,
                       z_max_nm=1000,
                       device="cpu")
    shared = {"pupil.gamma": torch.tensor(0.35), "pupil.ring_h": torch.tensor(60.0), "psf.sigma": torch.tensor([21.0]),
              "geometry.drift_rate": torch.tensor([[30.0, 10.0], [-15.0, 20.0]]),
              "geometry.tilt": torch.tensor([0.01, 0.0])}
    crlb = {"pupil.gamma": torch.tensor(0.002), "psf.sigma": torch.tensor([0.5])}
    text = model_text(SimpleNamespace(model=model, shared=shared, crlb=crlb))
    assert "γ 0.350 ± 0.002" in text and "ring h 60 nm" in text
    assert "blur σ 21.0 ± 0.5 nm beyond the pixel" in text
    assert "drift nm/µm x 30, y 10; x -15, y 20" in text and "coverslip tilt x 0.57°" in text
    free = bead_model(MATCHED, BeadSettings(psf=PSFModelSettings(zernike_order=4, magnitude="free"), drift="free"),
                      ROI, z_max_nm=1000,
                      device="cpu")
    drift = torch.zeros(K, 3)
    drift[:, 0] = torch.linspace(-10, 10, K)
    text = model_text(SimpleNamespace(model=free, shared={"psf.sigma": torch.tensor([30.0]),
                                                          "geometry.drift": drift}, crlb={}))
    assert "γ" not in text and "beyond the pixel" in text and f"over {K} slices" in text and "z" not in text


def test_a_spline_for_single_molecules_is_drawn_with_the_pixel_blur_and_without_the_beads_saf_damping():
    from uipsf.export.model_file import sample_psf
    model = bead_model(OPTICS, BeadSettings(psf=PSFModelSettings(zernike_order=4)), ROI, z_max_nm=1000, device="cpu")
    phase = torch.zeros(len(model.psf.pupil.phase_modes))
    shared = {"pupil.phase": phase, "pupil.magnitude": torch.zeros(0), "pupil.gamma": torch.tensor(GAMMA),
              "psf.sigma": torch.tensor([SIGMA])}
    xyz = torch.tensor([[0.0, 0.0, -300.0], [0.0, 0.0, 0.0], [0.0, 0.0, 300.0]])
    high = sample_psf(model, shared | {"pupil.ring_h": torch.tensor(200.0)}, xyz)
    glass = sample_psf(model, shared | {"pupil.ring_h": torch.tensor(0.0)}, xyz)
    assert torch.allclose(high, glass)                               # h is the beads', not a molecule's
    sharp = sample_psf(bead_model(OPTICS, BeadSettings(psf=PSFModelSettings(zernike_order=4, pixel_blur=False)), ROI,
                                  z_max_nm=1000,
                                  device="cpu"), shared | {"pupil.ring_h": torch.tensor(0.0)}, xyz)
    assert sharp.max() > glass.max() * 1.02                           # the pixel's blur is in the spline
    beads = sample_psf(model, shared | {"pupil.ring_h": torch.tensor(200.0)}, torch.tensor(
        [[0.0, 0.0, z] for z in (-100.0, 0.0, 100.0)]), include_bead=True)
    assert not torch.allclose(beads, glass)                          # the beads' own images keep it


def test_without_saf_damping_the_ring_is_a_points_on_the_coverslip():
    model = bead_model(OPTICS, BeadSettings(psf=PSFModelSettings(zernike_order=4, saf_damping=False)), ROI,
                       z_max_nm=1000, device="cpu")
    pupil = model.psf.pupil
    assert not pupil.ring and "pupil.ring_h" not in {s.name for s in pupil.specs()}
    assert not build(describe(model), device="cpu").psf.pupil.saf_damping


def test_every_value_the_model_learnt_or_held_is_in_the_line_and_the_zernike_tables_last_rows():
    """Besides the modes, the table holds every other number of the model -- learnt with its CRLB, held
    without -- and the page's line every one learnt: none left for the reader to find in the file."""
    from types import SimpleNamespace

    import numpy as np

    from uipsf.diagnostics.figures import model_text, model_values, zernike_table
    model = bead_model(OPTICS, BeadSettings(psf=PSFModelSettings(zernike_order=4), drift="linear", fit_tilt=True), ROI,
                       z_max_nm=1000, device="cpu")
    shared = {"pupil.phase": torch.zeros(len(model.psf.pupil.phase_modes)), "pupil.magnitude": torch.zeros(0),
              "pupil.gamma": torch.tensor(0.35), "pupil.ring_h": torch.tensor(60.0), "psf.sigma": torch.tensor([21.0]),
              "geometry.drift_rate": torch.tensor([[30.0, 10.0], [-15.0, 20.0]]),
              "geometry.tilt": torch.tensor([0.01, 0.0]), "geometry.z_scale": torch.tensor(0.97),
              "emitter.xyz": torch.zeros(3, 3), "something.new": torch.tensor([1.5, 2.5])}
    crlb = {"pupil.gamma": torch.tensor(0.002), "psf.sigma": torch.tensor([0.5]),
            "geometry.z_scale": torch.tensor(0.004), "something.new": torch.tensor([0.1, 0.2])}
    source = SimpleNamespace(model=model, shared=shared, crlb=crlb)
    rows = {(r["parameter"], r["element"]): r for r in model_values(source)}
    assert rows[("γ (apodisation)", "")]["value"] == pytest.approx(0.35)
    assert rows[("γ (apodisation)", "")]["error"] == pytest.approx(0.002)
    assert rows[("SAF ring height h", "")]["how"] == "held"                    # no CRLB: held
    assert rows[("drift rate", "stack 2 y")]["value"] == pytest.approx(20.0)
    assert rows[("coverslip tilt", "x")]["value"] == pytest.approx(np.degrees(0.01))
    assert rows[("z scale", "")]["how"] == "learnt"
    assert rows[("something.new", "2")]["value"] == pytest.approx(2.5)         # unknown: under its own name
    assert not any("emitter" in p or p.startswith("pupil.phase") for p, _ in rows)
    text = model_text(source)
    assert "z scale 0.97 ± 0.004" in text and "something.new 1 1.5 ± 0.1, 2 2.5 ± 0.2" in text
    # the table: the modes, then these
    columns, table, colours = zernike_table(source)
    extra = [row for row in table if row[1] == ""]
    assert len(extra) == len(rows) and columns[-4:] == ["value", "± (value)", "unit", "learnt or held"]
    assert {(row[2], row[3]) for row in extra} == set(rows)
    assert all(np.isnan(row[columns.index("value")]) for row in table if row[1] != "")
