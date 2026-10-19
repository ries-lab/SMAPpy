"""calibrate_beads: the model from the optics and a few choices, a start from a previous model, the figures."""
import matplotlib
import pytest
import torch
from matplotlib.figure import Figure

import uipsf
from uipsf.optics.zernike import noll_to_nm
from uipsf.simulate import parameters, simulate, template
from uipsf.workflows import BeadSettings, bead_model, calibrate_beads, magnitude_modes, start_values

matplotlib.use("Agg")

OPTICS = uipsf.Optics(NA=1.43, wavelength_nm=660, n_imm=1.518, n_med=1.33, pixel_size_nm=(110.0, 110.0))
PHASE = torch.tensor([0.6, -0.3, 0.2, 0.15])          # Noll 5–8: astigmatism and coma
SIGMA = 40.0


@pytest.fixture(scope="module")
def beads():
    """Ten bright beads through ±1 µm, aberrated, blurred by σ = 40 nm, with a uniform pupil magnitude."""
    n, roi, K = 10, (13, 13), 21
    model = bead_model(OPTICS, BeadSettings(zernike_order=4, magnitude="fixed"), roi, z_max_nm=1000, device="cpu")
    g = torch.Generator().manual_seed(1)
    phase = torch.zeros(len(model.psf.pupil.phase_modes))
    phase[:4] = PHASE
    values = {"pupil.phase": phase, "psf.sigma": torch.tensor([SIGMA, SIGMA]),
              "emitter.xyz": (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([150.0, 150.0, 200.0]),
              "emitter.photons": 20000 * (1 + torch.rand(n, generator=g)),
              "emitter.background": torch.full((n,), 20.0)}
    b0 = template(n, uipsf.Batch.z_from_objective(torch.linspace(-1000, 1000, K)), roi, device="cpu")
    return simulate(model, parameters(model, b0, values, device="cpu"), b0, seed=3)


@pytest.fixture(scope="module")
def symmetric_fixed(beads):
    return calibrate_beads(beads, OPTICS, BeadSettings(zernike_order=4, magnitude="symmetric", blur="fixed",
                                                         blur_nm=SIGMA), device="cpu")


def test_the_magnitude_choices_give_every_mode_the_symmetric_ones_or_none():
    every = magnitude_modes(6, "free")
    assert every[0] == 2 and len(every) == 27                       # no piston, up to n = 6
    assert magnitude_modes(6, "symmetric") == [4, 11, 22]
    assert all(noll_to_nm(j)[1] == 0 for j in magnitude_modes(6, "symmetric"))
    assert magnitude_modes(6, "fixed") == []


def test_settings_refuse_a_choice_they_do_not_have():
    with pytest.raises(ValueError, match="magnitude"):
        BeadSettings(magnitude="half")
    with pytest.raises(ValueError, match="blur"):
        BeadSettings(blur="maybe")


def test_a_calibration_recovers_the_aberrations_and_the_blur_with_everything_free(beads):
    cal = calibrate_beads(beads, OPTICS, BeadSettings(zernike_order=4), device="cpu")
    shared = cal.shared()
    assert torch.allclose(shared["pupil.phase"][:4].cpu(), PHASE, atol=0.01)
    assert shared["pupil.phase"][4:].abs().max() < 0.01
    assert torch.allclose(shared["psf.sigma"].cpu(), torch.tensor([SIGMA, SIGMA]), atol=2.0)
    assert cal.beads.used == list(range(10))


def test_a_fixed_blur_stays_at_its_value_and_the_model_file_records_the_choices(symmetric_fixed, tmp_path):
    cal = symmetric_fixed
    assert cal.shared()["psf.sigma"].tolist() == [SIGMA, SIGMA]
    assert "psf.sigma" not in cal.result.std                          # not fitted, so no CRLB
    assert cal.model.psf.pupil.magnitude_modes == [4, 11]
    assert torch.allclose(cal.shared()["pupil.phase"][:4].cpu(), PHASE, atol=0.01)
    path = tmp_path / "m.uipsf.h5"
    uipsf.save_model(cal, path)
    loaded = uipsf.load_model(path, device="cpu")
    assert loaded.options["magnitude"] == "symmetric" and loaded.options["blur"] == "fixed"
    assert loaded.model.psf.fit_sigma is False
    assert loaded.model.psf.pupil.magnitude_modes == [4, 11]


def test_a_previous_model_starts_a_higher_order_by_its_noll_modes(symmetric_fixed):
    model = bead_model(OPTICS, BeadSettings(zernike_order=6), (13, 13), device="cpu")
    values = start_values(model, symmetric_fixed)
    new = model.psf.pupil
    assert len(values["pupil.phase"]) == len(new.phase_modes)
    assert torch.allclose(values["pupil.phase"][:4].float(), symmetric_fixed.shared()["pupil.phase"][:4].cpu())
    assert values["pupil.phase"][len(symmetric_fixed.model.psf.pupil.phase_modes):].abs().max() == 0
    magnitude = dict(zip(new.magnitude_modes, values["pupil.magnitude"].tolist(), strict=True))
    old = dict(zip([4, 11], symmetric_fixed.shared()["pupil.magnitude"].tolist(), strict=True))
    assert all(magnitude[j] == pytest.approx(old[j]) for j in (4, 11))
    assert all(v == 0 for j, v in magnitude.items() if j not in (4, 11))
    assert values["psf.sigma"].tolist() == [SIGMA, SIGMA]            # the new blur is free: it starts there
    fixed = bead_model(OPTICS, BeadSettings(blur="fixed", blur_nm=25.0), (13, 13), device="cpu")
    assert "psf.sigma" not in start_values(fixed, symmetric_fixed)   # a fixed blur keeps the settings' value


def test_starting_from_a_previous_model_replaces_the_first_stage_and_lands_on_the_truth(beads, symmetric_fixed):
    cal = calibrate_beads(beads, OPTICS, BeadSettings(zernike_order=6), start=symmetric_fixed, device="cpu")
    assert cal.first is None
    assert {h["stage"] for h in cal.result.history} == {2}
    assert torch.allclose(cal.shared()["pupil.phase"][:4].cpu(), PHASE, atol=0.01)


def test_a_two_channel_model_needs_a_transform_per_channel():
    with pytest.raises(ValueError, match="transforms"):
        bead_model([OPTICS, OPTICS], BeadSettings(), (13, 13), transforms=[None], device="cpu")


def test_every_figure_of_a_calibration_draws_with_and_without_a_reference(symmetric_fixed):
    from uipsf.diagnostics import figures
    drawn = figures(symmetric_fixed, reference=symmetric_fixed)
    assert {"Pupil", "Zernikes", "Residual atlas"} <= set(drawn)
    for draw in list(drawn.values()) + list(figures(symmetric_fixed).values()):
        fig = Figure()
        draw(fig)
        assert fig.axes
