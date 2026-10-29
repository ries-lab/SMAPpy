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


@pytest.mark.slow
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


@pytest.mark.long
def test_a_previous_model_starts_the_first_stage_and_lands_on_the_truth(beads, symmetric_fixed):
    """Into stage 1, not instead of it: on the lab's M2 that was faster and never failed, where replacing
    stage 1 from another camera's model once did (docs/uipsf/architecture.md, "How a bead fit starts")."""
    cal = calibrate_beads(beads, OPTICS, BeadSettings(zernike_order=6), start=symmetric_fixed, device="cpu")
    assert cal.first is not None
    assert {h["stage"] for h in cal.result.history} == {1, 2}
    # stage 1 began at the previous model's low modes, not at zero
    assert cal.first.history[0]["objective"] < calibrate_beads(
        beads, OPTICS, BeadSettings(zernike_order=6), device="cpu").first.history[0]["objective"]
    assert torch.allclose(cal.shared()["pupil.phase"][:4].cpu(), PHASE, atol=0.01)


def test_a_two_channel_model_needs_a_transform_per_channel():
    with pytest.raises(ValueError, match="transforms"):
        bead_model([OPTICS, OPTICS], BeadSettings(), (13, 13), transforms=[None], device="cpu")


def test_every_figure_of_a_calibration_draws_with_and_without_a_reference(symmetric_fixed):
    from uipsf.diagnostics import figures
    drawn = figures(symmetric_fixed, reference=symmetric_fixed)
    assert {"Pupil and Zernikes", "Residual atlas"} <= set(drawn)
    for draw in list(drawn.values()) + list(figures(symmetric_fixed).values()):
        fig = Figure()
        draw(fig)
        assert fig.axes


def test_the_zernike_table_holds_the_bars_numbers_and_colours_the_key_aberrations(symmetric_fixed):
    from uipsf.diagnostics.figures import FAMILY_COLOURS, zernike_table, zernikes
    columns, rows, colours = zernike_table(symmetric_fixed, reference=symmetric_fixed)
    modes, values, _ = zernikes(symmetric_fixed, "phase")
    phase = {row[1]: row[columns.index("phase (rad)")] for row in rows}
    assert all(phase[j] == pytest.approx(v) for j, v in zip(modes, values, strict=True))
    start = columns.index("start phase (nm)")
    assert all(row[start] == pytest.approx(row[columns.index("phase (nm)")]) for row in rows if row[1] in modes)
    for row, colour in zip(rows, colours, strict=True):
        assert colour == FAMILY_COLOURS.get(row[columns.index("aberration")])


def test_the_phase_is_shown_in_rad_waves_or_nm_of_the_channels_wavelength(symmetric_fixed):
    import numpy as np

    from uipsf.diagnostics.figures import draw_zernikes, phase_scale, zernike_table, zernikes
    assert phase_scale(symmetric_fixed, 0, "waves")[0] == pytest.approx(1 / (2 * np.pi))
    assert phase_scale(symmetric_fixed, 0, "nm")[0] == pytest.approx(660 / (2 * np.pi))
    modes, values, crlb = zernikes(symmetric_fixed, "phase")
    fig = Figure()
    draw_zernikes(fig, symmetric_fixed, units="nm")
    heights = [bar.get_height() for bar in fig.axes[0].patches[:len(modes)]]
    assert np.allclose(heights, values * 660 / (2 * np.pi))
    assert fig.axes[0].get_ylabel() == "nm"
    columns, rows, _ = zernike_table(symmetric_fixed)
    for row in rows:
        rad, err = row[columns.index("phase (rad)")], row[columns.index("± (rad)")]
        assert row[columns.index("phase (waves)")] == pytest.approx(rad / (2 * np.pi), nan_ok=True)
        assert row[columns.index("phase (nm)")] == pytest.approx(rad * 660 / (2 * np.pi), nan_ok=True)
        assert row[columns.index("± (nm)")] == pytest.approx(err * 660 / (2 * np.pi), nan_ok=True)


def test_one_switch_sets_the_unit_every_aberration_is_shown_in(symmetric_fixed, monkeypatch):
    """`PHASE_UNIT`: nm by default -- the bars, their axis, the table's first phase column and the window's
    first *units* -- and set to waves, all of them follow."""
    import importlib

    import numpy as np
    figures = importlib.import_module("uipsf.diagnostics.figures")   # the package's `figures` is a function
    for unit, factor in (("nm", 660 / (2 * np.pi)), ("waves", 1 / (2 * np.pi))):
        monkeypatch.setattr(figures, "PHASE_UNIT", unit)
        modes, values, _ = figures.zernikes(symmetric_fixed, "phase")
        fig = Figure()
        figures.draw_zernikes(fig, symmetric_fixed)
        assert fig.axes[0].get_ylabel() == unit
        assert np.allclose([b.get_height() for b in fig.axes[0].patches[:len(modes)]], values * factor)
        columns, _, _ = figures.zernike_table(symmetric_fixed, reference=symmetric_fixed)
        assert columns[6] == f"phase ({unit})" and f"start phase ({unit})" in columns
        assert figures.unit_choices()[0] == unit and sorted(figures.unit_choices()) == sorted(figures.PHASE_UNITS)


@pytest.mark.long
def test_halves_of_simulated_beads_scatter_as_the_crlb_says_and_are_kept_in_the_model_file(beads, tmp_path):
    import numpy as np

    from uipsf.diagnostics import MIN_BEADS, half_splits, split_beads
    from uipsf.diagnostics.figures import draw_zernikes, zernike_table
    from uipsf.export import load_model, save_model
    settings = BeadSettings(zernike_order=4, magnitude="fixed", blur="fixed", blur_nm=SIGMA)
    cal = calibrate_beads(beads, OPTICS, settings, device="cpu")
    used = cal.beads.used
    pairs = split_beads(used, 3, seed=1)
    assert all(sorted(a + b) == sorted(used) and not set(a) & set(b) for a, b in pairs)   # complementary
    assert len({frozenset(map(frozenset, pair)) for pair in pairs}) == 3                     # all different
    halves = half_splits(beads, OPTICS, cal.options, used, splits=3, seed=1, device="cpu")
    spread = halves.std["pupil.phase"]
    crlb = cal.result.std["pupil.phase"].double().numpy()
    assert halves.values["pupil.phase"].shape == (3, 2, len(crlb)) and np.all(np.isfinite(spread))
    # an exact model and photon noise only: the halves agree with the CRLB, to the estimate's own ±40 %
    # at 3 splits (and the few beads); a factor of 3 either way is far outside that
    ratio = np.median(spread / crlb)
    assert 1 / 3 < ratio < 3
    fig = Figure()
    draw_zernikes(fig, cal, halves=halves)
    assert fig.axes
    columns, rows, _ = zernike_table(cal, halves=halves)
    assert "halves / CRLB" in columns
    cal.halves = halves
    save_model(cal, tmp_path / "model.h5")
    back = load_model(tmp_path / "model.h5", device="cpu").halves
    assert back.splits == [tuple(p) for p in halves.splits] and np.allclose(back.std["pupil.phase"], spread)
    with pytest.raises(ValueError, match="at least"):
        half_splits(beads, OPTICS, cal.options, used[:MIN_BEADS - 1], device="cpu")
