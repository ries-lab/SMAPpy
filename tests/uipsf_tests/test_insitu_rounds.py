"""In situ learning in rounds, and the coverslip edge that pins the focus (M5, step 5)."""
import numpy as np
import pytest
import torch

from uipsf.workflows.insitu_rounds import _focus_per_height, _moved, coverslip_edge


def heights(n=1500, on_coverslip=0.2, shift=120.0, sd=30.0, scatter=1.2, outliers=0.01, seed=0):
    """Localised heights of molecules uniform 0–1400 nm deep, ``on_coverslip`` of them at 0, all shifted by
    ``shift``; each with a CRLB about ``sd`` that understates its scatter ``scatter`` times, and a few
    ``outliers`` hundreds of nanometres too low."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 1400, n)
    t[rng.random(n) < on_coverslip] = 0.0
    s = sd * (0.6 + 0.8 * rng.random(n))
    h = t + shift + rng.normal(0, s * scatter)
    out = rng.random(n) < outliers
    h[out] -= rng.uniform(100, 400, out.sum())
    return h, s


def test_the_edge_is_where_the_molecules_on_the_coverslip_are_whatever_the_outliers():
    found = [coverslip_edge(*heights(seed=k)) for k in range(5)]
    assert all(e.spike > 0.2 for e in found)
    shifts = np.array([e.shift_nm for e in found])
    assert abs(shifts.mean() - 120.0) < 3 and shifts.std() < 5
    assert np.median([e.sd_nm for e in found]) < 5


def test_without_molecules_on_the_coverslip_the_flat_edge_is_found_and_a_spike_rarely_invented():
    found = [coverslip_edge(*heights(on_coverslip=0.0, seed=k)) for k in range(5)]
    assert sum(e.spike > 0 for e in found) <= 1
    assert abs(np.mean([e.shift_nm for e in found]) - 120.0) < 10


def test_too_few_heights_give_no_edge():
    assert coverslip_edge(np.arange(10.0), np.ones(10)) is None


def test_after_two_corrections_the_focus_per_height_is_their_secant():
    assert _focus_per_height([(700.0, 30.0)], 1.6) == 1.6
    assert _focus_per_height([(700.0, 30.0), (650.0, 5.0)], 1.6) == pytest.approx(2.0)
    assert _focus_per_height([(700.0, 30.0), (705.0, 5.0)], 1.6) == 1.6        # the foci too close


def test_moved_is_the_rms_change_in_standard_errors_of_the_change():
    sd = {"pupil.phase": torch.full((4,), 0.01), "geometry.focus": torch.tensor(5.0)}
    old = {"pupil.phase": torch.zeros(4), "geometry.focus": torch.tensor(700.0)}
    new = {"pupil.phase": torch.tensor([0.01, -0.01, 0.0, 0.0]) * np.sqrt(2), "geometry.focus": torch.tensor(700.0)}
    assert _moved(new, old, sd, sd) == pytest.approx(np.sqrt(2 / 5))


@pytest.mark.slow
def test_the_coverslip_edge_brings_a_focus_started_250_nm_off_to_the_truth():
    # 300 molecules 0–1400 nm deep, a fifth on the coverslip, 300 frames, the focus started at 450 nm against
    # 700; two rounds.  ~2.5 min on one core.
    from smappy.simulate import BlinkingSettings

    import uipsf
    from test_insitu_rois import FOCUS, OPTICS, I, model_and_shared
    from uipsf.model import Bleaching, InSituGeometry
    from uipsf.movie import MovieSettings, simulate_movie
    from uipsf.workflows.insitu_rounds import RoundSettings, calibrate_insitu
    model, shared = model_and_shared()
    s = MovieSettings(size_px=(128, 128), n_frames=300, fluorophores=300, heights_nm=(0.0, 1400.0), on_coverslip=0.2,
                      illumination_fwhm_px=160.0, seed=1,
                      blinking=BlinkingSettings(photons=6000.0, photons_std=3000.0, on_time=1.5, blinks=3.0))
    movie = simulate_movie(model, shared, s)
    sel = I.InSituSettings(per_bin=10)
    rois = I.insitu_rois(movie.frames, OPTICS.pixel_size_nm, sel)
    start_model = uipsf.ForwardModel(model.psf, geometry=InSituGeometry(450.0), intensity=Bleaching(order=0))
    phase = torch.zeros_like(shared["pupil.phase"])
    phase[1] = 0.4
    run = calibrate_insitu(start_model, rois, start={"pupil.phase": phase}, selection=sel,
                           settings=RoundSettings(max_rounds=2))
    first, last = (r.calibration.calibration for r in (run.rounds[0], run.rounds[-1]))
    f1, f2 = float(first.shared()["geometry.focus"]), float(last.shared()["geometry.focus"])
    assert len(run.rounds) == 2 and run.rounds[-1].edge.spike > 0.1
    assert abs(f2 - FOCUS) < 15 and abs(f2 - FOCUS) < abs(f1 - FOCUS) / 2
    sd = last.result.std["pupil.phase"]
    assert torch.all((last.shared()["pupil.phase"] - shared["pupil.phase"]).abs() < 4 * sd)
    assert run.final.twin.verdict == "agree"
