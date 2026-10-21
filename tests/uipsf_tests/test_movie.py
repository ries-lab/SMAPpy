"""Simulated blinking movies imaged with a uipsf model: the truth, the spots, the camera (M5, step 3)."""
import numpy as np
import torch

import uipsf
from uipsf.model import Bleaching, InSituGeometry
from uipsf.movie import MovieSettings, MovieTruth, close_pairs, illumination, render, simulate_movie

f64 = torch.float64


def model_and_shared(optics, focus=600.0, roi=(11, 11), P=24):
    grid = uipsf.PupilGrid.build(optics, P, device="cpu", dtype=f64)
    pupil = uipsf.ZernikePupil(grid, n_max=4)
    model = uipsf.ForwardModel(uipsf.VectorPSF(grid, pupil, roi, sigma_nm=(30.0, 30.0)),
                               geometry=InSituGeometry(focus), intensity=Bleaching(order=0))
    phase = torch.zeros(len(pupil.phase_modes), dtype=f64)
    phase[:3] = torch.tensor([0.1, 0.7, -0.2])
    shared = {"pupil.phase": phase, "pupil.magnitude": torch.zeros(len(pupil.magnitude_modes), dtype=f64),
              "psf.sigma": torch.tensor([35.0, 35.0], dtype=f64), "geometry.focus": torch.tensor(focus, dtype=f64)}
    return model, shared


def one(x_px, y_px, h_nm, photons, pixel, frame=0):
    a = np.array
    return MovieTruth(a([frame]), a([0]), a([0]), a([x_px * pixel[0]]), a([y_px * pixel[1]]), a([h_nm]),
                      a([photons], float))


def test_a_spot_cut_from_the_movie_is_the_forward_models_expected_photons(optics):
    model, shared = model_and_shared(optics)
    image = np.zeros((1, 40, 40))
    x, y, h, N = 20.3, 15.7, 400.0, 3000.0
    render(model, shared, one(x, y, h, N, optics.pixel_size_nm), image, spot_px=25)
    px, py = optics.pixel_size_nm
    batch = uipsf.Batch(data=torch.zeros(1, 1, 1, 11, 11, dtype=f64), z_nm=torch.zeros(1, dtype=f64))
    values = dict(shared, **{"emitter.xyz": torch.tensor([[(x - 20) * px, (y - 16) * py, h]], dtype=f64),
                             "emitter.photons": torch.tensor([N], dtype=f64),
                             "emitter.background": torch.tensor([0.0], dtype=f64)})
    mu = model(values, batch)[0, 0, 0].numpy()
    cut = image[0, 16 - 5:16 + 6, 20 - 5:20 + 6]
    # The fit's region blurs circularly (`optics.kernels`): at its edges it differs from the movie, which
    # has no wrap-around, by 0.2–0.6 % of the peak; in its centre the two are the same.
    assert np.abs(cut - mu).max() < 0.006 * mu.max()
    assert np.abs(cut - mu)[3:8, 3:8].max() < 2e-4 * mu.max()


def test_the_photons_of_a_spot_are_in_the_frame_but_for_what_falls_outside_its_region():
    # Matched indices and an NA below them: no light is lost to depth, so the frame holds the photons less
    # the rim beyond the spot's region.  (Into water at NA 1.43 an emitter away from the coverslip keeps
    # only ~60 %: the supercritical light the normalisation counts never leaves it.)
    optics = uipsf.Optics(NA=1.2, wavelength_nm=680, n_imm=1.33, n_med=1.33, pixel_size_nm=(110, 110))
    model, shared = model_and_shared(optics, focus=400.0, P=64)
    shared["pupil.phase"] = torch.zeros_like(shared["pupil.phase"])
    image = np.zeros((1, 60, 60))
    render(model, shared, one(30.0, 30.0, 400.0, 5000.0, optics.pixel_size_nm), image, spot_px=41)
    assert 0.95 * 5000.0 < image.sum() <= 5000.0                 # 96 %: the far rings beyond 20 pixels


def test_the_illumination_is_one_at_the_centre_and_half_at_half_its_width():
    s = MovieSettings(size_px=(65, 65), illumination_fwhm_px=40.0)
    assert np.isclose(illumination(s, 32.0, 32.0), 1.0)
    assert np.isclose(illumination(s, 52.0, 32.0), 0.5)
    assert np.all(illumination(MovieSettings(), np.arange(5), 3.0) == 1.0)


def test_a_movie_has_its_molecules_where_the_truth_says_with_poisson_noise_on_the_mean(optics):
    from smappy.simulate import BlinkingSettings
    model, shared = model_and_shared(optics)
    s = MovieSettings(size_px=(32, 40), n_frames=60, fluorophores=40, on_coverslip=0.3, seed=3,
                      blinking=BlinkingSettings(photons=4000.0, photons_std=0.0, on_time=2.0, blinks=2.0),
                      illumination_fwhm_px=30.0, spot_px=21)
    movie = simulate_movie(model, shared, s)
    t = movie.truth
    assert movie.frames.shape == (60, 32, 40) and len(t) > 40
    assert np.all(np.diff(t.frame) >= 0) and t.frame.max() < 60
    assert np.all((t.h_nm >= 0) & (t.h_nm <= 1500))
    on_glass = {f for f in np.unique(t.fluorophore) if t.h_nm[t.fluorophore == f][0] == 0.0}
    assert 0.1 < len(on_glass) / len(np.unique(t.fluorophore)) < 0.6
    # the frames of a blink are one emitter: one place, consecutive frames
    for b in np.unique(t.blink):
        r = t.blink == b
        assert len({(x, y, h) for x, y, h in zip(t.x_nm[r], t.y_nm[r], t.h_nm[r], strict=True)}) == 1
        assert np.all(np.diff(t.frame[r]) == 1)
    # the illumination scales each molecule's photons: no blink is brighter than its rate at the centre
    rel = illumination(s, t.x_nm / optics.pixel_size_nm[0], t.y_nm / optics.pixel_size_nm[1])
    assert np.all(t.photons <= 4000.0 / 2.0 * rel + 1e-9)
    # the camera: the noise is Poisson about the expected photons
    r = (movie.frames - movie.expected) / np.sqrt(movie.expected)
    assert abs(r.mean()) < 0.02 and abs(r.std() - 1) < 0.02
    assert np.isclose(movie.expected.min(), 10.0)


def test_close_pairs_are_molecules_on_together_within_a_distance():
    a = np.array
    t = MovieTruth(a([0, 0, 0, 1]), a([0, 1, 2, 3]), a([0, 1, 2, 3]), a([0.0, 300.0, 5000.0, 10.0]),
                   a([0.0, 0.0, 0.0, 0.0]), a([0.0] * 4), a([1.0] * 4))
    assert close_pairs(t, 500.0).tolist() == [True, True, False, False]
