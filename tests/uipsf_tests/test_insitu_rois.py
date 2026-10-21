"""Finding, measuring and choosing single molecules in a blinking movie for in situ learning (M5, step 4).

The module is SMAPpy's (`smappy.calibrate.insitu`); its tests are here because the movies are uipsf's.
"""
import numpy as np
import pytest
import torch
from smappy.calibrate import insitu as I

import uipsf
from uipsf.model import Bleaching, InSituGeometry
from uipsf.movie import MovieSettings, MovieTruth, render, simulate_movie

f64 = torch.float64
OPTICS = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.335, pixel_size_nm=(127, 116))
FOCUS = 700.0
IN_FOCUS = FOCUS * OPTICS.n_med / OPTICS.n_imm


def model_and_shared(P=32, roi=(15, 15)):
    grid = uipsf.PupilGrid.build(OPTICS, P, device="cpu", dtype=f64)
    pupil = uipsf.ZernikePupil(grid, n_max=4)
    model = uipsf.ForwardModel(uipsf.VectorPSF(grid, pupil, roi, sigma_nm=(30.0, 30.0)),
                               geometry=InSituGeometry(600.0), intensity=Bleaching(order=0))
    phase = torch.zeros(len(pupil.phase_modes), dtype=f64)
    phase[:5] = torch.tensor([0.1, 0.7, -0.2, 0.15, 0.1])
    shared = {"pupil.phase": phase, "pupil.magnitude": torch.zeros(len(pupil.magnitude_modes), dtype=f64),
              "psf.sigma": torch.tensor([35.0, 35.0], dtype=f64), "geometry.focus": torch.tensor(FOCUS, dtype=f64)}
    return model, shared


def frames_with(spots, size=(48, 48), n_frames=1, background=10.0, seed=0, noise=True):
    """Frames with molecules (frame, x_px, y_px, h_nm, photons) rendered by the model."""
    model, shared = model_and_shared()
    a = np.array
    px, py = OPTICS.pixel_size_nm
    t = MovieTruth(a([s[0] for s in spots]), a(list(range(len(spots)))), a(list(range(len(spots)))),
                   a([s[1] * px for s in spots]), a([s[2] * py for s in spots]), a([float(s[3]) for s in spots]),
                   a([float(s[4]) for s in spots]))
    image = np.full((n_frames, *size), background)
    render(model, shared, t, image, 31)
    return np.random.default_rng(seed).poisson(image).astype(np.float32) if noise else image.astype(np.float32)


def test_the_temporal_background_does_not_see_a_blink():
    frames = np.full((30, 8, 8), 10.0, np.float32)
    frames[10:13, 3:5, 3:5] += 500.0
    assert np.allclose(I.temporal_background(frames, 21), 10.0)


def test_a_defocused_molecule_is_found_on_its_significance_where_a_relative_threshold_loses_it():
    # Two molecules of 3000 photons, one in focus, one 700 nm below it: the defocused one's DoG peak is a
    # fraction of the other's, below v1's 0.2-0.3 of the frame's maximum, but its photons are as many.
    frames = frames_with([(0, 14, 24, IN_FOCUS, 3000.0), (0, 34, 24, IN_FOCUS + 700.0, 3000.0)])
    bg = np.full_like(frames, 10.0)
    s = I.InSituSettings()
    c = I.detect(frames, bg, s)
    kept, g = I.measure(frames, c, s)
    assert len(kept) == 2 and sorted(np.rint(g["x_px"] / 10).astype(int).tolist()) == [1, 3]
    from smappy.detect import DoGFilter
    dog = DoGFilter(1.2)(frames - bg)[0]
    assert dog[20:29, 30:39].max() < 0.2 * dog[20:29, 10:19].max()


def test_a_defocused_spots_several_maxima_are_one_molecule():
    frames = frames_with([(0, 24, 24, IN_FOCUS - 600.0, 6000.0)], noise=False)
    s = I.InSituSettings()
    c = I.detect(frames, np.full_like(frames, 10.0), s)
    kept, g = I.measure(frames, c, s)
    assert len(c) > 1 and len(kept) == 1


def test_a_neighbour_within_both_spots_sizes_rejects_and_a_farther_one_is_masked():
    s = I.InSituSettings()
    reason, mask = I.overlaps((20, 20), np.array([0]), 1.5, np.array([0]), np.array([[27.0, 20.0]]),
                              np.array([1.5]), np.array([1]), 0, s)
    assert reason == "overlap"                         # 7 px < 3·1.5 + 3·1.5
    reason, mask = I.overlaps((20, 20), np.array([0]), 1.5, np.array([0]), np.array([[29.0, 20.0]]),
                              np.array([1.0]), np.array([1]), 0, s)
    assert reason is None and mask[7, 7] and not mask[7, 14]          # its disc reaches the region's edge
    reason, _ = I.overlaps((20, 20), np.array([1]), 1.5, np.array([0]), np.array([[27.0, 20.0]]),
                           np.array([1.5]), np.array([1]), 0, s)
    assert reason is None                              # the neighbour is on in another frame
    reason, _ = I.overlaps((20, 20), np.array([0]), 1.0, np.array([0, 0]), np.array([[29.0, 20.0], [11.0, 20.0]]),
                           np.array([2.0, 2.0]), np.array([1, 2]), 0, I.InSituSettings(max_masked=0.1))
    assert reason == "masked"


def test_a_blinks_frames_are_linked_and_summed_into_one_region():
    spots = [(f, 24.3, 23.8, IN_FOCUS + 200.0, 3000.0) for f in (2, 3, 4)] + [(7, 10.0, 12.0, IN_FOCUS, 3000.0)]
    frames = frames_with(spots, n_frames=10)
    s = I.InSituSettings()
    c = I.detect(frames, np.full_like(frames, 10.0), s)
    kept, g = I.measure(frames, c, s)
    b = I.blinks(frames, kept, g, s)
    assert sorted(len(f) for f in b.frames) == [1, 3]
    i = [len(f) for f in b.frames].index(3)
    cx, cy = b.centre[i]
    assert np.allclose(b.regions[i], frames[2:5, cy - 7:cy + 8, cx - 7:cx + 8].sum(0))
    batch, noise = I.InSituROIs(b, ["used"] * len(b.frames), np.zeros(len(b.frames), int), np.array([]), len(c),
                                OPTICS.pixel_size_nm, read_noise_var=2.0).batch()
    assert batch.data.shape == (len(b.frames), 1, 1, 15, 15)
    assert sorted(batch.s.flatten().tolist()) == [2.0, 6.0]           # three frames: three times the read noise


def test_a_blink_whose_photons_step_between_inner_frames_is_rejected():
    spots = [(f, 24.0, 24.0, IN_FOCUS, n) for f, n in zip(range(1, 6), (1500.0, 4000.0, 1000.0, 4000.0, 1500.0),
                                                            strict=True)]
    frames = frames_with(spots, n_frames=7)
    s = I.InSituSettings()
    kept, g = I.measure(frames, I.detect(frames, np.full_like(frames, 10.0), s), s)
    b = I.blinks(frames, kept, g, s)
    assert b.reason == ["photon step"]


def _fake_blinks(sx, sy, photons):
    n = len(sx)
    gauss = {"sigma_x_pix": np.asarray(sx, float), "sigma_y_pix": np.asarray(sy, float),
             "photons": np.asarray(photons, float)}
    return I.Blinks([np.array([0])] * n, np.zeros(n), np.zeros(n), np.zeros((n, 2), int), np.zeros((n, 3, 3)),
                    np.ones((n, 3, 3), bool), gauss, [None] * n)


def test_selection_draws_from_a_brightness_window_below_the_bright_tail_and_holds_some_out():
    rng = np.random.default_rng(0)
    n = 2000
    sx, sy = 1.5 + 0.02 * rng.standard_normal(n), 1.5 + 0.02 * rng.standard_normal(n)
    N = rng.uniform(100, 3000, n)
    s = I.InSituSettings(bins=4, per_bin=50, proxy="astigmatism")
    state, bins, _ = I.select(_fake_blinks(sx, sy, N), s)
    state = np.array(state)
    drawn = np.isin(state, ["used", "held out"])
    assert drawn.sum() == 4 * 50 and (state == "held out").sum() == 4 * 5
    for k in range(4):
        top = np.quantile(N[(bins == k) & ~np.char.startswith(state, "rejected: shape")], 0.9)
        assert N[drawn & (bins == k)].max() <= top + 1e-9 and N[drawn & (bins == k)].min() >= top / 3 - 1e-9
    assert np.all(np.char.startswith(state[N > 2950], "rejected"))


def test_selection_rejects_a_spot_too_wide_for_its_bin():
    rng = np.random.default_rng(1)
    sx, sy = 1.5 + 0.02 * rng.standard_normal(200), 1.5 + 0.02 * rng.standard_normal(200)
    sx[7] = sy[7] = 2.4                          # round, so it falls in the same astigmatism bin, but wide
    state, _, _ = I.select(_fake_blinks(sx, sy, np.full(200, 1000.0)), I.InSituSettings(bins=1))
    assert state[7] == "rejected: shape"


@pytest.mark.slow
def test_a_pupil_is_learnt_from_the_molecules_chosen_in_a_simulated_movie():
    # 600 molecules 0-1400 nm deep, 600 frames, a Gaussian illumination; ~70 s on one core.
    from smappy.simulate import BlinkingSettings

    from uipsf.workflows.insitu import learn_insitu
    model, shared = model_and_shared()
    s = MovieSettings(size_px=(128, 128), n_frames=600, fluorophores=600, heights_nm=(0.0, 1400.0),
                      on_coverslip=0.1, illumination_fwhm_px=160.0, seed=1,
                      blinking=BlinkingSettings(photons=6000.0, photons_std=3000.0, on_time=1.5, blinks=3.0))
    movie = simulate_movie(model, shared, s)
    rois = I.insitu_rois(movie.frames, OPTICS.pixel_size_nm, I.InSituSettings(per_bin=40))
    used = rois.ids("used")
    assert len(used) > 120
    # every blink used is a molecule: near one on in its first frame.  A Gaussian truncated by the region
    # misplaces the spots 600-700 nm from focus (σ 4-5 px) by 2-4 px, so 95 % within 3 px, all within 5.
    t, px, py = movie.truth, *OPTICS.pixel_size_nm
    d = []
    for i in used:
        on = t.frame == rois.blinks.frames[i][0]
        d.append(np.hypot(t.x_nm[on] / px - rois.blinks.x_px[i], t.y_nm[on] / py - rois.blinks.y_px[i]).min())
    assert np.mean(np.array(d) < 3) > 0.95 and max(d) < 5
    batch, noise = rois.batch()
    start = torch.zeros_like(shared["pupil.phase"])
    start[1] = 0.4
    result = learn_insitu(model, batch.to(dtype=f64), start={"pupil.phase": start}, noise=noise)
    fit, learnt = result.calibration.result, result.calibration.shared()
    assert torch.all((learnt["pupil.phase"] - shared["pupil.phase"]).abs() < 4 * fit.std["pupil.phase"])
    assert abs(float(learnt["geometry.focus"]) - FOCUS) < 3 * float(fit.std["geometry.focus"])
    assert result.twin.verdict == "agree"
