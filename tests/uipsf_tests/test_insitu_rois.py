"""Finding, measuring and choosing single molecules in a blinking movie for in situ learning (M5, step 4).

The module is SMAPpy's (`smappy.calibrate.insitu`); its tests are here because the movies are uipsf's.
"""
import numpy as np
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
    near = dict(det_frame=np.array([0]), det_xy=np.array([[27.0, 20.0]]), det_sigma=np.array([1.5]),
                det_blink=np.array([1]), blink_id=0, settings=s)
    assert I.overlaps((20, 20), np.array([0]), 1.5, **near)                  # 7 px < 3·1.5 + 3·1.5
    assert not I.overlaps((20, 20), np.array([1]), 1.5, **near)              # on in another frame
    far = dict(det_frame=np.array([0]), det_xy=np.array([[29.0, 20.0]]), det_sigma=np.array([1.0]),
               det_blink=np.array([1]), blink_id=0, settings=s)
    assert not I.overlaps((20, 20), np.array([0]), 1.5, **far)
    mask = I.neighbour_mask((20, 20), np.array([0]), **far)
    assert mask[7, 7] and not mask[7, 14]                                    # its disc reaches the edge


def _units(frames, **kw):
    s = I.InSituSettings(**kw)
    kept, g = I.measure(frames, I.detect(frames, np.full_like(frames, 10.0), s), s)
    return I.units(frames, kept, g, s)


BLINK = [(f, 24.3, 23.8, IN_FOCUS + 200.0, 3000.0) for f in (2, 3, 4)] + [(7, 10.0, 12.0, IN_FOCUS, 3000.0)]


def test_each_frame_of_a_blink_is_a_region_of_its_own_and_knows_its_blink():
    frames = frames_with(BLINK, n_frames=10)
    u = _units(frames)
    assert sorted(f.tolist() for f in u.frames) == [[2], [3], [4], [7]]
    blink = {int(f[0]): b for f, b in zip(u.frames, u.blink, strict=True)}
    assert blink[2] == blink[3] == blink[4] != blink[7]
    i = [f[0] for f in u.frames].index(3)
    cx, cy = u.centre[i]
    assert np.array_equal(u.regions[i], frames[3, cy - 7:cy + 8, cx - 7:cx + 8])


def test_a_blinks_frames_summed_are_one_region_with_their_read_noise():
    frames = frames_with(BLINK, n_frames=10)
    u = _units(frames, unit="blink")
    assert sorted(len(f) for f in u.frames) == [1, 3]
    i = [len(f) for f in u.frames].index(3)
    cx, cy = u.centre[i]
    assert np.allclose(u.regions[i], frames[2:5, cy - 7:cy + 8, cx - 7:cx + 8].sum(0))
    batch, noise = I.InSituROIs(u, ["used"] * len(u.frames), np.zeros(len(u.frames), int), np.array([]), 0,
                                OPTICS.pixel_size_nm, read_noise_var=2.0).batch()
    assert batch.data.shape == (len(u.frames), 1, 1, 15, 15)
    assert sorted(batch.s.flatten().tolist()) == [2.0, 6.0]           # three frames: three times the read noise


def test_a_blink_whose_photons_step_between_inner_frames_is_rejected():
    spots = [(f, 24.0, 24.0, IN_FOCUS, n) for f, n in zip(range(1, 6), (1500.0, 4000.0, 1000.0, 4000.0, 1500.0),
                                                            strict=True)]
    u = _units(frames_with(spots, n_frames=7))
    assert u.reason == ["photon step"] * 5 and len(set(u.blink.tolist())) == 1


def test_a_unit_whose_own_fit_runs_wider_than_a_third_of_its_region_is_rejected(monkeypatch):
    # On Windows one unit's masked fit ran to σx 16 px (5 on macOS) and set the z proxy's range on its own.
    frames = frames_with(BLINK, n_frames=10)
    s = I.InSituSettings()
    kept, g = I.measure(frames, I.detect(frames, np.full_like(frames, 10.0), s), s)
    fit = I.gaussian

    def one_wide(regions):
        out = fit(regions)
        out["sigma_x_pix"][0] = s.roi_px / 3 + 0.5
        return out

    monkeypatch.setattr(I, "gaussian", one_wide)
    u = I.units(frames, kept, g, s)
    assert u.reason[0] == "wide" and u.reason[1:] == [None] * (len(u.reason) - 1)


def _fake_units(sx, sy, photons, blink=None):
    n = len(sx)
    gauss = {"sigma_x_pix": np.asarray(sx, float), "sigma_y_pix": np.asarray(sy, float),
             "photons": np.asarray(photons, float)}
    blink = np.arange(n) if blink is None else np.asarray(blink)
    return I.Units([np.array([0])] * n, blink, np.zeros(n), np.zeros(n), np.zeros((n, 2), int), np.zeros((n, 3, 3)),
                   np.ones((n, 3, 3), bool), gauss, [None] * n)


def test_selection_draws_from_a_brightness_window_below_the_bright_tail_and_holds_some_out():
    rng = np.random.default_rng(0)
    n = 2000
    sx, sy = 1.5 + 0.02 * rng.standard_normal(n), 1.5 + 0.02 * rng.standard_normal(n)
    N = rng.uniform(100, 3000, n)
    s = I.InSituSettings(bins=4, per_bin=50, proxy="astigmatism")
    state, bins, _ = I.select(_fake_units(sx, sy, N), s)
    state = np.array(state)
    drawn = np.isin(state, ["used", "held out"])
    assert drawn.sum() == 4 * 50 and 5 <= (state == "held out").sum() <= 40
    for k in range(4):
        top = np.quantile(N[(bins == k) & ~np.char.startswith(state, "rejected: shape")], 0.9)
        assert N[drawn & (bins == k)].max() <= top + 1e-9 and N[drawn & (bins == k)].min() >= top / 3 - 1e-9
    assert np.all(np.char.startswith(state[N > 2950], "rejected"))


def test_selection_rejects_a_spot_too_wide_for_its_bin():
    rng = np.random.default_rng(1)
    sx, sy = 1.5 + 0.02 * rng.standard_normal(200), 1.5 + 0.02 * rng.standard_normal(200)
    sx[7] = sy[7] = 2.4                          # round, so it falls in the same astigmatism bin, but wide
    state, _, _ = I.select(_fake_units(sx, sy, np.full(200, 1000.0)), I.InSituSettings(bins=1))
    assert state[7] == "rejected: shape"


def test_the_molecules_chosen_in_a_small_movie_are_molecules_at_every_height():
    # Learning from them is test_insitu's and the benchmark's (benchmarks/uipsf/synthetic/insitu_movie.py).
    from smappy.simulate import BlinkingSettings
    model, shared = model_and_shared(P=24)
    s = MovieSettings(size_px=(64, 64), n_frames=150, fluorophores=100, heights_nm=(0.0, 1400.0),
                      on_coverslip=0.2, illumination_fwhm_px=100.0, seed=1,
                      blinking=BlinkingSettings(photons=6000.0, photons_std=3000.0, on_time=1.5, blinks=3.0))
    movie = simulate_movie(model, shared, s)
    rois = I.insitu_rois(movie.frames, OPTICS.pixel_size_nm, I.InSituSettings(per_bin=3))
    used = rois.ids("used")
    assert len(used) > 25
    # each used unit is a molecule (a Gaussian truncated by the region misplaces the farthest by 2–4 px), and
    # the molecules used reach from the coverslip to well above the focus
    t, px, py = movie.truth, *OPTICS.pixel_size_nm
    d, h = [], []
    for i in used:
        on = np.flatnonzero(t.frame == rois.units.frames[i][0])
        dist = np.hypot(t.x_nm[on] / px - rois.units.x_px[i], t.y_nm[on] / py - rois.units.y_px[i])
        d.append(dist.min())
        h.append(t.h_nm[on[np.argmin(dist)]])
    assert np.mean(np.array(d) < 3) > 0.9 and max(d) < 5
    assert min(h) < 200 and max(h) > 1100


def test_a_molecules_frames_are_held_out_together():
    rng = np.random.default_rng(2)
    n = 900
    sx, sy = 1.5 + 0.02 * rng.standard_normal(n), 1.5 + 0.02 * rng.standard_normal(n)
    state, _, _ = I.select(_fake_units(sx, sy, np.full(n, 1000.0), blink=np.arange(n) // 3),
                           I.InSituSettings(bins=1, per_bin=n, window_top=1.0))
    state = np.array(state).reshape(-1, 3)
    held, used = (state == "held out").any(1), (state == "used").any(1)
    assert held.any() and not (held & used).any()


def test_a_neighbour_much_dimmer_than_the_molecule_neither_rejects_nor_masks_it():
    s = I.InSituSettings()
    near = dict(det_frame=np.array([0]), det_xy=np.array([[27.0, 20.0]]), det_sigma=np.array([1.5]),
                det_blink=np.array([1]), blink_id=0, settings=s)
    assert not I.overlaps((20, 20), np.array([0]), 1.5, **near, det_photons=np.array([100.0]), photons=1000.0)
    assert I.overlaps((20, 20), np.array([0]), 1.5, **near, det_photons=np.array([500.0]), photons=1000.0)
    far = dict(near, det_xy=np.array([[29.0, 20.0]]), det_sigma=np.array([1.0]))
    assert I.neighbour_mask((20, 20), np.array([0]), **far, det_photons=np.array([100.0]), photons=1000.0).all()
    assert not I.neighbour_mask((20, 20), np.array([0]), **far, det_photons=np.array([500.0]), photons=1000.0).all()


def test_a_region_keeps_only_the_molecules_inside_it_and_judges_them_against_the_whole_frame():
    # One channel of a split chip: two molecules left, one right, the region the right half.  The left ones
    # are out for the region, and nothing else changes for the right one.
    frames = frames_with([(0, 10, 24, IN_FOCUS, 3000.0), (0, 14, 12, IN_FOCUS, 3000.0),
                          (0, 36, 24, IN_FOCUS, 3000.0)])
    bg = np.full_like(frames, 10.0)
    whole, _ = I.insitu_units(frames, I.InSituSettings(), background=bg)
    half, _ = I.insitu_units(frames, I.InSituSettings(region=(24, 0, 24, 48)), background=bg)
    right = half.x_px >= 24
    assert right.sum() == 1 and len(half.reason) == len(whole.reason) == 3
    assert all(r == "outside the region" for r in np.array(half.reason, object)[~right])
    assert half.reason[int(np.flatnonzero(right)[0])] == whole.reason[int(np.flatnonzero(whole.x_px >= 24)[0])]
