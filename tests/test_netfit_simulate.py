"""netfit's simulation: the torch spline draws what SMAPpy's simulation draws, and the samples are
what the settings ask for."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from smappy.io.calibration import SplineCalibration
from smappy.netfit import simulate
from smappy.netfit.render import SplineRenderer
from smappy.netfit.settings import CameraModel, SimulationSettings
from smappy.simulate.camera import _draw_spline


def _calibration(seed=0):
    # random coefficients: not a PSF, but every coefficient and every monomial is exercised
    rng = np.random.default_rng(seed)
    coeff = rng.normal(0, 1, (64, 21, 14, 14)).astype(np.float32)
    return SplineCalibration(coeff=coeff, dz=50.0, z0=10.0, x0=6.5, em_on=False)


def test_the_torch_spline_draws_what_smappys_simulation_draws_to_float_precision():
    cal = _calibration()
    rng = np.random.default_rng(1)
    n = 200
    frame = rng.integers(0, 4, n)
    # beyond the frame's edges and the spline's z range too: the clamping must match
    x, y = rng.uniform(-4, 36, n), rng.uniform(-4, 36, n)
    z, photons = rng.uniform(-600, 600, n), rng.uniform(100, 1000, n)
    ref = np.zeros((4, 32, 32))
    _draw_spline(ref, frame, np.stack([x, y], 1), z, photons, cal)
    out = torch.zeros((4, 32, 32))
    t = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d)
    SplineRenderer(cal).draw(out, t(frame, torch.long), t(x), t(y), t(z), t(photons))
    assert np.abs(out.numpy() - ref).max() < 1e-5 * np.abs(ref).max()


def test_a_frame_holds_the_molecules_per_frame_asked_for_and_none_gives_a_frame_more_than_its_flux():
    sim = SimulationSettings(density_per_40px=20.0, lifetime_frames=2.0, window=5)
    g = torch.Generator().manual_seed(0)
    *_, flux, per_frame = simulate.emitters(sim, 3000, g, "cpu")
    on_mid = (per_frame[:, 2] > 0).sum().item() / 3000
    assert on_mid == pytest.approx(20.0, rel=0.03)
    # a density: a frame of 80 x 80 pixels holds four times as many
    big = SimulationSettings(frame_px=80, density_per_40px=20.0, lifetime_frames=2.0, window=5)
    *_, per_big = simulate.emitters(big, 1000, g, "cpu")
    assert (per_big[:, 2] > 0).sum().item() / 1000 == pytest.approx(80.0, rel=0.03)
    assert torch.all(per_frame <= flux[:, None] * (1 + 1e-6))
    # on for whole frames on average: the photons a frame gets are on average flux * (on-time share)
    full = (per_frame > 0.999 * flux[:, None]).sum().item()
    assert full > 0


def test_the_camera_gives_back_the_photons_with_the_em_registers_excess_noise():
    g = torch.Generator().manual_seed(2)
    lam = torch.full((200_000,), 50.0)
    em = simulate.camera(lam, CameraModel(em_gain=100.0, read_noise_e=0.0), g)
    assert em.mean().item() == pytest.approx(50.0, rel=0.01)
    # rounding to ADU adds (conversion / gain)^2 / 12, negligible here
    assert (em.var() / em.mean()).item() == pytest.approx(2.0, rel=0.03)
    plain = simulate.camera(lam, CameraModel(em_gain=0.0, conversion=1.0, read_noise_e=0.0), g)
    assert (plain.var() / plain.mean()).item() == pytest.approx(1.0, rel=0.03)


def test_a_batch_is_the_frames_the_settings_ask_for_and_its_targets_are_in_the_middle_frame():
    cal = _calibration()
    sim = SimulationSettings(frame_px=24, density_per_40px=5.0, photons_mean=2000.0, photons_sd=0.0)
    g = torch.Generator().manual_seed(3)
    b = simulate.draw_batch(SplineRenderer(cal), sim, CameraModel(), 16, g)
    assert b.frames.shape == (16, 3, 24, 24)
    xyz = b.xyz[b.mask]
    assert len(xyz) > 16
    assert torch.all((xyz[:, :2] >= -0.5) & (xyz[:, :2] < 23.5))
    assert torch.all(b.photons[b.mask] > 0) and torch.all(b.photons[b.mask] <= 2000.0 + 1e-3)
    assert torch.all((b.background >= 20.0) & (b.background <= 200.0))


def test_a_range_of_on_times_gives_each_sample_its_own_mean_and_keeps_the_molecules_per_frame():
    # a sample's molecules are on for an exponential time of its own mean, log-uniform in 1-3 frames
    sim = SimulationSettings(density_per_40px=20.0, lifetime_frames=1.0, lifetime_max_frames=3.0, window=7)
    g = torch.Generator().manual_seed(1)
    *_, per_frame = simulate.emitters(sim, 400, g, "cpu")
    assert (per_frame[:, 3] > 0).sum().item() / 400 == pytest.approx(20.0, rel=0.04)

    def means(s):
        # 300 frames, so few on-times are cut by the ends: each sample's mean on-time, in frames
        sample, *_, flux, pf = simulate.emitters(s, 200, torch.Generator().manual_seed(2), "cpu", frames=300)
        on = pf.sum(1) / flux
        whole = (pf[:, 0] == 0) & (pf[:, -1] == 0) & (on > 0)
        return torch.zeros(200).index_add_(0, sample[whole], on[whole]) / torch.bincount(sample[whole], minlength=200)

    m = means(sim)
    assert torch.quantile(m, 0.1) < 1.3 and torch.quantile(m, 0.9) > 2.4      # 3^0.1 = 1.12, 3^0.9 = 2.69
    assert m.min() > 0.85 and m.max() < 3.3
    # without a maximum, every sample at the one mean
    assert torch.all((means(SimulationSettings(density_per_40px=20.0, lifetime_frames=2.0)) - 2.0).abs() < 0.2)


def test_windows_cut_from_one_movie_share_its_frames_shifted_by_one_and_target_their_own_middle_frame():
    cal = _calibration()
    sim = SimulationSettings(frame_px=24, window=5, density_per_40px=8.0, photons_sd=0.0, movie_windows=4)
    torch.manual_seed(0)
    b = simulate.draw_batch(SplineRenderer(cal), sim, CameraModel(), 10, torch.Generator().manual_seed(2))
    assert b.frames.shape == (10, 5, 24, 24)            # 3 movies of 8 frames, the last window cut
    for m in range(2):
        for w in range(3):
            i = 4 * m + w
            assert torch.equal(b.frames[i, 1:], b.frames[i + 1, :-1])
            assert torch.equal(b.background_window[i, 1:], b.background_window[i + 1, :-1])
    assert not torch.equal(b.frames[3, 1:], b.frames[4, :-1])     # another movie
    # the movies come along for the network, which reads them whole: window i is movie i // 4 from frame i % 4
    assert b.movie.shape == (3, 8, 24, 24) and b.movie_windows == 4
    for i in range(10):
        assert torch.equal(b.frames[i], b.movie[i // 4, i % 4:i % 4 + 5])
    # a window's targets are the molecules in its middle frame: a molecule in frames 2-3 of window 0
    # is in frames 1-2 of window 1, and its target photons are its middle frame's
    for i in range(10):
        on = b.mask[i]
        assert torch.allclose(b.photons[i, on], b.photons_window[i, on, 2])
        assert torch.all(b.photons_window[i, on, 2] > 0)
    w0 = {tuple(v) for v in b.xyz[0, b.mask[0]].tolist()}
    w1 = {tuple(v) for v in b.xyz[1, b.mask[1]].tolist()}
    both = w0 & w1                                       # molecules on in frame 2 and 3 of the movie
    assert both, "some molecules span two middle frames at this density"
    for v in both:
        a = next(j for j in range(b.mask.shape[1]) if b.mask[0, j] and tuple(b.xyz[0, j].tolist()) == v)
        c = next(j for j in range(b.mask.shape[1]) if b.mask[1, j] and tuple(b.xyz[1, j].tolist()) == v)
        assert torch.equal(b.photons_window[0, a, 1:], b.photons_window[1, c, :-1])


def test_a_background_per_movie_is_one_level_jittered_by_the_fraction_asked():
    cal = _calibration()
    sim = SimulationSettings(frame_px=16, window=7, movie_windows=4, density_per_40px=1.0, background_jitter=0.02)
    b = simulate.draw_batch(SplineRenderer(cal), sim, CameraModel(), 64, torch.Generator().manual_seed(5))
    rel = b.background_window / b.background_window.mean(1, keepdim=True) - 1
    assert rel.std().item() == pytest.approx(0.02, rel=0.15)
    assert b.background_window.mean(1).std().item() > 30           # the levels still span 20-200
    jumpy = simulate.draw_batch(SplineRenderer(cal), SimulationSettings(frame_px=16, window=7), CameraModel(), 64,
                                torch.Generator().manual_seed(5))
    assert (jumpy.background_window / jumpy.background_window.mean(1, keepdim=True) - 1).std().item() > 0.3


def test_an_uneven_background_is_a_smooth_field_that_keeps_the_mean_and_is_the_target():
    """plex-plan §16 (2026-11-22): a share of the movies gets a smooth field, fixed over the movie, the
    level times 1 + c (2 f - 1); the map is the target, and the frames show it."""
    cal = _calibration()
    sim = SimulationSettings(frame_px=32, window=3, movie_windows=2, density_per_40px=0.0, background=(100.0, 100.0),
                             background_uneven=0.5, background_contrast=0.8)
    b = simulate.draw_batch(SplineRenderer(cal), sim, CameraModel(em_gain=0.0), 400, torch.Generator().manual_seed(2))
    m = b.background_map[:, 0]
    uneven = m.flatten(1).std(1) > 0
    assert 0.4 < uneven.float().mean().item() < 0.6
    assert torch.all(m[~uneven] == 100.0)
    assert m.mean().item() == pytest.approx(100.0, rel=0.03)              # the field keeps the level
    assert 20.0 - 1e-3 <= m.min().item() and m.max().item() <= 180.0 + 1e-3  # contrast at most 0.8
    # smooth: neighbouring pixels differ far less than the field's spread
    step = (m[uneven, :, 1:] - m[uneven, :, :-1]).abs().mean()
    assert step < 0.1 * m[uneven].std()
    # the frames show it: the middle frame's pixels follow the map, the camera's noise around it
    mid = b.frames[:, 1]
    assert (mid - m).mean().abs().item() < 1.0
    smooth = torch.nn.functional.avg_pool2d(mid[uneven][:, None], 8)[:, 0]
    assert np.corrcoef(smooth.flatten(), torch.nn.functional.avg_pool2d(m[uneven][:, None], 8).flatten())[0, 1] > 0.9
    # fixed over the movie: two windows of one movie share their map
    assert torch.equal(b.background_map[0], b.background_map[1])


def test_two_channels_share_most_of_the_field_and_the_choice_to_be_uneven():
    cal = _calibration()
    sim = SimulationSettings(frame_px=32, background_uneven=0.5)
    m = simulate.background_modulation(sim, 300, 2, 32, torch.Generator().manual_seed(3), "cpu")
    flat = m.flatten(2).std(2) == 0                                            # (n, 2)
    assert torch.equal(flat[:, 0], flat[:, 1])
    a, b = m[~flat[:, 0], 0].flatten(1), m[~flat[:, 0], 1].flatten(1)
    r = [np.corrcoef(a[i], b[i])[0, 1] for i in range(len(a))]
    assert 0.6 < np.median(r) < 0.98                                           # mostly shared, not identical
    del cal

