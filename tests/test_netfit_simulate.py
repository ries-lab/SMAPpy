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
    sim = SimulationSettings(emitters_per_frame=20.0, lifetime_frames=2.0, window=5)
    g = torch.Generator().manual_seed(0)
    *_, flux, per_frame = simulate.emitters(sim, 3000, g, "cpu")
    on_mid = (per_frame[:, 2] > 0).sum().item() / 3000
    assert on_mid == pytest.approx(20.0, rel=0.03)
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
    sim = SimulationSettings(frame_px=24, emitters_per_frame=5.0, photons_mean=2000.0, photons_sd=0.0)
    g = torch.Generator().manual_seed(3)
    b = simulate.draw_batch(SplineRenderer(cal), sim, CameraModel(), 16, g)
    assert b.frames.shape == (16, 3, 24, 24)
    xyz = b.xyz[b.mask]
    assert len(xyz) > 16
    assert torch.all((xyz[:, :2] >= -0.5) & (xyz[:, :2] < 23.5))
    assert torch.all(b.photons[b.mask] > 0) and torch.all(b.photons[b.mask] <= 2000.0 + 1e-3)
    assert torch.all((b.background >= 20.0) & (b.background <= 200.0))
