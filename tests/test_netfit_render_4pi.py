"""The 4Pi IAB model in torch (`netfit.render_4pi`): the spots Spline 4Pi fits, to float precision,
on every device, and the phase convention settled on real beads (docs/4pi.md §30)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from smappy import fourpi
from smappy.channels import from_corners
from smappy.netfit.render_4pi import IABRenderer
from smappy.psf import Global4PiPSF


@pytest.fixture(scope="module")
def cal():
    return fourpi.model_calibration()


def _emitters(renderer, m, seed=0, shift=0.0):
    """m spots near the square's centre, each channel a sub-pixel ``shift`` at most from channel 0."""
    rng = np.random.default_rng(seed)
    c0 = (renderer.size - 1) / 2
    x = c0 + rng.uniform(-1, 1, (m, 1)) + rng.uniform(-shift, shift, (m, renderer.n_channels))
    y = c0 + rng.uniform(-1, 1, (m, 1)) + rng.uniform(-shift, shift, (m, renderer.n_channels))
    z = rng.uniform(-500, 500, m)
    phase = rng.uniform(-np.pi, np.pi, m)
    return x, y, z, phase


def _t(*arrays):
    return [torch.as_tensor(a, dtype=torch.float64) for a in arrays]


def test_the_torch_model_is_the_numpy_one_beyond_the_grid_too(cal):
    """Positions, z beyond the spline's grid at both ends and phases beyond pi: the torch spots are
    `fourpi.spots`', whose clamping of the voxel and not the fraction is the fitter's."""
    r = IABRenderer(cal, dtype=torch.float64)
    x, y, _, _ = _emitters(r, 200)
    x[:] = x[:, :1]
    y[:] = y[:, :1]
    rng = np.random.default_rng(1)
    z = rng.uniform(-800, 800, 200)                  # the grid holds +-600 nm
    phase = rng.uniform(-8, 8, 200)
    spots, ox, oy = r.spots(*_t(x, y, z, phase), clamp=False)
    ref = fourpi.spots(cal, x[:, 0] - ox[:, 0].numpy(), y[:, 0] - oy[:, 0].numpy(),
                       cal.z_nm_to_index(z), phase, r.size)
    for c in range(cal.n_channels):
        assert np.abs(spots[:, c].numpy() - ref[:, c]).max() < 1e-6 * np.abs(ref).max()


def test_spots_drawn_in_torch_are_fitted_by_spline_4pi_back_to_their_truth(cal):
    """The C++ fitter, given noise-free torch spots of 10^5 photons in channels shifted against each
    other by fractions of a pixel (the link carries the shifts), finds every parameter as drawn: the
    two models are one to the fitter's float precision."""
    r = IABRenderer(cal, dtype=torch.float64)
    m = 300
    x, y, z, phase = _emitters(r, m, seed=2, shift=0.4)
    N, bg = 1e5, 50.0
    spots, ox, oy = r.spots(*_t(x, y, z, phase))
    rois = (N * spots.numpy() + bg).astype(np.float32)
    xr, yr = x - ox.numpy(), y - oy.numpy()            # each channel's ROI coordinates
    link = fourpi.link(m, cal.n_channels, dx=xr - xr[:, :1], dy=yr - yr[:, :1])
    model = Global4PiPSF(cal, phase_starts=2)
    v = model.unpack(model.fit(rois, link, iterations=200))
    assert np.abs(v["x_roi"] - xr[:, 0]).max() < 2e-3
    assert np.abs(v["y_roi"] - yr[:, 0]).max() < 2e-3
    assert np.abs(v["z_nm"] - z).max() < 0.1
    assert np.abs(np.angle(np.exp(1j * (v["phase_rad"] - phase)))).max() < 1e-3
    assert np.abs(v["photons"] / N - 1).max() < 1e-3


def test_frames_drawn_on_the_channel_images_are_fourpis_frames_cut_by_the_layout(cal):
    """Molecules drawn at each channel's position as the layout's registration puts them, in each
    channel's image: the frames `fourpi.add_frames` makes, mirrored channels and a stretched one
    included, once the layout has cut them."""
    layout = from_corners([0, 70, 140, 210], [0] * 4, 64, 64, flip_x=[0, 0, 1, 1])
    m3 = np.eye(3)
    m3[0, 2], m3[1, 2], m3[0, 0] = 0.6, -0.4, 1.002
    layout = layout.with_maps([np.eye(3), np.eye(3), m3, np.eye(3)])
    rng = np.random.default_rng(3)
    n, k = 5, 40
    frame = rng.integers(0, n, k)
    xy = rng.uniform(-3, 67, (k, 2))                 # some spots cut by the channels' edges
    z = rng.uniform(-500, 500, k)
    photons = rng.uniform(500, 5000, k)
    phase0 = rng.uniform(-np.pi, np.pi, k)
    ref = np.zeros((n,) + fourpi.frame_size(layout), np.float32)
    fourpi.add_frames(ref, cal, layout, frame, xy[:, 0], xy[:, 1], z, photons, phase0)
    ref = layout.cut(ref)

    r = IABRenderer(cal)
    at = np.stack([xy if c == 0 else layout.to_channel(c, xy) for c in range(4)], axis=1)
    frames = torch.zeros((n, 4, 64, 64))
    zt = torch.as_tensor(z, dtype=torch.float32)
    r.draw(frames, torch.as_tensor(frame), torch.as_tensor(at[..., 0], dtype=torch.float32),
           torch.as_tensor(at[..., 1], dtype=torch.float32), zt,
           r.phase_of(zt, torch.as_tensor(phase0, dtype=torch.float32)),
           torch.as_tensor(photons, dtype=torch.float32))
    assert np.abs(frames.numpy() - ref).max() < 1e-4 * ref.max()


def _devices():
    out = []
    if torch.backends.mps.is_available():
        out.append("mps")
    if torch.cuda.is_available():
        out.append("cuda")
    return out


@pytest.mark.parametrize("device", _devices() or [pytest.param("none", marks=pytest.mark.skip(
    reason="no GPU here"))])
def test_a_gpu_draws_what_the_cpu_draws(cal, device):
    """In float32 on MPS or CUDA, the spots the CPU draws (MPS has no float64)."""
    cpu, gpu = IABRenderer(cal), IABRenderer(cal, device=device)
    x, y, z, phase = (torch.as_tensor(a, dtype=torch.float32) for a in _emitters(cpu, 500, seed=4, shift=0.4))
    a, _, _ = cpu.spots(x, y, z, phase)
    b, _, _ = gpu.spots(x.to(device), y.to(device), z.to(device), phase.to(device))
    assert torch.abs(b.cpu() - a).max() < 1e-5 * a.max()


def _v1_model():
    import os
    from pathlib import Path
    root = Path(os.environ.get("UIPSF_DATA", Path.home() / ".cache/uipsf/zenodo"))
    path = root / "results/4pi_40nm_bead/psfmodel_zernike_4pi.h5"
    if not path.exists():
        pytest.skip("uiPSF's 4Pi bead model is not downloaded "
                    "(scripts/uipsf/fetch_zenodo.py results/4pi_40nm_bead/)")
    return path


@pytest.mark.slow
def test_real_beads_fit_with_no_phase_offset_per_channel_beyond_a_and_b():
    """The phase convention, settled: uiPSF v1's beads, fitted with its model as SMAPpy reads it and
    with each channel's interference turned by 0, pi/2, pi, 3 pi/2 more (DECODE_Internal's 4Pi branch,
    `psf_kernel.py:1812`).  SMAPpy's fits every ROI better and gives back the mirrors' phase steps and
    the period; the branch's counts the channels' offsets twice."""
    import h5py
    path = _v1_model()
    cal = fourpi.load_uipsf_v1(path, transpose=False)
    with h5py.File(path, "r") as f:
        data = f["rois/psf_data"][...]
    C, nb, nph, nz, size, _ = data.shape
    o = (size - 15) // 2
    rois = np.moveaxis(data[..., o:o + 15, o:o + 15], 0, 3).reshape(-1, C, 15, 15).astype(np.float32)
    I, A, B = (cal.volumes[:, k].astype(float) for k in range(3))
    off = (np.arange(C) * np.pi / 2)[:, None, None, None]
    branch = fourpi.from_volumes(I, A * np.cos(off) + B * np.sin(off), B * np.cos(off) - A * np.sin(off),
                                 dz=cal.dz, z0=cal.z0, period_nm=cal.period_nm,
                                 phase_per_nm=cal.phase_per_nm)
    ours, theirs = (Global4PiPSF(c, phase_starts=4).fit(rois, iterations=100) for c in (cal, branch))
    assert np.all(ours.logl > theirs.logl)
    phase = Global4PiPSF(cal).unpack(ours)["phase_rad"].reshape(nb, nph, nz)
    steps = np.median(np.angle(np.exp(1j * (phase[:, 1:] - phase[:, :1]))), axis=(0, 2))
    assert steps == pytest.approx([2.0, -2.0], abs=0.15)
