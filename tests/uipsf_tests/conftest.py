"""uipsf's fixtures.  The tiers are SMAPpy's root conftest's: ``--slow`` adds benchmarks and end-to-end
runs, ``--gpu`` the checks that need CUDA or Apple's GPU (run by hand on a Mac and a CUDA machine)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))      # tests/uipsf_tests/oracles

import uipsf  # noqa: E402

# One thread per xdist worker: four workers with four threads each are slower than one.
if os.environ.get("PYTEST_XDIST_WORKER"):
    torch.set_num_threads(1)


@pytest.fixture
def optics():
    """The review's synthetic microscope: NA 1.43 oil objective, 680 nm, water, M2's pixels."""
    return uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.335, pixel_size_nm=(127, 116))


def make_psf(optics, P=64, roi=(21, 21), dtype=torch.float64, device="cpu", n_max=6, **kw):
    """A VectorPSF and its default physical parameters."""
    grid = uipsf.PupilGrid.build(optics, P, device=device, dtype=dtype)
    pupil = uipsf.ZernikePupil(grid, n_max=n_max)
    psf = uipsf.VectorPSF(grid, pupil, roi, **kw)
    return psf, uipsf.Params(psf.specs(), dtype=dtype, device=device).physical()


def stack(z_nm, x=0.0, y=0.0, n=1, dtype=torch.float64):
    """Positions [n, K, 3] of an emitter at (x, y) through the defocus values z_nm."""
    z = torch.as_tensor(z_nm, dtype=dtype)
    xyz = torch.zeros(n, len(z), 3, dtype=dtype)
    xyz[..., 0], xyz[..., 1], xyz[..., 2] = x, y, z
    return xyz


def small_beads(optics, n=5, dtype=torch.float64, seed=0, P=24, roi=(9, 9), K=9, photons=4000.0,
                background=10.0, **model_kw):
    """A small simulated bead problem: (model, truth Params, Batch).  Aberrated, so every block is
    informative, and small enough for float64 tests on the CPU."""
    from uipsf.simulate import parameters, simulate, template
    grid = uipsf.PupilGrid.build(optics, P, device="cpu", dtype=dtype)
    pupil = uipsf.ZernikePupil(grid, n_max=4)
    model = uipsf.ForwardModel(uipsf.VectorPSF(grid, pupil, roi, sigma_nm=(30.0, 30.0)), **model_kw)
    g = torch.Generator().manual_seed(seed)
    xyz = (torch.rand(n, 3, generator=g, dtype=torch.float64) - 0.5) * torch.tensor([200.0, 200.0, 300.0])
    phase = torch.zeros(len(pupil.phase_modes))
    phase[:3] = torch.tensor([0.5, -0.3, 0.2])
    values = {"pupil.phase": phase, "psf.sigma": torch.tensor([40.0, 35.0]), "emitter.xyz": xyz,
              "emitter.photons": photons * (1 + torch.rand(n, generator=g, dtype=torch.float64)),
              "emitter.background": torch.full((n,), background)}
    b0 = template(n, uipsf.Batch.z_from_objective(torch.linspace(-600, 600, K)), roi, dtype=dtype)
    truth = parameters(model, b0, values, dtype=dtype)
    return model, truth, simulate(model, truth, b0, seed=seed + 1)


# A two-channel splitter on one chip, up–down: the secondary half 256 px below, turned by 0.4°,
# 0.3 % larger and a little projective.  Secondary → main chip pixels, SMAPpy's direction.
SPLIT_TRANSFORM = [[0.997 * 0.99997, -0.007, 3.2], [0.007, 0.997 * 0.99997, -255.4], [2e-6, -1e-6, 1.0]]


def two_channel_beads(optics, n=6, kind="biplane", dtype=torch.float64, seed=0, P=24, roi=(9, 9), K=9,
                      photons=8000.0, background=(10.0, 6.0), n_max=4, bead_radius_nm=0.0, mirrored=False,
                      z_range=600.0, phase2=(-0.2, 0.25, 0.1), transform=None, **model_kw):
    """(model, truth Params, Batch) of beads seen in two channels of one chip.

    ``kind`` "biplane": the same optics, the second focal plane 350 nm off, half the light each;
    "ratiometric": the second channel 30 nm redder, a 40 nm chromatic focal shift, 30 % of the light.
    The secondary channel has its own aberrations (``phase2``).  ``mirrored`` flips the secondary half
    across the split (y), SMAPpy's "up-down mirrored".  Beads lie over the upper half of the chip."""
    import numpy as np

    from uipsf.model import Channel
    from uipsf.model.channels import map_points
    from uipsf.simulate import parameters, simulate, template
    H = np.array(SPLIT_TRANSFORM if transform is None else transform, float)
    if mirrored:
        H = H @ np.array([[1.0, 0, 0], [0, -1, 767], [0, 0, 1]])          # secondary y → 767 − y first
    optics2 = optics if kind == "biplane" else uipsf.Optics(
        NA=optics.NA, wavelength_nm=optics.wavelength_nm + 30, n_imm=optics.n_imm, n_med=optics.n_med,
        pixel_size_nm=optics.pixel_size_nm)
    share, dz = (0.5, 350.0) if kind == "biplane" else (0.3, 40.0)
    channels = []
    for o in (optics, optics2):
        grid = uipsf.PupilGrid.build(o, P, device="cpu", dtype=dtype)
        psf = uipsf.VectorPSF(grid, uipsf.ZernikePupil(grid, n_max=n_max), roi, sigma_nm=(30.0, 30.0),
                              bead_radius_nm=bead_radius_nm)
        channels.append(Channel(psf, transform=H))
    model = uipsf.ForwardModel(channels, **model_kw)
    g = torch.Generator().manual_seed(seed)
    centre = torch.rand(n, 2, generator=g, dtype=torch.float64) * torch.tensor([400.0, 200.0]) + 40
    centre = centre.round()                                                     # main ROI centres, chip px
    xyz = (torch.rand(n, 3, generator=g, dtype=torch.float64) - 0.5) * torch.tensor([200.0, 200.0, 300.0])
    pos = centre + xyz[:, :2] / torch.tensor(optics.pixel_size_nm, dtype=torch.float64)
    second = map_points(torch.tensor(np.linalg.inv(H)), pos).round()            # where the partner is cut
    phase = torch.zeros(len(channels[0].psf.pupil.phase_modes))
    phase[:3] = torch.tensor([0.5, -0.3, 0.2])
    phase_2 = torch.zeros_like(phase)
    phase_2[:3] = torch.tensor(phase2)
    values = {"pupil.phase": phase, "psf.sigma": torch.tensor([40.0, 35.0]), "emitter.xyz": xyz,
              "ch1.pupil.phase": phase_2, "ch1.psf.sigma": torch.tensor([30.0, 38.0]),
              "ch1.share": share, "ch1.dz": dz,
              "emitter.photons": photons * (1 + torch.rand(n, generator=g, dtype=torch.float64)),
              "emitter.background": torch.tensor(background, dtype=torch.float64).expand(n, 2)}
    b0 = template(n, uipsf.Batch.z_from_objective(torch.linspace(-z_range, z_range, K)), roi, dtype=dtype,
                  channels=2)
    b0.roi_xy = torch.stack([centre, second], 1).to(dtype)
    truth = parameters(model, b0, values, dtype=dtype)
    return model, truth, simulate(model, truth, b0, seed=seed + 1)
