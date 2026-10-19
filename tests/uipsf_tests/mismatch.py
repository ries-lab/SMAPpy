"""Simulated bead stacks with one piece of physics the default fit lacks (review §B), for the atlas tests."""
import torch

import uipsf
from uipsf.model import BeadStackGeometry, Constant, PerObservation, PerSliceFactor, Planar
from uipsf.simulate import parameters, simulate, template

KINDS = ("matched", "background ramp", "piezo scale", "laser", "speckle")
TRUE_PHASE_NOLL = {6: 0.5, 7: 0.15, 11: 0.2}


def mismatched(kind, n=12, seed=0, P=32, roi=(15, 15), K=21, z_range_nm=500.0, scale=1.05):
    """(default model, batch simulated with ``kind``)."""
    optics = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.335, pixel_size_nm=(127, 116))
    grid = uipsf.PupilGrid.build(optics, P, device="cpu")
    pupil = uipsf.ZernikePupil(grid, n_max=4)
    psf = uipsf.VectorPSF(grid, pupil, roi, sigma_nm=(30.0, 30.0))
    g = torch.Generator().manual_seed(seed)
    phase = torch.zeros(len(pupil.phase_modes))
    for j, v in TRUE_PHASE_NOLL.items():
        phase[pupil.phase_modes.index(j)] = v
    values = {"pupil.phase": phase, "psf.sigma": torch.tensor([40.0, 35.0]),
              "emitter.xyz": (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([150.0, 150.0, 200.0]),
              "emitter.photons": 3000 + 1000 * torch.rand(n, generator=g),
              "emitter.background": torch.full((n,), 20.0)}
    truth = dict(channels=psf)
    if kind == "background ramp":
        truth["background"] = Planar(optics.pixel_size_nm)
        values["emitter.background_slope"] = torch.tensor([0.004, -0.003]).expand(n, 2)
    elif kind == "piezo scale":
        truth["geometry"] = BeadStackGeometry(z_scale=True)
        values["geometry.z_scale"] = torch.tensor(scale)
    elif kind == "laser":
        truth["intensity"] = PerSliceFactor()
        values["intensity.slice_factor"] = 1 + 0.1 * torch.randn(K, generator=g)
    elif kind == "speckle":
        truth["intensity"] = PerObservation()
        values["emitter.photons_per_slice"] = values["emitter.photons"][:, None] * (
            1 + 0.1 * torch.randn(n, K, generator=g))
    z = uipsf.Batch.z_from_objective(torch.linspace(-z_range_nm, z_range_nm, K))
    b0 = template(n, z, roi)
    b0 = uipsf.Batch(data=b0.data, z_nm=b0.z_nm, field_xy=torch.rand(n, 2, generator=g) * 500)
    tm = uipsf.ForwardModel(**truth)
    batch = simulate(tm, parameters(tm, b0, values), b0, seed=seed + 1)
    batch = uipsf.Batch(data=batch.data, z_nm=batch.z_nm, s=batch.s, field_xy=b0.field_xy)
    return uipsf.ForwardModel(psf, background=Constant()), batch


def _true_phase():
    from uipsf.optics.zernike import zernike_modes
    modes = zernike_modes(4)
    phase = torch.zeros(len(modes))
    for j, v in TRUE_PHASE_NOLL.items():
        phase[modes.index(j)] = v
    return phase


TRUE_PHASE = _true_phase()
