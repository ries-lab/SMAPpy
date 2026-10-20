"""The astigmatic start (`uipsf.model.astig`): Noll 5 and 6 and each bead's z read from elliptical Gaussians
through focus, matched to the model's own PSFs, come back from simulated beads."""
import numpy as np
import pytest
import torch

import uipsf
from uipsf.model.astig import astig_start, channel_probe, gaussian_start
from uipsf.simulate import parameters, simulate, template

OPTICS = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.335, pixel_size_nm=(127, 116))


def _beads(phase: dict, n=8, seed=0, photons=2e4, background=30.0):
    grid = uipsf.PupilGrid.build(OPTICS, 48, device="cpu")
    psf = uipsf.VectorPSF(grid, uipsf.ZernikePupil(grid, n_max=4), (17, 17), sigma_nm=(30.0, 30.0))
    model = uipsf.ForwardModel(psf)
    rng = np.random.default_rng(seed)
    z = uipsf.Batch.z_from_objective(torch.linspace(-800, 800, 33))
    modes = psf.pupil.phase_modes
    values = {"pupil.phase": torch.tensor([phase.get(j, 0.0) for j in modes], dtype=torch.float32),
              "emitter.xyz": torch.tensor(np.c_[rng.uniform(-60, 60, (n, 2)), rng.uniform(-200, 200, n)],
                                          dtype=torch.float32),
              "emitter.photons": torch.full((n,), photons), "emitter.background": torch.full((n,), background)}
    b0 = template(n, z, psf.roi_shape)
    truth = parameters(model, b0, values)
    return model, truth, simulate(model, truth, b0, seed=seed + 1)


@pytest.mark.parametrize("method", ["gauss_xy", pytest.param("fit", marks=pytest.mark.slow)])
def test_vertical_astigmatism_and_every_beads_z_come_back_from_gaussians_through_focus(method):
    model, truth, batch = _beads({6: 0.6})
    r = astig_start(batch.data[:, 0], batch.z_nm, OPTICS.pixel_size_nm, channel_probe(model.psf, batch.z_nm),
                    method=method)
    assert r["a6"] == pytest.approx(0.6, abs=0.06)
    assert abs(r["a5"]) < 0.05
    dz = r["z"] - truth.physical()["emitter.xyz"][:, 2].double()
    assert dz.abs().max() < 30                                                  # nm


@pytest.mark.slow
def test_only_the_angled_gaussian_sees_oblique_astigmatism():
    model, truth, batch = _beads({5: 0.6})
    probe = channel_probe(model.psf, batch.z_nm)
    angled = astig_start(batch.data[:, 0], batch.z_nm, OPTICS.pixel_size_nm, probe, method="fit")
    aligned = astig_start(batch.data[:, 0], batch.z_nm, OPTICS.pixel_size_nm, probe, method="gauss_xy")
    assert angled["a5"] == pytest.approx(0.6, abs=0.06)
    assert aligned["a5"] == 0.0


def test_the_sign_of_the_astigmatism_is_the_models():
    """A positive Noll 6 stretches the spot along y above focus (docs/uipsf/conventions.md): read back
    positive, not mirrored."""
    model, truth, batch = _beads({6: -0.5})
    r = astig_start(batch.data[:, 0], batch.z_nm, OPTICS.pixel_size_nm, channel_probe(model.psf, batch.z_nm))
    assert r["a6"] == pytest.approx(-0.5, abs=0.06)


def test_the_gaussian_start_puts_every_bead_where_the_simulation_did():
    model, truth, batch = _beads({6: 0.6})
    g = gaussian_start(model, batch)
    got, true = g["emitters"]["xyz"].double(), truth.physical()["emitter.xyz"].double()
    assert (got[:, :2] - true[:, :2]).abs().max() < 15                         # nm
    assert (got[:, 2] - true[:, 2]).abs().max() < 30
    assert g["shared"]["pupil.phase"][model.psf.pupil.phase_modes.index(6)] == pytest.approx(0.6, abs=0.06)


def test_a_batchs_own_start_is_where_its_beads_start_in_any_subset():
    from dataclasses import replace
    model, truth, batch = _beads({6: 0.6}, n=4)
    xyz = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    start = dict(xyz=xyz, photons=torch.full((4,), 5e3), background=torch.full((4,), 7.0))
    sub = replace(batch, start=start).subset(torch.tensor([3, 1]))
    specs = {s.name: s for s in model.specs(sub)}
    assert torch.equal(torch.as_tensor(specs["emitter.xyz"].init), xyz[[3, 1]])
