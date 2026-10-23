"""The layered model with a tilted coverslip (`optics.layers`): what a tilt does and whether a fit finds it."""
import math

import h5py
import numpy as np
import pytest
import torch

import uipsf
from uipsf.export import build, describe, load_model, save_model
from uipsf.model import BeadStackGeometry, Bleaching, InSituGeometry
from uipsf.noise import Poisson
from uipsf.optics import layers
from uipsf.simulate import parameters, simulate, template
from uipsf.solve import GaussNewtonLBFGS, Problem
from uipsf.workflows.insitu import twin

f64 = torch.float64
OIL = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.335, pixel_size_nm=(127, 116))
WATER = uipsf.Optics(NA=1.2, wavelength_nm=680, n_imm=1.333, n_med=1.333, n_cov=1.52, pixel_size_nm=(110, 110))
PHASE = (0.1, 0.5, -0.1, 0.1, 0.1)          # Noll 5-9


def insitu(optics, n=8, tilt=None, fit_tilt=False, P=24, roi=(9, 9), heights=(0.0, 1500.0), focus=700.0,
           photons=5000.0, seed=0):
    """(model, truth Params, Batch): molecules at heights uniform in ``heights`` under a coverslip tilted by
    ``tilt`` (None: no tilt declared)."""
    grid = uipsf.PupilGrid.build(optics, P, device="cpu", dtype=f64)
    psf = uipsf.VectorPSF(grid, uipsf.ZernikePupil(grid, n_max=4), roi, sigma_nm=(30.0, 30.0))
    geometry = InSituGeometry(focus, fit_tilt=fit_tilt, tilt_rad=tilt)
    model = uipsf.ForwardModel(psf, geometry=geometry, intensity=Bleaching(order=0))
    g = torch.Generator().manual_seed(seed)
    xy = (torch.rand(n, 2, generator=g, dtype=f64) - 0.5) * 150
    h = heights[0] + (heights[1] - heights[0]) * (torch.arange(n, dtype=f64) + 0.5) / n
    phase = torch.zeros(len(psf.pupil.phase_modes), dtype=f64)
    phase[:len(PHASE)] = torch.tensor(PHASE, dtype=f64)
    values = {"pupil.phase": phase, "psf.sigma": torch.tensor([35.0, 35.0]), "geometry.focus": torch.tensor(focus),
              "emitter.xyz": torch.cat([xy, h[:, None]], 1), "emitter.background": torch.full((n,), 10.0, dtype=f64),
              "emitter.photons": photons * (1 + torch.rand(n, generator=g, dtype=f64))}
    if tilt is not None:
        values["geometry.tilt"] = torch.tensor(tilt, dtype=f64)
    b0 = template(n, torch.zeros(1, dtype=f64), roi, dtype=f64)
    truth = parameters(model, b0, values, dtype=f64)
    return model, truth, simulate(model, truth, b0, seed=seed + 1)


def beads(optics, n=6, tilt=None, fit_tilt=False, P=24, roi=(11, 11), K=11, photons=20000.0, n_max=4, seed=0):
    """(model, truth Params, Batch): bead stacks on the coverslip under a tilt."""
    grid = uipsf.PupilGrid.build(optics, P, device="cpu", dtype=f64)
    psf = uipsf.VectorPSF(grid, uipsf.ZernikePupil(grid, n_max=n_max), roi, sigma_nm=(30.0, 30.0))
    model = uipsf.ForwardModel(psf, geometry=BeadStackGeometry(fit_tilt=fit_tilt, tilt_rad=tilt))
    g = torch.Generator().manual_seed(seed)
    xyz = (torch.rand(n, 3, generator=g, dtype=f64) - 0.5) * torch.tensor([200.0, 200.0, 200.0], dtype=f64)
    phase = torch.zeros(len(psf.pupil.phase_modes), dtype=f64)
    phase[:3] = torch.tensor([0.4, -0.3, 0.1], dtype=f64)
    values = {"pupil.phase": phase, "psf.sigma": torch.tensor([35.0, 35.0]), "emitter.xyz": xyz,
              "emitter.photons": photons * (1 + torch.rand(n, generator=g, dtype=f64)),
              "emitter.background": torch.full((n,), 10.0, dtype=f64)}
    if tilt is not None:
        values["geometry.tilt"] = torch.tensor(tilt, dtype=f64)
    b0 = template(n, uipsf.Batch.z_from_objective(torch.linspace(-700, 700, K, dtype=f64)), roi, dtype=f64)
    truth = parameters(model, b0, values, dtype=f64)
    return model, truth, simulate(model, truth, b0, seed=seed + 1)


def shared_of(params):
    return {n: v for n, v in params.physical().items() if params.specs[n].scope == "shared"}


@pytest.mark.parametrize("optics", [OIL, WATER], ids=["oil", "water"])
def test_a_tilt_of_zero_is_todays_psf_in_situ_and_for_beads(optics):
    model, truth, batch = insitu(optics, n=3)
    tilted, _, _ = insitu(optics, n=3, tilt=(0.0, 0.0))
    p = truth.physical()
    q = dict(p, **{"geometry.tilt": torch.zeros(2, dtype=f64)})
    assert torch.allclose(tilted(q, batch), model(p, batch), rtol=1e-9, atol=1e-12)
    model, truth, batch = beads(optics, n=2)
    tilted, _, _ = beads(optics, n=2, tilt=(0.0, 0.0))
    p = truth.physical()
    q = dict(p, **{"geometry.tilt": torch.zeros(2, dtype=f64)})
    assert torch.allclose(tilted(q, batch), model(p, batch), rtol=1e-9, atol=1e-12)


def test_without_a_tilt_the_model_declares_none_and_its_wave_is_todays():
    model, truth, _ = insitu(OIL, n=2)
    assert "geometry.tilt" not in truth.specs
    assert not layers.layered(model.psf.grid, truth.physical())


def test_with_an_oil_objective_a_tilt_changes_nothing_for_beads_on_the_coverslip():
    model, truth, batch = beads(OIL, n=2, tilt=(0.03, -0.02))
    p = truth.physical()
    level = dict(p, **{"geometry.tilt": torch.zeros(2, dtype=f64)})
    assert torch.equal(model(p, batch), model(level, batch))


def test_with_an_oil_objective_a_tilt_bends_the_psf_of_molecules_the_more_the_deeper_they_are():
    model, truth, batch = insitu(OIL, n=6, tilt=(0.03, 0.0), heights=(0.0, 1500.0))
    p = truth.physical()
    level = dict(p, **{"geometry.tilt": torch.zeros(2, dtype=f64)})
    a, b = model(p, batch), model(level, batch)
    change = ((a - b).abs().flatten(1).sum(1) / b.flatten(1).sum(1)).tolist()
    assert change[0] < 0.02 and change[-1] > 5 * change[0] and change[-1] > 0.05


def test_a_tilt_gives_coma_along_its_axis_and_astigmatism_only_at_second_order():
    for optics, depth in ((OIL, 1000.0), (WATER, 0.0)):
        z1, s1 = layers.aberrations(optics, (math.radians(1.0), 0.0), depth, size=64)
        z2, _ = layers.aberrations(optics, (math.radians(2.0), 0.0), depth, size=64)
        zy, _ = layers.aberrations(optics, (0.0, math.radians(1.0)), depth, size=64)
        assert abs(z1[8]) > 0.1 and abs(z1[7]) < 1e-3                # coma along x (Noll 8), none along y
        assert abs(zy[7]) == pytest.approx(abs(z1[8]), rel=1e-3) and abs(zy[8]) < 1e-3
        assert abs(z1[6]) < 0.1 * abs(z1[8]) and abs(z1[5]) < 1e-3     # astigmatism small, along the tilt
    z1, _ = layers.aberrations(WATER, (math.radians(1.0), 0.0), size=64)
    z2, _ = layers.aberrations(WATER, (math.radians(2.0), 0.0), size=64)
    assert z2[6] == pytest.approx(4 * z1[6], rel=0.05)                # through the plate: quadratic


def test_a_tilt_towards_x_moves_a_molecule_above_the_glass_towards_x():
    # The apparent position lies on the interface's normal through the molecule, farther from the glass
    # (`docs/uipsf/conventions.md`): the normal leans towards +x for t_x > 0.
    _, (sx, sy) = layers.aberrations(OIL, (0.02, 0.0), 1000.0, size=64)
    assert sx > 5.0 and abs(sy) < 0.1


@pytest.mark.parametrize("which", ["in situ", "water beads"])
def test_the_emitter_linearisation_gives_the_jvps_normal_equations_with_a_tilt(which):
    if which == "in situ":
        model, truth, batch = insitu(OIL, n=3, tilt=(0.02, -0.01), fit_tilt=True, P=16, roi=(7, 7))
    else:
        model, truth, batch = beads(WATER, n=2, tilt=(0.01, 0.005), fit_tilt=True, P=16, roi=(7, 7), K=5)
    p = truth.replace(shared=truth.shared_vector() * 1.02 + 0.01, emitter=truth.emitter_matrix() * 1.02 + 1e-3)
    a = Problem(model, Poisson(), batch, linearise=False).normal_equations(p)
    b = Problem(model, Poisson(), batch).normal_equations(p)
    for k in ("H_ss", "H_se", "H_ee", "g_s", "g_e"):
        assert torch.allclose(getattr(b, k), getattr(a, k), rtol=1e-9, atol=1e-7), k


def test_a_tilt_freed_without_fit_tilt_is_differentiated_through_the_whole_model():
    model, truth, batch = insitu(OIL, n=2, tilt=(0.02, 0.0), P=16, roi=(7, 7))
    p = truth.free("geometry.tilt")
    assert "geometry.tilt" in model.psf_names()
    a = Problem(model, Poisson(), batch, linearise=False).normal_equations(p)
    b = Problem(model, Poisson(), batch).normal_equations(p)
    assert torch.allclose(b.H_ss, a.H_ss, rtol=1e-9, atol=1e-7) and float(b.H_ss.abs().max()) > 0


def fit_from_level(model, truth, batch):
    """The fit of every parameter from the truth, the tilt started at zero."""
    values = dict(truth.physical(), **{"geometry.tilt": torch.zeros(2, dtype=f64)})
    p0 = uipsf.Params(list(truth.specs.values()), truth.n_emitters, dtype=f64, values=values)
    return GaussNewtonLBFGS().solve(Problem(model, Poisson(), batch), p0)


def test_an_in_situ_fit_finds_the_tilt_from_molecules_at_several_depths():
    # Oil objective: the tilt is seen only through the glass–water interface, as coma growing with depth;
    # the pupil's own coma is learnt beside it.  ~1 s on one core.
    tilt = (0.035, -0.025)                                  # 2.0° and 1.4°
    model, truth, batch = insitu(OIL, n=24, tilt=tilt, fit_tilt=True, photons=8000.0)
    fit = fit_from_level(model, truth, batch)
    t, sd = fit.params.physical()["geometry.tilt"], fit.std["geometry.tilt"]
    assert torch.all(sd < 0.01)                              # well inside the prior's 0.05
    assert torch.all((t - torch.tensor(tilt, dtype=f64)).abs() < 4 * sd), (t, sd)


def test_a_bead_fit_with_a_water_objective_finds_the_tilt_through_the_coverslip():
    # Beads on the coverslip of a water objective: the tilted plate's coma, of a shape the pupil's modes up to
    # n = 4 do not make (its secondary coma, n = 5, among them).  ~0.6 s on one core.
    tilt = (0.012, -0.008)                                  # 0.7° and 0.5°
    model, truth, batch = beads(WATER, n=6, tilt=tilt, fit_tilt=True)
    fit = fit_from_level(model, truth, batch)
    t, sd = fit.params.physical()["geometry.tilt"], fit.std["geometry.tilt"]
    assert torch.all(sd < 0.005)
    assert torch.all((t - torch.tensor(tilt, dtype=f64)).abs() < 4 * sd), (t, sd)


def test_where_nothing_sees_the_tilt_it_stays_at_its_prior():
    # An oil objective and beads on the coverslip: the PSF does not depend on the tilt at all.
    model, truth, batch = beads(OIL, n=3, tilt=(0.0, 0.0), fit_tilt=True, P=16, roi=(7, 7), K=5)
    fit = GaussNewtonLBFGS(max_iter=5).solve(Problem(model, Poisson(), batch), truth)
    assert torch.equal(fit.params.physical()["geometry.tilt"], torch.zeros(2, dtype=f64))
    assert torch.allclose(fit.std["geometry.tilt"], torch.full((2,), 0.05, dtype=f64), rtol=1e-6)


def test_a_tilt_does_not_move_the_image():
    # Through the plate of a water objective the image moves by 0.7 µm per degree; the model takes that out.
    _, (sx, _) = layers.aberrations(WATER, (math.radians(1.0), 0.0), size=64)
    assert sx < -300.0
    grid = uipsf.PupilGrid.build(WATER, 32, device="cpu", dtype=f64)
    for c in layers.coefficients(grid, torch.tensor([0.02, -0.01], dtype=f64)):
        inside = grid.aperture > 0
        for k in (grid.kx[None, :].expand_as(c), grid.ky[:, None].expand_as(c)):
            assert abs(complex((c * k)[inside].sum())) < 1e-9 * float((c.abs() * k.abs())[inside].sum())


def test_the_twin_keeps_the_tilt():
    model, truth, _ = insitu(OIL, n=1, tilt=(0.02, -0.01), fit_tilt=True)
    shared = shared_of(truth)
    assert torch.equal(twin(model, shared)["geometry.tilt"], shared["geometry.tilt"])


def test_the_tilt_and_the_coverslip_go_into_the_description_and_older_files_have_neither():
    optics = uipsf.Optics(NA=1.2, wavelength_nm=680, n_imm=1.333, n_med=1.333, n_cov=1.52, d_cov_nm=165_000.0,
                          pixel_size_nm=(110, 110))
    model, truth, batch = beads(optics, n=2, tilt=(0.01, 0.0), fit_tilt=True, P=16, roi=(7, 7), K=5)
    config = describe(model)
    assert config["geometry"]["fit_tilt"] and config["channels"][0]["optics"]["d_cov_nm"] == 165_000.0
    again = build(config, device="cpu", dtype=f64)
    assert describe(again) == config and again.geometry.tilt_rad == (0.01, 0.0)
    p = truth.physical()
    assert torch.allclose(again(p, batch), model(p, batch), rtol=1e-12, atol=1e-14)
    plain, _, _ = beads(OIL, n=1)
    old = describe(plain)
    assert "fit_tilt" not in old["geometry"] and "d_cov_nm" not in old["channels"][0]["optics"]
    old_model = build(old, device="cpu")
    assert old_model.geometry.fit_tilt is False and old_model.psf.grid.optics.d_cov_nm == 170_000.0


def test_a_saved_tilted_model_is_version_3_and_loads_to_the_same_psf(tmp_path):
    from uipsf.workflows import learn_beads
    model, _, batch = beads(WATER, n=4, tilt=(0.01, 0.0), fit_tilt=True, P=16, roi=(7, 7), K=5)
    model = build(describe(model), device="cpu")           # float32, as the workflows make it
    batch = uipsf.Batch(data=batch.data.float(), z_nm=batch.z_nm.float())
    cal = learn_beads(model, batch, first_order=None, solver=GaussNewtonLBFGS(max_iter=3))
    path = tmp_path / "tilted.uipsf.h5"
    save_model(cal, path)
    with h5py.File(path) as f:
        assert f.attrs["version"] == 3
    m = load_model(path, device="cpu")
    assert torch.equal(m.shared["geometry.tilt"].cpu(), cal.shared()["geometry.tilt"].detach().cpu())
    xyz = torch.tensor([[0.0, 0.0, 0.0], [30.0, -20.0, 250.0]])
    assert torch.allclose(m.psf(xyz), cal.psf(xyz), atol=1e-7)
    assert np.isfinite(float(m.shared["geometry.tilt"][0]))
