"""The sample's refractive index learnt (``optics.n_med``, `VectorPSF.learn_n_med`, `optics.medium`)."""
import dataclasses

import pytest
import torch

import uipsf
from uipsf.model import ForwardModel, InSituGeometry
from uipsf.noise import Poisson
from uipsf.optics import medium
from uipsf.params import Params
from uipsf.simulate import parameters, simulate, template
from uipsf.solve import Problem

f64 = torch.float64


def _psf(optics, learn=False, kind="vector", analyser="unpolarised", P=32, roi=(11, 11), minimal=False):
    from types import SimpleNamespace

    from uipsf.model import make_psf, pupil_grid
    spec = SimpleNamespace(order=4, magnitude="minimal" if minimal else "free", saf_damping=True, blur="round",
                           blur_nm=30.0, pixel_blur=True, learn_n_med=learn)
    grid = pupil_grid(optics, spec, size=P, device="cpu", dtype=f64)
    return make_psf(optics, spec, roi, grid=grid, kind=kind, analyser=analyser)


@pytest.mark.parametrize("kind, analyser", [("vector", "unpolarised"), ("vector", "x"), ("scalar", "unpolarised")])
@pytest.mark.parametrize("n_med", [1.335, 1.516])
def test_the_maps_recomputed_at_the_typed_index_are_the_grids(optics, kind, analyser, n_med):
    """At the typed n_med the recomputed fields are the precomputed ones, but within a few smoothing widths of
    the critical angle (`optics.medium`): there the root is smoothed, and t·apodisation is formed without its
    0/0."""
    optics = dataclasses.replace(optics, n_med=n_med)
    psf = _psf(optics, kind=kind, analyser=analyser, P=64)
    F, _, _, weight, kz = psf.medium({"optics.n_med": torch.tensor(n_med, dtype=f64)})
    g = psf.grid
    a = 1 - (g.rho * optics.NA / n_med) ** 2
    far = (a.abs() > 4 * medium.eps(g)).expand_as(F)
    assert torch.allclose(F[far], psf.fields[far], rtol=1e-9, atol=1e-12)
    assert torch.allclose(kz[far[0]], g.kz_med[far[0]], rtol=1e-9, atol=1e-15)
    assert torch.isfinite(F.real).all() and torch.isfinite(F.imag).all()
    assert abs(float(weight.sum() / psf.energy) - 1) < 1e-2


def test_held_the_psf_and_its_parameters_are_unchanged(optics):
    """Off (the default) there is no ``optics.n_med`` and the PSF is the precomputed one; learnt and at the typed
    value, the PSF is the same to the smoothing of the critical angle."""
    held, learnt = _psf(optics), _psf(optics, learn=True)
    names = [s.name for s in held.specs()]
    assert "optics.n_med" not in names and [s.name for s in learnt.specs()] == names + ["optics.n_med"]
    spec = learnt.specs()[-1]
    assert float(spec.init) == optics.n_med and spec.scope == "shared"
    p = Params(held.specs(), dtype=f64).physical()
    xyz = torch.tensor([[[10.0, -20.0, z] for z in (-500.0, 0.0, 300.0)]], dtype=f64)
    I = held(p, xyz)
    q = dict(p, **{"optics.n_med": torch.tensor(optics.n_med, dtype=f64)})
    assert torch.allclose(learnt(q, xyz), I, rtol=0, atol=2e-4 * float(I.max()))
    assert torch.equal(learnt(p, xyz), I)                    # no n_med among the parameters: the grid's
    d = learnt.describe(q)
    assert d["n_medium"] == pytest.approx(optics.n_med) and d["n_medium_learnt"]
    assert held.describe(p)["n_medium"] == pytest.approx(optics.n_med) and "n_medium_learnt" not in held.describe(p)


def test_the_image_integrates_to_one_photon_at_any_index(optics):
    psf = _psf(optics, learn=True, P=64, roi=(41, 41))
    p = Params(psf.specs(), dtype=f64).physical()
    xyz = torch.zeros(1, 1, 3, dtype=f64)
    for n in (1.30, 1.335, 1.40, 1.52):
        total = float(psf(dict(p, **{"optics.n_med": torch.tensor(n, dtype=f64)}), xyz).sum())
        assert 0.85 < total <= 1.0 + 1e-9, n         # the rest falls outside the region


def test_the_index_is_differentiable_through_the_critical_angle(optics):
    """Moved across the NA, n_med changes which pupil samples are supercritical; the PSF's derivative stays finite
    and matches a finite difference (the root is smoothed, `optics.medium`)."""
    psf = _psf(optics, learn=True, minimal=True)
    p = Params(psf.specs(), dtype=f64).physical()
    p["pupil.ring_h"] = torch.tensor(50.0, dtype=f64)
    xyz = torch.tensor([[[0.0, 0.0, -200.0], [0.0, 0.0, 200.0]]], dtype=f64)
    for n in torch.linspace(1.38, 1.48, 7, dtype=f64):
        def image(v):
            return psf(dict(p, **{"optics.n_med": v}), xyz)
        _, d = torch.func.jvp(image, (n,), (torch.ones_like(n),))
        h = 1e-5
        fd = (image(n + h) - image(n - h)) / (2 * h)
        assert torch.isfinite(d).all()
        assert torch.allclose(d, fd, rtol=0, atol=1e-4 * float(fd.abs().max()) + 1e-9), float(n)


def _beads(optics, learn=True, n=3, minimal=False, two=False, radius=0.0):
    psf = _psf(optics, learn=learn, P=24, roi=(9, 9), minimal=minimal)
    psf.bead_radius_nm = radius
    if two:
        from uipsf.model import Channel
        from uipsf_tests.conftest import SPLIT_TRANSFORM
        psf2 = _psf(optics, learn=learn, P=24, roi=(9, 9), minimal=minimal)
        model = ForwardModel([Channel(psf, transform=SPLIT_TRANSFORM), Channel(psf2, transform=SPLIT_TRANSFORM)])
    else:
        model = ForwardModel(psf)
    g = torch.Generator().manual_seed(0)
    xyz = (torch.rand(n, 3, generator=g, dtype=f64) - 0.5) * torch.tensor([200.0, 200.0, 300.0])
    values = {"emitter.xyz": xyz, "emitter.photons": torch.full((n,), 5000.0, dtype=f64),
              "emitter.background": torch.full((n,) + ((2,) if two else ()), 10.0, dtype=f64),
              "optics.n_med": torch.tensor(1.36, dtype=f64)}
    b0 = template(n, uipsf.Batch.z_from_objective(torch.linspace(-600, 600, 7)), (9, 9), dtype=f64,
                  channels=2 if two else 1)
    if two:
        b0.roi_xy = torch.tensor([[[100.0, 60.0], [97.0, 316.0]]] * n, dtype=f64)
        values.update({"ch1.share": 0.5, "ch1.dz": 300.0})
    truth = parameters(model, b0, values, dtype=f64)
    return model, truth, simulate(model, truth, b0, seed=1)


@pytest.mark.parametrize("case", ["beads", "bead kernel", "minimal", "two channels"])
def test_the_linearised_normal_equations_are_the_jvps_with_the_index_learnt(optics, case):
    """The solver's linearised columns (`pupil_linearisation` at the learnt index's fields, n_med's own column by a
    JVP through the model, `psf_names`) give the JVPs' normal equations."""
    model, truth, batch = _beads(optics, minimal=case == "minimal", two=case == "two channels",
                                 radius=50.0 if case == "bead kernel" else 0.0)
    names = {s.name for s in model.specs(batch)}
    assert "optics.n_med" in names and "ch1.optics.n_med" not in names
    assert "optics.n_med" in model.psf_names()
    p = truth.replace(shared=truth.shared_vector() * 1.01 + 0.01, emitter=truth.emitter_matrix() * 1.02 + 1e-3)
    a = Problem(model, Poisson(), batch, linearise=False).normal_equations(p)
    b = Problem(model, Poisson(), batch, tangents=3).normal_equations(p)
    for k in ("H_ss", "H_se", "H_ee", "g_s", "g_e"):
        assert torch.allclose(getattr(b, k), getattr(a, k), rtol=1e-10, atol=1e-8), k


def test_the_linearised_normal_equations_are_the_jvps_in_situ_with_the_index_learnt(optics):
    psf = _psf(optics, learn=True, P=24, roi=(9, 9))
    model = ForwardModel(psf, geometry=InSituGeometry(500.0), intensity=uipsf.model.Bleaching(order=0))
    g = torch.Generator().manual_seed(0)
    n = 3
    values = {"geometry.focus": torch.tensor(500.0, dtype=f64),
              "emitter.xyz": torch.cat([(torch.rand(n, 2, generator=g, dtype=f64) - 0.5) * 150,
                                        torch.rand(n, 1, generator=g, dtype=f64) * 900], 1),
              "emitter.photons": torch.full((n,), 3000.0, dtype=f64),
              "emitter.background": torch.full((n,), 10.0, dtype=f64), "optics.n_med": torch.tensor(1.36)}
    b0 = template(n, torch.zeros(1, dtype=f64), (9, 9), dtype=f64)
    truth = parameters(model, b0, values, dtype=f64)
    batch = simulate(model, truth, b0, seed=1)
    p = truth.replace(shared=truth.shared_vector() * 1.01 + 0.01, emitter=truth.emitter_matrix() * 1.02 + 1e-3)
    a = Problem(model, Poisson(), batch, linearise=False).normal_equations(p)
    b = Problem(model, Poisson(), batch).normal_equations(p)
    for k in ("H_ss", "H_se", "H_ee", "g_s", "g_e"):
        assert torch.allclose(getattr(b, k), getattr(a, k), rtol=1e-10, atol=1e-8), k


def test_the_model_file_keeps_the_learnt_index(optics, tmp_path):
    from uipsf.export.model_file import build, describe
    model, truth, batch = _beads(optics)
    again = build(describe(model), device="cpu", dtype=f64)
    assert again.psf.learn_n_med and "optics.n_med" in {s.name for s in again.specs(batch)}


@pytest.mark.long
@pytest.mark.parametrize("truth", [1.36, 1.40])
def test_beads_give_their_sample_index_back_within_its_crlb(truth):
    """One objective's bead stacks drawn with the sample at ``truth`` -- below the NA, the minimal pupil's ring damped
    by h -- give n_med back from the typed 1.33 within 3 CRLB (*learn sample index*), γ and h beside it."""
    from uipsf.workflows import BeadSettings, PSFModelSettings, bead_model, calibrate_beads
    from uipsf_tests.test_minimal_pupil import GAMMA, OPTICS, RING_H, ROI, simulated
    settings = BeadSettings(psf=PSFModelSettings(zernike_order=4, learn_n_med=True), drift="none")
    model = bead_model(OPTICS, settings, ROI, z_max_nm=1000, device="cpu")
    beads = simulated(model, extra={"pupil.gamma": torch.tensor(GAMMA), "pupil.ring_h": torch.tensor(RING_H),
                                    "optics.n_med": torch.tensor(truth)})
    cal = calibrate_beads(beads, OPTICS, settings, device="cpu")
    shared, std = cal.shared(), cal.result.std
    n, s = float(shared["optics.n_med"]), float(std["optics.n_med"])
    print(f"n_med {n:.4f} ± {s:.4f} (truth {truth}), γ {float(shared['pupil.gamma']):.3f}")
    assert abs(n - truth) < 3 * s and s < 0.02, (n, s)
    assert abs(float(shared["pupil.gamma"]) - GAMMA) < 3 * float(std["pupil.gamma"]) + 0.01
