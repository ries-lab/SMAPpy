import dataclasses
import os
import sys

import pytest
import torch

import uipsf
from uipsf.model import Bleaching, ForwardModel
from uipsf.noise import Poisson
from uipsf.params import Gaussian, Params, ParamSpec, Softplus
from uipsf.simulate import parameters, simulate, template
from uipsf.solve import BlockLM, GaussNewtonCG, GaussNewtonLBFGS, LevenbergMarquardt, Problem, schur_solve
from uipsf_tests.conftest import small_beads

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "benchmarks", "uipsf", "synthetic"))
from beads import rescale  # noqa: E402

f64 = torch.float64


class LeastSquares:
    """A Gaussian noise model of unit variance: the solver on a linear problem is plain least squares."""

    def deviance(self, mu, data, s=0.0, mask=None):
        return (mu - data) ** 2

    def derivatives(self, mu, data, s=0.0, mask=None):
        return mu - data, torch.ones_like(mu)


class Linear:
    """μ_e = X_s·θ_s + X_e[e]·θ_e: shared and per-emitter blocks, as in a bead fit (one chunk)."""

    def __init__(self, Xs, Xe):
        self.Xs, self.Xe = Xs, Xe

    def specs(self, batch, start=None):
        return [ParamSpec("theta", (self.Xs.shape[1],)), ParamSpec("emitter.theta", (self.Xe.shape[2],), "emitter")]

    def __call__(self, p, batch):
        mu = self.Xs @ p["theta"] + torch.einsum("emk,ek->em", self.Xe, p["emitter.theta"])
        return mu[:, None, None, None, :]


def _linear(E=4, M=30, Ps=3, Pe=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    Xs, Xe = torch.randn(M, Ps, generator=g, dtype=f64), torch.randn(E, M, Pe, generator=g, dtype=f64)
    y = torch.randn(E, M, generator=g, dtype=f64)
    batch = uipsf.Batch(data=y[:, None, None, None, :], z_nm=torch.zeros(1, dtype=f64))
    A = torch.zeros(E * M, Ps + E * Pe, dtype=f64)                 # the dense problem, for comparison
    for e in range(E):
        A[e * M:(e + 1) * M, :Ps] = Xs
        A[e * M:(e + 1) * M, Ps + e * Pe:Ps + (e + 1) * Pe] = Xe[e]
    solution = torch.linalg.lstsq(A, y.reshape(-1, 1)).solution[:, 0]
    return Linear(Xs, Xe), LeastSquares(), batch, solution


def test_lm_solves_linear_least_squares_in_one_step():
    model, noise, batch, solution = _linear()
    params = Params(model.specs(batch), n_emitters=4, dtype=f64)
    result = LevenbergMarquardt(max_iter=1, damping=0.0).solve(Problem(model, noise, batch), params)
    got = torch.cat([result.params.shared_vector(), result.params.emitter_matrix().reshape(-1)])
    assert result.history[1]["accepted"]
    assert torch.allclose(got, solution, atol=1e-10)


def test_the_schur_solve_equals_the_dense_solve():
    model, noise, batch, _ = _linear(E=5, Ps=4, Pe=3, seed=3)
    params = Params(model.specs(batch), n_emitters=5, dtype=f64)
    ne = Problem(model, noise, batch).normal_equations(params)
    E, Ps, Pe = 5, 4, 3
    H = torch.zeros(Ps + E * Pe, Ps + E * Pe, dtype=f64)
    H[:Ps, :Ps] = ne.H_ss
    for e in range(E):
        sl = slice(Ps + e * Pe, Ps + (e + 1) * Pe)
        H[:Ps, sl], H[sl, :Ps], H[sl, sl] = ne.H_se[e], ne.H_se[e].T, ne.H_ee[e]
    g = torch.cat([ne.g_s, ne.g_e.reshape(-1)])
    D = torch.rand(Ps + E * Pe, dtype=f64) + 0.5
    lam = 0.3
    dense = torch.linalg.solve(H + lam * torch.diag(D), -g)
    ds, de = schur_solve(ne, lam, D[:Ps], D[Ps:].reshape(E, Pe))
    assert torch.allclose(torch.cat([ds, de.reshape(-1)]), dense, atol=1e-12)


def _with_priors(params, priors):
    specs = [ParamSpec(s.name, s.shape, s.scope, s.init, s.transform, priors.get(s.name, s.prior), s.fixed)
             for s in params.specs.values()]
    return Params(specs, params.n_emitters, dtype=f64, values=params.physical())


def test_priors_add_exactly_to_the_objective_gradient_and_normal_equations(optics):
    model, truth, batch = small_beads(optics, n=3)
    problem = Problem(model, Poisson(), batch)
    plain = truth
    with_p = _with_priors(truth, {"psf.sigma": Gaussian(30.0, 4.0), "emitter.xyz": Gaussian(0.0, 50.0)})
    a, b = problem.normal_equations(plain), problem.normal_equations(with_p)
    phys = truth.physical()
    assert abs((b.objective - a.objective) - float(with_p.prior_penalty())) < 1e-9
    i = dict((n, (lo, hi)) for n, lo, hi in truth.layout("shared"))["psf.sigma"]
    j = dict((n, (lo, hi)) for n, lo, hi in truth.layout("emitter"))["emitter.xyz"]
    dH_s = torch.zeros_like(a.H_ss)
    dH_s[range(*i), range(*i)] = 1 / 4.0 ** 2
    assert torch.allclose(b.H_ss - a.H_ss, dH_s, atol=1e-12)
    assert torch.allclose((b.g_s - a.g_s)[slice(*i)], (phys["psf.sigma"] - 30.0) / 16.0, atol=1e-12)
    dH_e = torch.zeros_like(a.H_ee)
    dH_e[:, range(*j), range(*j)] = 1 / 50.0 ** 2
    assert torch.allclose(b.H_ee - a.H_ee, dH_e, atol=1e-12)
    assert torch.allclose((b.g_e - a.g_e)[:, slice(*j)], phys["emitter.xyz"] / 2500.0, atol=1e-12)
    assert torch.equal(b.H_se, a.H_se)


def test_value_and_grad_equals_the_normal_equations_gradient_for_any_chunking(optics):
    model, truth, batch = small_beads(optics, n=3)
    p = _with_priors(truth, {"psf.sigma": Gaussian(30.0, 4.0), "emitter.xyz": Gaussian(0.0, 50.0)})
    start = p.replace(shared=p.shared_vector() + 0.01, emitter=p.emitter_matrix() * 1.01)
    ne = Problem(model, Poisson(), batch).normal_equations(start)
    for size in (1, 3):
        F, g_s, g_e = Problem(model, Poisson(), batch, chunk_size=size).value_and_grad(start)
        assert abs(F - ne.objective) < 1e-10 * ne.objective
        assert torch.allclose(g_s, ne.g_s, rtol=1e-9, atol=1e-9)
        assert torch.allclose(g_e, ne.g_e, rtol=1e-9, atol=1e-9)


def test_value_and_grad_is_the_gradient_where_the_pupil_magnitude_dips_below_zero(optics):
    # torch.polar's backward flips the gradient's sign where its magnitude is negative (torch 2.14), and
    # 1 + Σ c_j Z_j is, at the rim, once the coefficients reach -0.1 to -0.2: L-BFGS's steps then failed
    # and its damping ratcheted up until the step test stopped it, 1400 nats above the optimum (roadmap 4).
    model, truth, batch = small_beads(optics, n=3)
    values = truth.physical()
    values["pupil.magnitude"] = torch.linspace(-0.25, 0.05, len(values["pupil.magnitude"]), dtype=f64)
    magnitude, _ = model.psf.pupil(values)
    assert (magnitude[model.psf.weight > 0] < 0).any()
    p = Params(model.specs(batch), n_emitters=3, dtype=f64, values=values)
    ne = Problem(model, Poisson(), batch).normal_equations(p)
    _, g_s, g_e = Problem(model, Poisson(), batch).value_and_grad(p)
    assert torch.allclose(g_s, ne.g_s, rtol=1e-9, atol=1e-6)
    assert torch.allclose(g_e, ne.g_e, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("bead_radius_nm", [0.0, 50.0])
def test_the_pupil_linearisation_gives_the_jvps_normal_equations(optics, bead_radius_nm):
    model, truth, batch = small_beads(optics, n=3)
    model.psf.bead_radius_nm = bead_radius_nm
    p = truth.replace(shared=truth.shared_vector() + 0.05)
    a = Problem(model, Poisson(), batch, linearise=False).normal_equations(p)
    b = Problem(model, Poisson(), batch, tangents=3).normal_equations(p)
    for k in ("H_ss", "H_se", "H_ee", "g_s", "g_e"):
        assert torch.allclose(getattr(b, k), getattr(a, k), rtol=1e-12, atol=1e-9), k


def test_the_gauss_newton_operator_multiplies_by_the_normal_equations_matrix(optics):
    model, truth, batch = small_beads(optics, n=3)
    p = _with_priors(truth, {"psf.sigma": Gaussian(30.0, 4.0), "emitter.xyz": Gaussian(0.0, 50.0)})
    p = p.replace(shared=p.shared_vector() + 0.02)
    for size in (1, 3):
        problem = Problem(model, Poisson(), batch, chunk_size=size)
        ne, product = problem.normal_equations(p), problem.gauss_newton_operator(p)
        g = torch.Generator().manual_seed(1)
        v_s, v_e = torch.randn(ne.g_s.shape, generator=g, dtype=f64), torch.randn(ne.g_e.shape, generator=g, dtype=f64)
        Hv_s, Hv_e = product(v_s, v_e)
        assert torch.allclose(Hv_s, ne.H_ss @ v_s + torch.einsum("eij,ej->i", ne.H_se, v_e), rtol=1e-9, atol=1e-9)
        assert torch.allclose(Hv_e, torch.einsum("eji,j->ei", ne.H_se, v_s) + torch.einsum("eij,ej->ei", ne.H_ee, v_e),
                              rtol=1e-9, atol=1e-9)



def test_the_blurs_curvature_is_the_objectives_where_gauss_newton_alone_has_none(optics):
    # docs/4pi.md §21: σ enters only squared, so Gauss–Newton's curvature of it vanishes as σ → 0 while the
    # objective's does not; on the 4Pi-SMS beads, whose σ fell to 0, the steps ∝ 1/σ stalled 7 fits of 12.
    model, truth, batch = small_beads(optics, n=3)
    values = truth.physical()
    values["psf.sigma"] = torch.zeros(2, dtype=f64)
    batch = simulate(model, parameters(model, batch, values, dtype=f64), batch, seed=3)
    problem = Problem(model, Poisson(), batch)
    i = dict((n, lo) for n, lo, _ in truth.layout("shared"))["psf.sigma"]
    for sigma in (5.0, 1.0):
        values["psf.sigma"] = torch.full((2,), sigma, dtype=f64)
        p = parameters(model, batch, values, dtype=f64)
        h, step = problem.normal_equations(p).H_ss[i, i], torch.zeros(p.shared_vector().numel(), dtype=f64)
        step[i] = 1e-3 * sigma
        curvature = (problem.value_and_grad(p.replace(shared=p.shared_vector() + step))[1][i]
                     - problem.value_and_grad(p.replace(shared=p.shared_vector() - step))[1][i]) / (2 * step[i])
        assert abs(h - curvature) < 0.03 * curvature
        operator = problem.gauss_newton_operator(p, trace=False)
        unit = torch.zeros_like(step)
        unit[i] = 1.0
        Hv_s, _ = operator(unit, torch.zeros(p.n_emitters, p.emitter_matrix().shape[1], dtype=f64))
        assert abs(Hv_s[i] - h) < 1e-9 * h

@pytest.mark.parametrize("parts", ["per-observation", "per-slice", "planar"])
def test_the_emitter_linearisation_gives_the_jvps_normal_equations_for_every_component(optics, parts):
    from uipsf.model import PerObservation, PerSliceFactor, Planar
    kw = {"per-observation": dict(intensity=PerObservation()), "per-slice": dict(intensity=PerSliceFactor())}
    model, truth, batch = small_beads(optics, n=3, **kw.get(parts, {}))
    if parts == "planar":
        model.background = Planar((127.0, 116.0))
        truth = Params(model.specs(batch), 3, dtype=f64, values=truth.physical())
    p = truth.replace(shared=truth.shared_vector() + 0.05, emitter=truth.emitter_matrix() * 1.02 + 1e-3)
    a = Problem(model, Poisson(), batch, linearise=False).normal_equations(p)
    b = Problem(model, Poisson(), batch).normal_equations(p)
    for k in ("H_ss", "H_se", "H_ee", "g_s", "g_e"):
        assert torch.allclose(getattr(b, k), getattr(a, k), rtol=1e-10, atol=1e-8), k


def test_rescaling_a_parameter_leaves_the_iterates_unchanged(optics):
    # The key regression (plan §9 M2): v1 needs hand-set weights because its solver depends on units.
    model, truth, batch = small_beads(optics, n=3)
    problem = Problem(model, Poisson(), batch)
    start = Params(model.specs(batch), n_emitters=3, dtype=f64)
    lm = LevenbergMarquardt(max_iter=5)
    ref = lm.solve(problem, start)
    for factors in ({"emitter.photons": 1e3, "emitter.background": 1e-2},
                    {"pupil.phase": 10.0, "emitter.xyz": 1e-3, "psf.sigma": 1e2, "pupil.magnitude": 1e-1}):
        res = lm.solve(problem, rescale(start, factors))
        for h0, h1 in zip(ref.history, res.history, strict=True):
            assert abs(h0["objective"] - h1["objective"]) < 1e-8 * abs(h0["objective"])
            assert h0.get("accepted") == h1.get("accepted")
        a, b = ref.params.physical(), res.params.physical()
        for k in a:
            assert torch.allclose(a[k], b[k], rtol=1e-7, atol=1e-9), k


@pytest.mark.parametrize("solver", [GaussNewtonLBFGS(), GaussNewtonLBFGS(memory=0), GaussNewtonCG(trace=False)],
                         ids=["lbfgs", "chord", "cg"])
def test_the_cheaper_solvers_reach_lms_optimum_whatever_the_units(optics, solver):
    # Plan §9 M7: cheaper steps must not bring back v1's dependence on units (review, self.weight).
    model, truth, batch = small_beads(optics, n=3)
    problem = Problem(model, Poisson(), batch)
    start = Params(model.specs(batch), n_emitters=3, dtype=f64)
    ref = LevenbergMarquardt().solve(problem, start)
    res = solver.solve(problem, start)
    assert res.converged and abs(res.objective - ref.objective) < 0.01
    for k in ref.std:
        a, b = ref.params.physical()[k], res.params.physical()[k]
        assert ((a - b).abs() / ref.std[k]).max() < 0.05, k
    scaled = solver.solve(problem, rescale(start, {"pupil.phase": 10.0, "emitter.xyz": 1e-3, "psf.sigma": 1e2,
                                                   "emitter.photons": 1e3}))
    for h0, h1 in zip(res.history, scaled.history, strict=True):
        assert abs(h0["objective"] - h1["objective"]) < 1e-8 * abs(h0["objective"])


class WithLevel(Linear):
    """`Linear` plus a positive level per emitter (a background), through softplus."""

    def specs(self, batch, start=None):
        return super().specs(batch) + [ParamSpec("emitter.level", (), "emitter", init=1e-3, transform=Softplus())]

    def __call__(self, p, batch):
        return super().__call__(p, batch) + p["emitter.level"][:, None, None, None, None]


def test_a_step_into_softplus_flat_tail_holds_back_that_parameter_only():
    # Roadmap item 4: three Zenodo 2ch backgrounds started at 10⁻³ and every step drove them further into
    # softplus' tail; shortening the whole step until it did not held every parameter of the fit to
    # α ≈ 2·10⁻⁴, and the step test then stopped stage 1, 22 M nats up.  Here the data ask for levels below
    # zero: their steps are shortened, and the rest of the problem moves at the full step.
    lin, noise, batch, _ = _linear()
    model = WithLevel(lin.Xs, lin.Xe)
    batch = dataclasses.replace(batch, data=batch.data - 1.0)
    params = Params(model.specs(batch), n_emitters=4, dtype=f64)
    first = GaussNewtonLBFGS(max_iter=1).solve(Problem(model, noise, batch), params)
    best = LevenbergMarquardt(damping=0.0).solve(Problem(lin, noise, batch),
                                                 Params(lin.specs(batch), n_emitters=4, dtype=f64))
    F0, F1 = first.history[0]["objective"], first.history[1]["objective"]
    assert first.history[1]["alpha"] == 1.0
    assert F0 - F1 > 0.2 * (F0 - best.objective)
    assert (first.params.physical()["emitter.level"] > 0).all()


def test_chunk_size_changes_neither_the_sums_nor_the_fit(optics):
    # v1's results depend on batch_size (review F1); here a chunk is only a unit of memory.
    model, truth, batch = small_beads(optics, n=5)
    start = Params(model.specs(batch), n_emitters=5, dtype=f64)
    ref = Problem(model, Poisson(), batch, chunk_size=5).normal_equations(start)
    for size in (1, 2, 3):
        ne = Problem(model, Poisson(), batch, chunk_size=size, tangents=size).normal_equations(start)
        assert abs(ne.objective - ref.objective) < 1e-10 * ref.objective
        for k in ("H_ss", "H_se", "H_ee", "g_s", "g_e"):
            assert torch.allclose(getattr(ne, k), getattr(ref, k), rtol=1e-11, atol=1e-11), k
    # The whole fit in float32, as it runs on a GPU: the same answer to float32 round-off.
    model32, truth32, batch32 = small_beads(optics, n=4, dtype=torch.float32)
    fits = [LevenbergMarquardt().solve(Problem(model32, Poisson(), batch32, chunk_size=c), truth32)
            for c in (1, 4)]
    a = fits[-1].params.physical()
    for f in fits[:-1]:
        b = f.params.physical()
        for k in fits[-1].std:
            assert ((a[k] - b[k]).double().abs() / fits[-1].std[k]).max() < 0.01, k


def test_tangent_batches_change_neither_the_normal_equations_nor_the_fit(optics):
    # v1's results depended on batch_size (review F1).  In v2 how many Jacobian columns are computed
    # together is only memory: float64 sums are equal, and float32 fits of both solvers agree.
    model, truth, batch = small_beads(optics, n=4)
    start = Params(model.specs(batch), n_emitters=4, dtype=f64)
    ref = Problem(model, Poisson(), batch, tangents=1).normal_equations(start)
    for t in (4, 64):
        ne = Problem(model, Poisson(), batch, tangents=t).normal_equations(start)
        for k in ("H_ss", "H_se", "H_ee", "g_s", "g_e"):
            assert torch.allclose(getattr(ne, k), getattr(ref, k), rtol=1e-11, atol=1e-11), k
    model32, truth32, batch32 = small_beads(optics, n=4, dtype=torch.float32)
    # L-BFGS's gradient is summed per chunk in float32, so the fits stop at different points within its
    # tolerance: a predicted gain below 10⁻³ nats is up to √(2·10⁻³) ≈ 0.045 standard errors from the optimum
    # (0.005–0.007 on a Mac, 0.0103 on Windows' CI).  LM's chunks change nothing it sums.
    for solver, bound in ((LevenbergMarquardt(), 0.01), (GaussNewtonLBFGS(), 0.05)):
        fits = [solver.solve(Problem(model32, Poisson(), batch32, chunk_size=c, tangents=t), truth32)
                for c, t in ((4, 64), (1, 1), (2, 4))]
        a = fits[0].params.physical()
        for f in fits[1:]:
            b = f.params.physical()
            for k in fits[0].std:
                assert ((a[k] - b[k]).double().abs() / fits[0].std[k]).max() < bound, (type(solver).__name__, k)


@pytest.fixture(scope="module")
def localisations():
    """200 single-frame astigmatic emitters localised with the pupil known: the localisation problem."""
    optics = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.516, n_med=1.335, pixel_size_nm=(127, 116))
    E, roi = 200, (11, 11)
    grid = uipsf.PupilGrid.build(optics, 32, device="cpu", dtype=f64)
    pupil = uipsf.ZernikePupil(grid, n_max=4)
    model = ForwardModel(uipsf.VectorPSF(grid, pupil, roi, sigma_nm=(30.0, 30.0)), intensity=Bleaching(order=0))
    g = torch.Generator().manual_seed(5)
    phase = torch.zeros(len(pupil.phase_modes), dtype=f64)
    phase[pupil.phase_modes.index(6)] = 0.6
    xyz = (torch.rand(E, 3, generator=g, dtype=f64) - 0.5) * torch.tensor([120.0, 120.0, 600.0], dtype=f64)
    values = {"pupil.phase": phase, "psf.sigma": torch.tensor([35.0, 35.0], dtype=f64), "emitter.xyz": xyz,
              "emitter.photons": torch.full((E,), 3000.0, dtype=f64),
              "emitter.background": torch.full((E,), 10.0, dtype=f64)}
    b0 = template(E, torch.zeros(1), roi, dtype=f64)
    truth = parameters(model, b0, values, dtype=f64)
    batch = simulate(model, truth, b0, seed=6)
    start = Params(model.specs(batch), E, dtype=f64,
                   values={k: v for k, v in values.items() if k.startswith(("pupil", "psf"))})
    start = start.fix("pupil.phase", "pupil.magnitude", "psf.sigma")
    problem = Problem(model, Poisson(), batch)
    return LevenbergMarquardt().solve(problem, start), truth, problem


def test_the_crlb_matches_the_spread_of_200_simulated_localisations(localisations):
    result, truth, _ = localisations
    assert result.converged, result.stop_reason
    fit, tru = result.params.physical(), truth.physical()
    zscore = (fit["emitter.xyz"] - tru["emitter.xyz"]) / result.std["emitter.xyz"]
    assert ((zscore.std(0) - 1).abs() < 0.1).all(), zscore.std(0)
    zp = (fit["emitter.photons"] - tru["emitter.photons"]) / result.std["emitter.photons"]
    assert abs(zp.std().item() - 1) < 0.1


def test_block_lm_localises_every_emitter_where_lm_does(localisations):
    # One damping and stop per emitter (solve/blocks.py): the same answer as the block LM.
    result, truth, problem = localisations
    start = Params(problem.model.specs(problem.batch), problem.batch.n_emitters, dtype=f64,
                   values={k: v for k, v in result.params.physical().items() if k.startswith(("pupil", "psf"))})
    start = start.fix("pupil.phase", "pupil.magnitude", "psf.sigma")
    blocks = BlockLM().solve(problem, start)
    assert blocks.converged
    a, b = result.params.physical(), blocks.params.physical()
    for k in ("emitter.xyz", "emitter.photons", "emitter.background"):
        assert ((a[k] - b[k]).abs() / result.std[k]).max() < 0.01, k
    assert torch.allclose(blocks.std["emitter.xyz"], result.std["emitter.xyz"], rtol=1e-3)


def test_the_deviance_of_a_matched_fit_is_one_per_degree_of_freedom(localisations):
    result, _, problem = localisations
    D = problem.deviance(result.params)
    M = D[0].numel()
    # Poisson's mean deviance at μ ≈ 10 photons is 1 + 1/(6μ); each fitted parameter removes 1/M.
    expected = (1 + 1 / 60) * (M - 5) / M
    assert abs(D.mean().item() / expected - 1) < 0.02


def test_every_stop_is_reported_and_a_cancelled_fit_stops_before_it_starts(optics):
    model, truth, batch = small_beads(optics, n=2)
    problem = Problem(model, Poisson(), batch)
    seen = []
    res = LevenbergMarquardt(max_iter=2).solve(problem, truth, progress=seen.append, cancel=lambda: True)
    assert res.stop_reason == "cancelled" and res.iterations == 0 and len(seen) == 1
    res = LevenbergMarquardt(max_iter=1, xtol=0.0, tol=0.0).solve(problem, truth)
    assert res.stop_reason == "max_iter" and not res.converged
    res = LevenbergMarquardt().solve(problem, truth)
    assert res.converged and res.stop_reason in ("decrease", "step", "gradient", "no decrease")


def test_a_parameter_without_information_gets_a_huge_crlb_not_a_crash():
    model, noise, batch, _ = _linear(E=3, Ps=2, Pe=2)
    model.Xe[1, :, 1] = 0.0                                  # emitter 1's second parameter does nothing
    params = Params(model.specs(batch), n_emitters=3, dtype=f64)
    res = LevenbergMarquardt(max_iter=3).solve(Problem(model, noise, batch), params)
    assert res.std["emitter.theta"][1, 1] > 1e5
    assert torch.isfinite(res.std["emitter.theta"][[0, 2]]).all() and (res.std["emitter.theta"][[0, 2]] < 1).all()
