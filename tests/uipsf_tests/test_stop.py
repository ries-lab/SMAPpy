"""*Stop* ends a learning at once, inside an iteration too (`solve.stop`): one Gauss–Newton matrix of real beads
took minutes, and the check between iterations alone let a stopped run go on as long."""
import threading

import pytest
import torch

import uipsf
from uipsf.noise import Poisson
from uipsf.solve import GaussNewtonLBFGS, LevenbergMarquardt, Problem
from uipsf.workflows import learn_beads
from uipsf_tests.conftest import small_beads

OPTICS = uipsf.Optics(NA=1.43, wavelength_nm=680, n_imm=1.518, n_med=1.33, pixel_size_nm=(110, 110))


@pytest.mark.parametrize("solver", [LevenbergMarquardt(), GaussNewtonLBFGS()])
def test_a_stop_ends_the_fit_between_two_chunks_of_one_matrix(solver):
    model, truth, batch = small_beads(OPTICS, n=6)
    problem = Problem(model, Poisson(), batch, chunk_size=1)
    asked = []

    def cancel():                       # pressed while the first matrix is being built, after two beads
        asked.append(1)
        return len(asked) > 2
    res = solver.solve(problem, truth, cancel=cancel)
    assert res.stop_reason == "cancelled" and res.iterations == 0 and not res.converged
    assert len(asked) == 3              # asked before each chunk: the third bead was never computed
    assert res.params is truth and torch.isnan(res.std["pupil.phase"]).all()
    assert problem.cancel is None       # the problem is the solver's only while it runs
    assert LevenbergMarquardt(max_iter=1).solve(problem, truth).stop_reason != "cancelled"


@pytest.mark.parametrize("stage", [1, 2])
def test_a_stop_ends_the_learning_in_the_stage_it_was_pressed_in(stage):
    model, _, batch = small_beads(OPTICS, n=6)
    stop, seen = threading.Event(), []

    def progress(entry):
        seen.append(entry["stage"])
        if entry["stage"] == stage:
            stop.set()
    cal = learn_beads(model, batch, first_order=2, progress=progress, cancel=stop)
    assert cal.result.stop_reason == "cancelled"
    assert max(seen) == stage                     # nothing after the stage it was stopped in
    assert cal.atlas is None and cal.heldout_report is None       # neither scored nor drawn: minutes on real beads
    assert len(cal.beads.rows) == 6
    if stage == 1:
        assert cal.beads.used == list(range(6))   # no bead judged on a fit that was stopped
