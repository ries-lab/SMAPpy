"""Stopping a fit from outside: the window's *Stop*.

A solver checked its ``cancel`` only between iterations, and on real beads one iteration -- a Gauss–Newton
matrix over every chunk -- took minutes (M6, 19 beads of 100 slices), so *Stop* seemed to do nothing.  Now a
solver hands its ``cancel`` to the `Problem` for the time it runs (`stoppable`), the problem checks it before
every chunk (`Problem.chunks`) and raises `Cancelled`, and the solver returns at once, stopped: the parameters
it started from (a stopped fit is never kept: nothing is saved after it) and the iterations made so far.
"""
from __future__ import annotations

import functools
import math

import torch


class Cancelled(Exception):
    """Raised by `Problem.chunks` when the fit it serves was stopped."""


def _cancelled(cancel) -> bool:
    return cancel is not None and (cancel.is_set() if hasattr(cancel, "is_set") else bool(cancel()))


def stoppable(solve):
    """``solve(self, problem, params, progress=None, cancel=None)`` that ends as soon as ``cancel`` is set, even
    inside an iteration: a `FitResult` with ``stop_reason`` "cancelled" (`stopped`)."""
    @functools.wraps(solve)
    def run(self, problem, params, progress=None, cancel=None):
        history = []

        def heard(entry):
            history.append(entry)
            if progress:
                progress(entry)
        before = getattr(problem, "cancel", None)
        problem.cancel = cancel
        try:
            return solve(self, problem, params, progress=heard, cancel=cancel)
        except Cancelled:
            return stopped(params, history)
        finally:
            problem.cancel = before
    return run


def stopped(params, history):
    """The `FitResult` of a fit stopped inside an iteration: ``params`` (where it started), ``history``, no
    uncertainties (NaN)."""
    from .result import FitResult
    nan = float("nan")
    Ps = params.shared_vector().numel()
    std = {n: torch.full(params.specs[n].shape, nan, dtype=torch.float64) for n, _, _ in params.layout("shared")}
    std.update({n: torch.full((params.n_emitters, *params.specs[n].shape), nan, dtype=torch.float64)
                for n, _, _ in params.layout("emitter")})
    objective = history[-1].get("objective", nan) if history else nan
    return FitResult(params, list(history), "cancelled", False, objective if math.isfinite(objective) else nan,
                     torch.full((Ps, Ps), nan, dtype=torch.float64), std, nan, nan)
